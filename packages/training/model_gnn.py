"""
model_gnn.py — CatanGNN: a graph network over the board structure.

Motivation (measured, not theoretical): the flat MLP plateaued ~20 points
below its MCTS teacher in Expert Iteration — it must memorise position-by-
position what a graph model can compute relationally ("good expansion spot",
"blockable road", "robber pressure") once and apply at any location.

Tokens
──────
The observation (encoding.ts, 1328 floats) already factors into per-element
blocks; we reshape it into 145 tokens + a global context:
    global   obs[   0:  68]               (hands, bank, phase, robber, LR/LA)
    vertex   obs[  68: 770] → [54, 13]    (owner, kind, port)
    tile     obs[ 770:1112] → [19, 18]    (terrain, token, robber)
    edge     obs[1112:1328] → [72,  3]    (road owner)

Each token type gets its own input projection plus a learned per-token
position embedding (the topology is fixed, so position carries meaning:
coastline vs centre, port locations). The global block is projected and
added to every token as context.

Message passing
───────────────
Dense normalised adjacency A [145,145] built from topology.json
(vertex–edge incidence, vertex–tile incidence, vertex–vertex via shared
edges, self-loops), applied as A @ h inside residual blocks:
    h ← h + ReLU(LN(W_self·h + W_nbr·(A·h)))
Dense 145×145 matmuls are cheap and export to ONNX trivially.

Heads (mirroring the action layout in encoding.ts)
──────
    0- 43  global actions          ← MLP(global context ‖ mean-pooled tokens)
   44- 97  buildSettlement ×54     ← linear on vertex tokens
   98-151  buildCity       ×54     ← linear on vertex tokens
  152-223  buildRoad       ×72     ← linear on edge tokens
  224-242  moveRobber(null)×19     ← linear on tile tokens
  243-261  moveRobber(steal)×19    ← linear on tile tokens
  262-280  playKnight(null)×19     ← linear on tile tokens
  281-299  playKnight(steal)×19    ← linear on tile tokens
  value                            ← MLP(global context ‖ pooled), raw logit

Interface-compatible with CatanNet: forward(obs[B,1328]) →
(policy_logits[B,300], value[B,1]); exports to ONNX with the same
input/output names, so the TS workers need no changes.
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn

OBS_SIZE = 1328
ACT_SIZE = 300

N_VERT, N_EDGE, N_TILE = 54, 72, 19
N_TOKENS = N_VERT + N_EDGE + N_TILE          # 145
G_END, V_END, T_END = 68, 770, 1112          # block boundaries in the obs
VF, TF, EF = 13, 18, 3                       # per-token feature widths


def _load_adjacency() -> torch.Tensor:
    """Dense row-normalised adjacency over [vertices | edges | tiles]."""
    topo_path = Path(__file__).resolve().parent / "topology.json"
    if not topo_path.exists():
        raise FileNotFoundError(
            f"{topo_path} missing — run "
            "`node node_modules/tsx/dist/cli.mjs scripts/dump_topology.ts` "
            "in packages/ai")
    topo = json.loads(topo_path.read_text())
    assert topo["nVertices"] == N_VERT and topo["nEdges"] == N_EDGE \
        and topo["nTiles"] == N_TILE

    A = torch.zeros(N_TOKENS, N_TOKENS)
    e_off = N_VERT
    t_off = N_VERT + N_EDGE

    for e, (v1, v2) in enumerate(topo["edgeVerts"]):
        A[e_off + e, v1] = A[v1, e_off + e] = 1   # vertex ↔ edge
        A[e_off + e, v2] = A[v2, e_off + e] = 1
        A[v1, v2] = A[v2, v1] = 1                 # vertex ↔ vertex (neighbours)

    for t, verts in enumerate(topo["tileVerts"]):
        for v in verts:
            A[t_off + t, v] = A[v, t_off + t] = 1  # vertex ↔ tile

    A += torch.eye(N_TOKENS)                       # self-loops
    A = A / A.sum(dim=1, keepdim=True)             # row-normalise
    return A


class GraphBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(d, d)
        self.w_nbr  = nn.Linear(d, d)
        self.norm   = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        m = torch.matmul(A, h)                     # [B, 145, d]
        return h + torch.relu(self.norm(self.w_self(h) + self.w_nbr(m)))


class CatanGNN(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.d = d
        self.register_buffer("A", _load_adjacency())

        self.proj_v = nn.Linear(VF, d)
        self.proj_t = nn.Linear(TF, d)
        self.proj_e = nn.Linear(EF, d)
        self.proj_g = nn.Linear(G_END, d)          # global → token context
        self.pos    = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)

        self.blocks = nn.ModuleList(GraphBlock(d) for _ in range(layers))

        # Per-token action heads.
        self.head_settle = nn.Linear(d, 1)
        self.head_city   = nn.Linear(d, 1)
        self.head_road   = nn.Linear(d, 1)
        self.head_tile   = nn.Linear(d, 4)         # robber/knight × null/steal

        # Global action + value heads on (global context ‖ pooled tokens).
        self.head_global = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 44))
        self.head_value = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        g  = obs[:, :G_END]
        hv = self.proj_v(obs[:, G_END:V_END].reshape(B, N_VERT, VF))
        ht = self.proj_t(obs[:, V_END:T_END].reshape(B, N_TILE, TF))
        he = self.proj_e(obs[:, T_END:].reshape(B, N_EDGE, EF))

        gctx = self.proj_g(g)                       # [B, d]
        h = torch.cat([hv, he, ht], dim=1)          # [B, 145, d] (V | E | T)
        h = h + self.pos.unsqueeze(0) + gctx.unsqueeze(1)

        for block in self.blocks:
            h = block(h, self.A)

        hv = h[:, :N_VERT]
        he = h[:, N_VERT:N_VERT + N_EDGE]
        ht = h[:, N_VERT + N_EDGE:]

        pooled = h.mean(dim=1)                      # [B, d]
        ctx = torch.cat([gctx, pooled], dim=-1)     # [B, 2d]

        tile = self.head_tile(ht)                   # [B, 19, 4]
        logits = torch.cat([
            self.head_global(ctx),                  # 0-43
            self.head_settle(hv).squeeze(-1),       # 44-97
            self.head_city(hv).squeeze(-1),         # 98-151
            self.head_road(he).squeeze(-1),         # 152-223
            tile[:, :, 0],                          # 224-242 moveRobber null
            tile[:, :, 1],                          # 243-261 moveRobber steal
            tile[:, :, 2],                          # 262-280 playKnight null
            tile[:, :, 3],                          # 281-299 playKnight steal
        ], dim=-1)

        value = self.head_value(ctx)                # [B, 1] raw logit
        return logits, value


def build_model(arch: str, hidden: int = 256):
    """Factory shared by train_bc.py and train_exit.py."""
    if arch == "mlp":
        from model import CatanNet
        return CatanNet(hidden=hidden)
    if arch == "gnn":
        return CatanGNN(d=hidden // 2, layers=4)
    raise ValueError(f"unknown arch: {arch}")
