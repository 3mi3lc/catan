"""
model_gnn.py — CatanGNN: a graph network over the board structure.

Motivation (measured, not theoretical): the flat MLP plateaued ~20 points
below its MCTS teacher in Expert Iteration — it must memorise position-by-
position what a graph model can compute relationally ("good expansion spot",
"blockable road", "robber pressure") once and apply at any location.

Tokens
──────
The observation (encoding.ts, 1619 floats, 4-seat + trades) factors into
per-element blocks; we reshape it into 145 tokens + a global context:
    global   obs[   0: 147]               (hands, 3×opp, bank, phase, robber, LR/LA, negotiation)
    vertex   obs[ 147: 957] → [54, 15]    (owner none/me/opp1-3, kind, port)
    tile     obs[ 957:1299] → [19, 18]    (terrain, token, robber)
    edge     obs[1299:1659] → [72,  5]    (road owner none/me/opp1-3)

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
    0- 43  global actions             ← MLP(global context ‖ mean-pooled tokens)
   44- 97  buildSettlement ×54        ← linear on vertex tokens
   98-151  buildCity       ×54        ← linear on vertex tokens
  152-223  buildRoad       ×72        ← linear on edge tokens
  224-299  moveRobber ×19 × {none,opp1,opp2,opp3}   ← linear on tile tokens
  300-375  playKnight ×19 × {none,opp1,opp2,opp3}   ← linear on tile tokens
  376-395  trade negotiation (20): add give/want, broadcast, cancel, accept,
           reject, counter, submit, confirm×3, decline  ← MLP(global ‖ pooled)
  value    [B,4]                      ← MLP(global ‖ pooled): per-seat win logits,
                                         seat-relative (slot 0 = acting seat)

forward(obs[B,1659]) → (policy_logits[B,396], value[B,4]); exports to ONNX
with the same input/output names ("obs"→"policy_logits","value").
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn

OBS_SIZE = 1659
ACT_SIZE = 396
N_SEATS  = 4                                 # value head width (seat-relative)
N_TRADE  = 20                                # negotiation: addGive5 addWant5 broadcast cancel accept reject counter submit confirm3 decline

N_VERT, N_EDGE, N_TILE = 54, 72, 19
N_TOKENS = N_VERT + N_EDGE + N_TILE          # 145
G_END, V_END, T_END = 147, 957, 1299         # block boundaries in the obs
VF, TF, EF = 15, 18, 5                        # per-token feature widths


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
        # 8 channels per tile: moveRobber {none,opp1,opp2,opp3} then
        # playKnight {none,opp1,opp2,opp3} (steal target = relative opponent).
        self.head_tile   = nn.Linear(d, 8)

        # Global action + value heads on (global context ‖ pooled tokens).
        self.head_global = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 44))
        # Trade actions (also non-spatial): respondToTrade(2) + proposeTrade(60).
        self.head_trade = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, N_TRADE))
        self.head_value = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, N_SEATS))

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

        tile = self.head_tile(ht)                   # [B, 19, 8]
        logits = torch.cat([
            self.head_global(ctx),                  # 0-43
            self.head_settle(hv).squeeze(-1),       # 44-97
            self.head_city(hv).squeeze(-1),         # 98-151
            self.head_road(he).squeeze(-1),         # 152-223
            tile[:, :, 0],                          # 224-242 moveRobber none
            tile[:, :, 1],                          # 243-261 moveRobber opp1
            tile[:, :, 2],                          # 262-280 moveRobber opp2
            tile[:, :, 3],                          # 281-299 moveRobber opp3
            tile[:, :, 4],                          # 300-318 playKnight none
            tile[:, :, 5],                          # 319-337 playKnight opp1
            tile[:, :, 6],                          # 338-356 playKnight opp2
            tile[:, :, 7],                          # 357-375 playKnight opp3
            self.head_trade(ctx),                   # 376-437 respond(2) + propose(60)
        ], dim=-1)

        value = self.head_value(ctx)                # [B, 4] per-seat raw logits
        return logits, value


def build_model(arch: str, hidden: int = 256):
    """Factory shared by train_bc.py and train_exit.py."""
    if arch == "mlp":
        from model import CatanNet
        return CatanNet(hidden=hidden)
    if arch == "gnn":
        return CatanGNN(d=hidden // 2, layers=4)
    raise ValueError(f"unknown arch: {arch}")
