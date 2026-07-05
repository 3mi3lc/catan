#!/usr/bin/env python3
"""
train_exit.py — Expert Iteration (AlphaZero-style) self-play for CatanNet.

The loop per generation:
  1. Export the current net to ONNX; broadcast to the worker pool.
  2. Workers play self-play games where EVERY decision comes from PUCT MCTS
     guided by the net (policy priors + value leaf evaluation, Dirichlet root
     noise, visit-proportional sampling early in the game).
  3. Each recorded decision yields a training sample:
       policy target  = the MCTS visit distribution (an improvement on the
                        net's prior — search is the improvement operator)
       value  target  = the winner's seat offset relative to the acting seat
                        (0 = acting seat won) — cross-entropy over N_SEATS
  4. Train the net supervised on a replay buffer of recent generations
     (cross-entropy to visit distributions + BCE on outcomes).
  5. Evaluate: raw-net argmax vs greedy every generation (cheap); the search
     itself only gets stronger on top of that.

Unlike PPO, the gradient step here is plain supervised learning toward
search-improved targets, so it cannot random-walk a strong BC policy
downhill — the failure mode that capped the PPO experiments.

Usage:
    # Warm-start from BC (recommended)
    python train_exit.py --bc-checkpoint models/bc_v1/best.pt --out-dir models/exit_v1

    # Resume
    python train_exit.py --resume models/exit_v1/latest.pt --out-dir models/exit_v1

Requires `pnpm install` (workers run via the tsx CLI in packages/ai).
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# Windows consoles default to cp1252, which cannot encode the unicode glyphs
# torch's ONNX exporter prints. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from model import CatanNet, OBS_SIZE, ACT_SIZE
from model_gnn import build_model
from worker_pool import (
    WorkerPool, export_rollout_onnx, load_bc_warmstart,
)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """FIFO sample buffer over the last N samples (numpy, CPU)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.obs:    deque[np.ndarray] = deque()
        self.mask:   deque[np.ndarray] = deque()
        self.policy: deque[np.ndarray] = deque()
        self.value:  deque[np.ndarray] = deque()
        self.size = 0

    def add_block(self, obs, mask, policy, value) -> None:
        self.obs.append(obs); self.mask.append(mask)
        self.policy.append(policy); self.value.append(value)
        self.size += len(obs)
        while self.size - len(self.obs[0]) >= self.capacity:
            self.size -= len(self.obs.popleft())
            self.mask.popleft(); self.policy.popleft(); self.value.popleft()

    def arrays(self):
        return (np.concatenate(self.obs), np.concatenate(self.mask),
                np.concatenate(self.policy), np.concatenate(self.value))


# ── Self-play collection ──────────────────────────────────────────────────────

def collect_generation(
    pool: WorkerPool,
    n_games: int,
    sims: int,
    temp_moves: int,
    seed_offset: int,
    value_mix: float,
    rollout_cap: int,
    full_prob: float,
    fast_sims: int,
    noise_eps: float,
    target_prune: float,
    num_players: int,
):
    """Plays n_games of MCTS self-play; returns sample arrays + stats."""
    jobs = [{
        "seed": seed_offset + g,
        "opponent": "self",
        "numPlayers": num_players,
        "netSeat": 0,
        "mode": "mcts",
        "sims": sims,
        "tempMoves": temp_moves,
        "noise": True,
        "valueMix": value_mix,
        "rolloutCap": rollout_cap,
        "fullProb": full_prob,
        "fastSims": fast_sims,
        "noiseEps": noise_eps,
        "targetPrune": target_prune,
    } for g in range(n_games)]

    t_start = time.time()

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"    collecting {done}/{total} games  "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    games = pool.play_games(jobs, on_progress=progress)

    obs_l, mask_l, pol_l, val_l = [], [], [], []
    unfinished = 0
    total_decisions = 0
    for traj in games:
        if traj.winner is None or traj.policy is None or len(traj.actions) == 0:
            unfinished += 1
            continue
        # Value target: relative turn offset of the winner from each acting seat
        # (0 = the acting seat won). Cast to int before the modulo so uint8
        # wraparound can't corrupt the class for num_players not dividing 256.
        outcome = ((int(traj.winner) - traj.seats.astype(np.int64))
                   % num_players).astype(np.int64)
        obs_l.append(traj.obs)
        mask_l.append(traj.mask)
        pol_l.append(traj.policy)
        val_l.append(outcome)
        total_decisions += len(traj.actions)

    if not obs_l:
        raise RuntimeError("self-play produced no samples (all games unfinished?)")

    return (np.concatenate(obs_l), np.concatenate(mask_l),
            np.concatenate(pol_l), np.concatenate(val_l),
            {"unfinished": unfinished, "decisions": total_decisions})


# ── Supervised update on the replay buffer ────────────────────────────────────

def train_generation(
    model: CatanNet,
    optimiser: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    value_coef: float,
) -> dict[str, float]:
    obs, mask, policy, value = buffer.arrays()
    n = len(obs)
    model.train()
    stats = dict(pol=0.0, val=0.0, n=0)

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            obs_t  = torch.from_numpy(obs[idx]).to(device)
            mask_t = torch.from_numpy(mask[idx].astype(np.float32)).to(device)
            pol_t  = torch.from_numpy(policy[idx]).to(device)
            val_t  = torch.from_numpy(value[idx]).to(device).long()  # seat-relative winner class

            logits, v = model(obs_t)
            masked = logits.masked_fill(mask_t == 0, float("-inf"))
            logp   = F.log_softmax(masked, dim=-1)
            # Targets are zero on illegal slots; avoid -inf * 0 = nan.
            pol_loss = -(torch.where(mask_t.bool(), pol_t * logp,
                                     torch.zeros_like(logp))
                         .sum(dim=-1).mean())
            # Value: per-seat win-prob softmax, cross-entropy to the winner's
            # relative seat offset (v is [B, N_SEATS]).
            val_loss = F.cross_entropy(v, val_t)

            loss = pol_loss + value_coef * val_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            b = len(idx)
            stats["pol"] += pol_loss.item() * b
            stats["val"] += val_loss.item() * b
            stats["n"]   += b

    n_seen = stats["n"] or 1
    return {k: v / n_seen for k, v in stats.items() if k != "n"}


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_vs_greedy(
    pool: WorkerPool,
    n_games: int,
    seed_base: int = 900_000,
    mode: str = "argmax",
    sims: int = 96,
    value_mix: float = 0.25,
    rollout_cap: int = 600,
    num_players: int = 4,
) -> float:
    """Win rate vs greedy, rotating the net's seat. mode='argmax' measures the
    raw net; mode='mcts' (noise off, no temperature) measures net + search —
    the deployed strength. In 4-player the no-skill baseline is ~25%."""
    jobs = [{
        "seed": seed_base + g, "opponent": "greedy",
        "numPlayers": num_players, "netSeat": g % num_players,
        "mode": mode,
        **({"sims": sims, "tempMoves": 0, "noise": False,
            "valueMix": value_mix, "rolloutCap": rollout_cap}
           if mode == "mcts" else {}),
    } for g in range(n_games)]
    games = pool.play_games(jobs)
    wins = decided = 0
    for job, traj in zip(jobs, games):
        if traj.winner is None:
            continue
        decided += 1
        wins += int(traj.winner == job["netSeat"])
    return wins / decided if decided else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def value_mix_at(gen: int, args: argparse.Namespace) -> float:
    """Linear anneal of the leaf value-mix: rollouts carry evaluation while
    the value head is young; the net takes over (and collection gets cheaper)
    as generations accumulate."""
    if args.value_mix_anneal_gens <= 0:
        return args.value_mix_start
    frac = min(1.0, gen / args.value_mix_anneal_gens)
    return args.value_mix_start + frac * (args.value_mix_end - args.value_mix_start)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.arch, hidden=args.hidden).to(device)
    start_gen = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_gen = ckpt.get("generation", 0) + 1
    elif args.bc_checkpoint:
        print(f"Warm-starting from BC checkpoint: {args.bc_checkpoint}")
        load_bc_warmstart(model, args.bc_checkpoint, device)
    else:
        print("Training from scratch (no BC warm-start)")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    os.makedirs(args.out_dir, exist_ok=True)
    rollout_onnx = os.path.join(args.out_dir, "rollout.onnx")

    pool = WorkerPool(args.workers)
    print(f"Workers: {args.workers}   sims/decision: {args.sims}   "
          f"games/gen: {args.games_per_gen}")

    try:
        export_rollout_onnx(model, rollout_onnx)
        pool.set_model(rollout_onnx)
        wr0 = eval_vs_greedy(pool, args.eval_games, num_players=args.players)
        print(f"Baseline raw-net eval vs greedy "
              f"({args.players}p argmax, {args.eval_games} games): {wr0 * 100:.1f}%")

        print(f"\n{'Gen':>4}  {'games':>6}  {'samples':>8}  {'buffer':>9}  "
              f"{'pol':>7}  {'val':>7}  {'net%':>6}  {'collect_s':>9}  {'train_s':>8}")
        print("-" * 88)

        for gen in range(start_gen, args.generations + 1):
            # 1+2: export current net, MCTS self-play.
            t0 = time.time()
            export_rollout_onnx(model, rollout_onnx)
            pool.set_model(rollout_onnx)
            mix = value_mix_at(gen, args)
            obs, mask, pol, val, info = collect_generation(
                pool, args.games_per_gen, args.sims, args.temp_moves,
                seed_offset=gen * 1_000_000,
                value_mix=mix, rollout_cap=args.rollout_cap,
                full_prob=args.full_search_prob, fast_sims=args.fast_sims,
                noise_eps=args.noise_eps, target_prune=args.target_prune,
                num_players=args.players,
            )
            t_collect = time.time() - t0
            buffer.add_block(obs, mask, pol, val)

            # 3+4: supervised update toward search targets.
            t1 = time.time()
            stats = train_generation(
                model, optimiser, buffer, device,
                epochs=args.epochs, batch_size=args.batch_size,
                value_coef=args.value_coef,
            )
            t_train = time.time() - t1

            # 5: raw-net eval (cheap; search strength sits above this).
            export_rollout_onnx(model, rollout_onnx)
            pool.set_model(rollout_onnx)
            wr = eval_vs_greedy(pool, args.eval_games, num_players=args.players)

            print(f"{gen:4d}  {args.games_per_gen:6d}  {len(obs):8,d}  "
                  f"{buffer.size:9,d}  {stats['pol']:7.4f}  {stats['val']:7.4f}  "
                  f"{wr * 100:6.1f}  {t_collect:9.1f}  {t_train:8.1f}  mix={mix:.2f}")

            ckpt = {"model": model.state_dict(), "generation": gen}
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
            if gen % args.save_interval == 0:
                torch.save(ckpt, os.path.join(args.out_dir, f"gen_{gen:04d}.pt"))

            if args.search_eval_interval and gen % args.search_eval_interval == 0:
                wrs = eval_vs_greedy(pool, args.search_eval_games,
                                     mode="mcts", sims=args.sims,
                                     value_mix=mix,
                                     rollout_cap=args.rollout_cap,
                                     num_players=args.players)
                print(f"  -> net+MCTS({args.sims}) vs greedy "
                      f"({args.search_eval_games} games): {wrs * 100:.1f}%")

        print("\nDone.")
    finally:
        pool.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Expert Iteration self-play for CatanNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bc-checkpoint", default=None)
    p.add_argument("--resume",        default=None)
    p.add_argument("--out-dir",       default="models/exit_v1")
    p.add_argument("--hidden",        type=int, default=256)
    p.add_argument("--arch",          choices=["mlp", "gnn"], default="mlp",
                   help="Must match the architecture of the checkpoint being "
                        "loaded (BC warm-start or resume)")

    p.add_argument("--players",       type=int, default=4, choices=[2, 3, 4],
                   help="Seats per self-play game (4 = the target environment)")
    p.add_argument("--generations",   type=int, default=500)
    p.add_argument("--games-per-gen", type=int, default=256,
                   help="MCTS self-play games per generation")
    p.add_argument("--sims",          type=int, default=96,
                   help="MCTS simulations per decision")
    p.add_argument("--temp-moves",    type=int, default=12,
                   help="Decisions per seat sampled ∝ visits before argmax")
    p.add_argument("--value-mix-start", type=float, default=0.25,
                   help="P(net-value leaf) at gen 0; the rest use greedy rollouts")
    p.add_argument("--value-mix-end",   type=float, default=0.85,
                   help="P(net-value leaf) after the anneal completes")
    p.add_argument("--value-mix-anneal-gens", type=int, default=80,
                   help="Generations to anneal value-mix start -> end "
                        "(0 = fixed at start). Collection cost falls with the "
                        "mix, since net-value leaves skip the rollout")
    p.add_argument("--rollout-cap",   type=int, default=600,
                   help="Ply cap for MCTS leaf rollouts (deep enough to reach "
                        "real outcomes; probe: cap 600 beat cap 80 decisively)")
    p.add_argument("--full-search-prob", type=float, default=0.25,
                   help="Playout-cap randomization: fraction of decisions that "
                        "get the full search and become training samples; the "
                        "rest play with --fast-sims and are not recorded")
    p.add_argument("--fast-sims",     type=int, default=12,
                   help="Simulations for non-recorded decisions")
    p.add_argument("--noise-eps",     type=float, default=0.1,
                   help="Dirichlet root-noise weight in full searches "
                        "(0.25 is AlphaZero's value at 800 sims; too high "
                        "at 96 sims — noise dominates the visit targets)")
    p.add_argument("--target-prune",  type=float, default=0.1,
                   help="Zero actions below this fraction of the max visit "
                        "count in recorded policy targets (removes the "
                        "noise tail that teaches the net to play flat)")

    p.add_argument("--epochs",        type=int, default=2,
                   help="Passes over the replay buffer per generation")
    p.add_argument("--batch-size",    type=int, default=1024)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--value-coef",    type=float, default=1.0)
    p.add_argument("--buffer-size",   type=int, default=400_000,
                   help="Replay buffer capacity in samples (~several generations)")

    p.add_argument("--workers",       type=int,
                   default=min(32, 2 * (os.cpu_count() or 8)))
    p.add_argument("--eval-games",    type=int, default=100,
                   help="Raw-net eval games per generation")
    p.add_argument("--search-eval-interval", type=int, default=10,
                   help="Every N generations also eval net+MCTS (slow); 0 disables")
    p.add_argument("--search-eval-games",    type=int, default=50)
    p.add_argument("--save-interval", type=int, default=10)

    train(p.parse_args())
