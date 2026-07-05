#!/usr/bin/env python3
"""
tournament.py — round-robin Elo tournament across checkpoints.

Answers "which generation is actually the best?" properly: single 100-game
evals vs greedy have ±10pt noise and saturate near the ceiling, while
head-to-head play between checkpoints discriminates small real differences.
Plays every pair of models N games (seats alternating), then fits
Bradley–Terry strengths reported on an Elo scale (mean anchored at 1000).

Usage:
    # Tournament over every 20th generation of a run
    python tournament.py --models "models/exit_gnn_v1/gen_*.pt" --arch gnn

    # Specific contenders, more games per pair
    python tournament.py --models models/exit_gnn_v1/gen_0120.pt models/exit_gnn_v1/latest.pt \
        models/bc_gnn_v1/best.pt --arch gnn --games-per-pair 400
"""

from __future__ import annotations
import argparse
import glob as globmod
import itertools
import math
import os
import sys
import tempfile
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from model_gnn import build_model
from worker_pool import WorkerPool, export_rollout_onnx


def to_onnx(path: str, arch: str, tmpdir: str, tag: str) -> str:
    if path.endswith(".onnx"):
        return path
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = build_model(arch)
    model.load_state_dict(state)
    out = os.path.join(tmpdir, f"{tag}.onnx")
    export_rollout_onnx(model, out)
    return out


def fit_elo(n: int, wins: list[list[int]], games: list[list[int]],
            iters: int = 2000, lr: float = 8.0) -> list[float]:
    """Gradient fit of Bradley–Terry ratings on the Elo scale (mean = 1000)."""
    r = [0.0] * n
    scale = math.log(10) / 400
    for _ in range(iters):
        grad = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i == j or games[i][j] == 0:
                    continue
                p = 1 / (1 + math.exp(-(r[i] - r[j]) * scale))
                grad[i] += wins[i][j] - games[i][j] * p
        for i in range(n):
            r[i] += lr * grad[i] / max(1, sum(games[i]))
        mean = sum(r) / n
        r = [x - mean for x in r]
    return [1000 + x for x in r]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Round-robin Elo tournament across model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Checkpoint paths or globs (.pt or .onnx)")
    p.add_argument("--arch", choices=["mlp", "gnn"], default="gnn")
    p.add_argument("--games-per-pair", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=7_000_000)
    p.add_argument("--workers", type=int,
                   default=min(32, 2 * (os.cpu_count() or 8)))
    args = p.parse_args()

    paths: list[str] = []
    for pat in args.models:
        hits = sorted(globmod.glob(pat))
        paths.extend(hits if hits else [pat])
    # De-dup, keep order.
    seen: set[str] = set()
    paths = [x for x in paths if not (x in seen or seen.add(x))]
    if len(paths) < 2:
        p.error(f"need at least 2 models, found {len(paths)}: {paths}")

    names = [os.path.relpath(x).replace("\\", "/") for x in paths]
    short = [os.path.basename(x) for x in paths]
    n = len(paths)
    pairs = list(itertools.combinations(range(n), 2))
    total_games = len(pairs) * args.games_per_pair
    print(f"{n} models, {len(pairs)} pairs, {args.games_per_pair} games/pair "
          f"= {total_games:,} games\n")

    wins = [[0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx = [to_onnx(x, args.arch, tmpdir, f"m{i}") for i, x in enumerate(paths)]
        pool = WorkerPool(args.workers)
        try:
            for k, (i, j) in enumerate(pairs):
                pool.set_model(onnx[i])
                pool.set_model2(onnx[j])
                jobs = [{
                    "seed": args.seed_base + k * args.games_per_pair + g,
                    "opponent": "net2", "netSeat": g % 2, "mode": "argmax",
                } for g in range(args.games_per_pair)]
                for job, traj in zip(jobs, pool.play_games(jobs)):
                    if traj.winner is None:
                        continue
                    games[i][j] += 1
                    games[j][i] += 1
                    if traj.winner == job["netSeat"]:
                        wins[i][j] += 1
                    else:
                        wins[j][i] += 1
                wr = wins[i][j] / max(1, games[i][j])
                print(f"  [{k + 1:3d}/{len(pairs)}] {short[i]} vs {short[j]}: "
                      f"{wr * 100:5.1f}%  ({time.time() - t0:.0f}s)")
        finally:
            pool.close()

    elo = fit_elo(n, wins, games)
    order = sorted(range(n), key=lambda i: -elo[i])

    print(f"\n{'rank':>4}  {'Elo':>6}  {'overall':>8}  model")
    print("-" * 64)
    for rank, i in enumerate(order, 1):
        g = sum(games[i])
        w = sum(wins[i])
        print(f"{rank:4d}  {elo[i]:6.0f}  {w / max(1, g) * 100:7.1f}%  {names[i]}")
    print(f"\nbest: {names[order[0]]}   ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
