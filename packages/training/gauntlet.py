#!/usr/bin/env python3
"""
gauntlet.py — rank checkpoints by win rate vs the fixed greedy reference.

A ladder, not a head-to-head: every checkpoint plays the SAME opponent
(greedy) for `--games` games each, so the win rates are directly comparable
and the ranking is anchored to a fixed yardstick. Unlike tournament.py this
runs in the TARGET environment — N-player (default 4), the net on one seat
against greedy on the rest — which is what these nets were trained for.

The no-skill baseline is 1/players (≈25% at 4p); strength shows as win rate
above that. With ~300 games the 95% CI is ≈±5pt, enough to order generations.

Usage:
    python gauntlet.py --models "models/exit_gnn_4p/gen_*.pt" \
        models/exit_gnn_4p/latest.pt --arch gnn --players 4 --games 300

    # Deployed strength (slower — net + search):
    python gauntlet.py --models "models/exit_gnn_4p/gen_*.pt" --mode mcts --sims 64
"""

from __future__ import annotations
import argparse
import glob as globmod
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rank checkpoints by win rate vs greedy (target N-player env)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Checkpoint paths or globs (.pt or .onnx)")
    p.add_argument("--arch", choices=["mlp", "gnn"], default="gnn")
    p.add_argument("--players", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--games", type=int, default=300,
                   help="Games per checkpoint vs greedy")
    p.add_argument("--mode", choices=["argmax", "mcts"], default="argmax",
                   help="argmax = raw net (fast); mcts = deployed net+search")
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--value-mix", type=float, default=0.5)
    p.add_argument("--rollout-cap", type=int, default=300)
    p.add_argument("--seed-base", type=int, default=8_000_000)
    p.add_argument("--workers", type=int,
                   default=min(32, 2 * (os.cpu_count() or 8)))
    args = p.parse_args()

    paths: list[str] = []
    for pat in args.models:
        hits = sorted(globmod.glob(pat))
        paths.extend(hits if hits else [pat])
    seen: set[str] = set()
    paths = [x for x in paths if not (x in seen or seen.add(x))]
    if not paths:
        p.error("no checkpoints matched")

    names = [os.path.basename(x) for x in paths]
    baseline = 100.0 / args.players
    label = f"mcts({args.sims})" if args.mode == "mcts" else "argmax"
    print(f"{len(paths)} checkpoints · {args.players}p vs greedy · {args.games} games each "
          f"· {label}   (no-skill baseline {baseline:.0f}%)\n")

    results: list[tuple[str, float, float, int]] = []  # name, rate%, ci%, decided
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = WorkerPool(args.workers)
        try:
            for idx, (path, name) in enumerate(zip(paths, names)):
                pool.set_model(to_onnx(path, args.arch, tmpdir, f"m{idx}"))
                jobs = [{
                    "seed": args.seed_base + idx * args.games + g,
                    "opponent": "greedy",
                    "numPlayers": args.players,
                    "netSeat": g % args.players,
                    "mode": args.mode,
                    **({"sims": args.sims, "tempMoves": 0, "noise": False,
                        "valueMix": args.value_mix, "rolloutCap": args.rollout_cap}
                       if args.mode == "mcts" else {}),
                } for g in range(args.games)]

                wins = decided = 0
                for job, traj in zip(jobs, pool.play_games(jobs)):
                    if traj.winner is None:
                        continue
                    decided += 1
                    wins += int(traj.winner == job["netSeat"])
                rate = wins / decided if decided else float("nan")
                ci = 1.96 * math.sqrt(rate * (1 - rate) / decided) if decided else float("nan")
                results.append((name, rate * 100, ci * 100, decided))
                print(f"  [{idx + 1}/{len(paths)}] {name:<16} "
                      f"{rate * 100:5.1f}% ±{ci * 100:4.1f}  "
                      f"({decided} decided, {time.time() - t0:.0f}s)")
        finally:
            pool.close()

    print(f"\n{'rank':>4}  {'win%':>6}  {'95% CI':>8}  checkpoint")
    print("-" * 48)
    for rank, (name, rate, ci, _) in enumerate(
            sorted(results, key=lambda r: -r[1]), 1):
        print(f"{rank:4d}  {rate:6.1f}  ±{ci:6.1f}  {name}")
    best = max(results, key=lambda r: r[1])
    print(f"\nbest: {best[0]}  ({best[1]:.1f}% vs greedy, {args.players}p)   "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
