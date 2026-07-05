#!/usr/bin/env python3
"""
eval_suite.py — head-to-head evaluation harness for trained Catan nets.

The greedy yardstick saturates above ~85% win rate; this provides the
non-saturating instruments:

  • vs greedy        — the historical baseline (continuity with old numbers)
  • vs search        — the hand-built flat-MC searchPolicy (K rollouts/candidate)
  • vs net2          — any other checkpoint, raw argmax (the checkpoint ladder)

The evaluated model plays either raw argmax ('argmax') or full net-guided
MCTS ('mcts'). Seats alternate every game; results come with a 95% CI.

Models may be .onnx files or .pt checkpoints (raw state_dict or
{"model": ...} dicts); .pt files are exported to a temporary ONNX first
(requires --arch / --arch2 to match the checkpoint).

Examples:
    # The changing-of-the-guard match: net+MCTS vs the hand-built search bot
    python eval_suite.py --model models/exit_gnn_v1/latest.pt --arch gnn \
        --mode mcts --opponent search --games 100

    # Raw net vs the hand-built search bot
    python eval_suite.py --model models/exit_gnn_v1/latest.pt --arch gnn \
        --mode argmax --opponent search --games 200

    # Checkpoint ladder: current vs generation 10
    python eval_suite.py --model models/exit_gnn_v1/latest.pt --arch gnn \
        --opponent net2 --model2 models/exit_gnn_v1/gen_0010.pt --arch2 gnn \
        --games 200
"""

from __future__ import annotations
import argparse
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
    """Returns an ONNX path for `path`, exporting .pt checkpoints if needed."""
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
        description="Catan net evaluation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    required=True, help=".onnx or .pt checkpoint")
    p.add_argument("--arch",     choices=["mlp", "gnn"], default="gnn",
                   help="Architecture of --model when it is a .pt")
    p.add_argument("--mode",     choices=["argmax", "mcts"], default="argmax",
                   help="How the evaluated model plays")
    p.add_argument("--opponent", choices=["greedy", "search", "net2"],
                   default="greedy")
    p.add_argument("--model2",   default=None, help="Opponent .onnx/.pt for net2")
    p.add_argument("--arch2",    choices=["mlp", "gnn"], default="gnn")
    p.add_argument("--players",  type=int, default=4, choices=[2, 3, 4],
                   help="Seats per game (no-skill baseline ≈ 1/players)")
    p.add_argument("--games",    type=int, default=200)
    p.add_argument("--seed-base", type=int, default=5_000_000)
    p.add_argument("--workers",  type=int,
                   default=min(32, 2 * (os.cpu_count() or 8)))
    # mcts-mode knobs (match training defaults)
    p.add_argument("--sims",        type=int,   default=96)
    p.add_argument("--value-mix",   type=float, default=0.30)
    p.add_argument("--rollout-cap", type=int,   default=600)
    # search-opponent knobs (benchmark.ts defaults)
    p.add_argument("--search-k",    type=int, default=10)
    p.add_argument("--search-cap",  type=int, default=1200)
    args = p.parse_args()

    if args.opponent == "net2" and not args.model2:
        p.error("--opponent net2 requires --model2")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_onnx = to_onnx(args.model, args.arch, tmpdir, "model")
        pool = WorkerPool(args.workers)
        try:
            pool.set_model(model_onnx)
            if args.opponent == "net2":
                pool.set_model2(to_onnx(args.model2, args.arch2, tmpdir, "model2"))

            jobs = [{
                "seed": args.seed_base + g,
                "opponent": args.opponent,
                "numPlayers": args.players,
                "netSeat": g % args.players,
                "mode": args.mode,
                **({"sims": args.sims, "tempMoves": 0, "noise": False,
                    "valueMix": args.value_mix, "rolloutCap": args.rollout_cap}
                   if args.mode == "mcts" else {}),
                **({"searchK": args.search_k, "searchCap": args.search_cap}
                   if args.opponent == "search" else {}),
            } for g in range(args.games)]

            t0 = time.time()
            done = [0]

            def progress(d: int, total: int) -> None:
                if d % 25 == 0 or d == total:
                    print(f"  {d}/{total} games  ({time.time() - t0:.0f}s)")
                done[0] = d

            games = pool.play_games(jobs, on_progress=progress)
        finally:
            pool.close()

    wins = decided = 0
    turns_all, steps_all = [], []
    turns_won, turns_lost = [], []
    for job, traj in zip(jobs, games):
        if traj.winner is None:
            continue
        decided += 1
        turns_all.append(traj.turns)
        steps_all.append(traj.steps)
        if traj.winner == job["netSeat"]:
            wins += 1
            turns_won.append(traj.turns)
        else:
            turns_lost.append(traj.turns)

    rate = wins / decided if decided else float("nan")
    ci = 1.96 * math.sqrt(rate * (1 - rate) / decided) if decided else float("nan")
    label = f"{args.mode}({args.sims})" if args.mode == "mcts" else "argmax"
    opp = {"greedy": "greedy",
           "search": f"searchPolicy(K={args.search_k},cap={args.search_cap})",
           "net2": f"net2[{args.model2}]"}[args.opponent]

    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print()
    print(f"  {os.path.basename(args.model)} [{label}]  vs  {opp}")
    print(f"  games: {args.games}  decided: {decided}  unfinished: {args.games - decided}")
    print(f"  win rate: {rate * 100:.1f}%  ±{ci * 100:.1f} (95% CI)")
    print(f"  avg turns: {avg(turns_all):.1f}  (won: {avg(turns_won):.1f}  "
          f"lost: {avg(turns_lost):.1f})   avg steps: {avg(steps_all):.1f}")
    print(f"  elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
