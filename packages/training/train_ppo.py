#!/usr/bin/env python3
"""
train_ppo.py — PPO self-play: train CatanNet to beat itself (and greedy).

⚠️ DEPRECATED / INCOMPATIBLE with the 4-seat representation. This was the
   capped/abandoned 2-player path (scalar sigmoid value + binary win/loss
   reward). The current pipeline is seat-relative 4-player with a per-seat
   value head ([B,4], cross-entropy) and Expert Iteration (train_exit.py).
   This script's GAE/value code assumes the old scalar value and will break
   against the new model — use train_exit.py instead. Kept only for reference.

This is Phase 2 of the RL pipeline.  Warm-starts from the BC checkpoint so the
agent already knows the game mechanics; self-play then pushes it beyond greedy.

Algorithm
─────────
  Proximal Policy Optimisation (clip variant, Schulman et al. 2017)
  • Rollout  N parallel games → collect (obs, action, mask, reward, value, done)
  • GAE-λ    compute advantages and returns
  • Update   K epochs over minibatches, clipped surrogate + value + entropy

Two-player note
───────────────
  Each game has two seats.  We collect trajectories for BOTH players from the
  same game and label the terminal reward as +1 (win) / 0 (loss).  Because the
  net plays against a copy of itself (or a mix of self + greedy), both sides
  contribute gradient signal every game.

Opponent pool
─────────────
  --opponent-mix controls the fraction of games where the opponent is frozen
  greedy vs a past checkpoint.  Start at 0.5 (half greedy, half self) and
  lower it as the agent improves.

Usage:
    # First run — warm-start from BC
    python train_ppo.py --bc-checkpoint models/bc_v1/best.pt --out-dir models/ppo_v1

    # Resume
    python train_ppo.py --resume models/ppo_v1/latest.pt --out-dir models/ppo_v1

    # Evaluate every N iterations against greedy (in-loop, deterministic argmax)
    python train_ppo.py --bc-checkpoint models/bc_v1/best.pt --out-dir models/ppo_v1 --eval-interval 50

Requires `pnpm install` to have been run (the rollout worker is launched via
the tsx CLI in packages/ai/node_modules).
"""

from __future__ import annotations
import os
import sys
import time

# Windows consoles default to cp1252, which cannot encode the unicode glyphs
# torch's ONNX exporter (and our log lines) print. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import random
import copy
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import CatanNet, make_optimizer, OBS_SIZE, ACT_SIZE


# ── Rollout buffer ────────────────────────────────────────────────────────────

class Rollout(NamedTuple):
    obs:      torch.Tensor   # [T, OBS_SIZE]
    actions:  torch.Tensor   # [T]         int64
    masks:    torch.Tensor   # [T, ACT_SIZE] float32  legal-action mask
    log_probs:torch.Tensor   # [T]         float32   log π_old(a|s)
    values:   torch.Tensor   # [T]         float32   V_old(s)
    returns:  torch.Tensor   # [T]         float32   GAE targets
    advantages:torch.Tensor  # [T]         float32   GAE advantages


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:  list[float],
    values:   list[float],
    dones:    list[bool],
    gamma:    float = 0.99,
    lam:      float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    Returns (advantages, returns) for one trajectory.
    values should include V(s_T) as the last element (bootstrap).
    """
    adv = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


# ── JS bridge ──────────────────────────────────────────────────────────────────
#
# The worker pool (persistent Node self-play workers with in-process ONNX
# inference) lives in worker_pool.py, shared with train_exit.py.
# log-probs and values are NOT computed during rollout — Python recomputes
# them afterwards in a few large batched forward passes over the collected
# observations.  The weights are identical to what sampled the actions, so
# the recomputed log π_old matches up to ONNX/torch float noise, which the
# PPO ratio clamp absorbs.

from worker_pool import (
    WorkerPool, GameTrajectory, export_rollout_onnx, load_bc_warmstart,
)



# ── Recompute log π_old and V_old for collected trajectories ──────────────────

def recompute_logp_values(
    model:  CatanNet,
    obs:    np.ndarray,    # [N, OBS_SIZE] float32
    mask:   np.ndarray,    # [N, ACT_SIZE] uint8
    actions:np.ndarray,    # [N] int
    device: torch.device,
    temperature: float = 1.0,
    chunk:  int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched forward passes over everything the workers collected.
    `temperature` must match what the workers sampled with: log π_old is the
    BEHAVIOUR policy's log-prob, so the PPO ratio stays a correct importance
    weight even when collection is tempered."""
    model.eval()
    lps, vals = [], []
    with torch.no_grad():
        for s in range(0, len(obs), chunk):
            e      = min(s + chunk, len(obs))
            obs_t  = torch.from_numpy(obs[s:e]).to(device)
            mask_t = torch.from_numpy(mask[s:e].astype(np.float32)).to(device)
            act_t  = torch.from_numpy(actions[s:e].astype(np.int64)).to(device)
            logits, value = model(obs_t)
            masked = (logits / temperature).masked_fill(mask_t == 0, float("-inf"))
            dist   = torch.distributions.Categorical(logits=masked)
            lps.append(dist.log_prob(act_t).cpu().numpy())
            vals.append(value.squeeze(-1).cpu().numpy())
    return np.concatenate(lps), np.concatenate(vals)


# ── Collect N games into a Rollout ────────────────────────────────────────────

def collect_rollout(
    model:        CatanNet,
    pool:         WorkerPool,
    n_games:      int,
    gamma:        float,
    lam:          float,
    device:       torch.device,
    rollout_onnx: str,
    seed_offset:  int = 0,
    opponent_mix: float = 0.5,    # fraction of games vs frozen greedy
    temperature:  float = 1.0,    # sampling temperature during collection
    rng:          random.Random | None = None,
) -> tuple[Rollout, dict[str, float]]:
    """
    Exports the current policy, plays n_games across the pool (workers run
    inference in-process), then recomputes log π_old / V_old in batched torch
    passes and builds GAE targets.  Each net-controlled seat yields its own
    trajectory: terminal reward +1 (win) / 0 (loss); GAE is computed per
    trajectory — never across the two interleaved seats.
    info contains 'greedy_winrate' (NaN if no greedy games this batch).
    """
    rng = rng or random

    export_rollout_onnx(model, rollout_onnx)
    pool.set_model(rollout_onnx)

    jobs = [{
        "seed":     seed_offset + g,
        "opponent": "greedy" if rng.random() < opponent_mix else "self",
        "netSeat":  g % 2,   # alternate seats to cancel first-player advantage
        "mode":     "sample",
        "temp":     temperature,
    } for g in range(n_games)]

    games = pool.play_games(jobs)

    # Keep only finished games; track stats.
    kept: list[tuple[dict, GameTrajectory]] = []
    greedy_games = greedy_wins = unfinished = 0
    for job, traj in zip(jobs, games):
        if traj.winner is None or len(traj.actions) == 0:
            unfinished += 1
            continue
        if job["opponent"] == "greedy":
            greedy_games += 1
            greedy_wins  += int(traj.winner == job["netSeat"])
        kept.append((job, traj))

    if not kept:
        raise RuntimeError("rollout produced no samples (all games unfinished?)")

    # One big recompute pass over every collected step.
    obs_all  = np.concatenate([t.obs  for _, t in kept])
    mask_all = np.concatenate([t.mask for _, t in kept])
    act_all  = np.concatenate([t.actions for _, t in kept])
    lp_all, val_all = recompute_logp_values(
        model, obs_all, mask_all, act_all, device, temperature=temperature)

    # Per-(game, seat) trajectories → rewards, dones, GAE.
    ret_all = np.empty(len(obs_all), dtype=np.float32)
    adv_all = np.empty(len(obs_all), dtype=np.float32)
    offset = 0
    for _, traj in kept:
        k = len(traj.actions)
        for seat in np.unique(traj.seats):
            idx = offset + np.flatnonzero(traj.seats == seat)   # play order
            rewards = [0.0] * len(idx)
            rewards[-1] = 1.0 if traj.winner == seat else 0.0
            dones = [False] * len(idx)
            dones[-1] = True
            values = [float(val_all[i]) for i in idx] + [0.0]   # terminal bootstrap

            adv, ret = compute_gae(rewards, values, dones, gamma, lam)
            adv_all[idx] = adv
            ret_all[idx] = ret
        offset += k

    obs_t  = torch.from_numpy(obs_all)
    act_t  = torch.from_numpy(act_all.astype(np.int64))
    mask_t = torch.from_numpy(mask_all.astype(np.float32))
    lp_t   = torch.from_numpy(lp_all)
    val_t  = torch.from_numpy(val_all)
    ret_t  = torch.from_numpy(ret_all)
    adv_t  = torch.from_numpy(adv_all)

    # Normalise advantages over the whole batch
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    info = {
        "greedy_winrate": greedy_wins / greedy_games if greedy_games else float("nan"),
        "unfinished": float(unfinished),
    }
    return Rollout(obs_t, act_t, mask_t, lp_t, val_t, ret_t, adv_t), info


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(
    model:       CatanNet,
    optimiser:   torch.optim.Optimizer,
    rollout:     Rollout,
    clip_eps:    float,
    value_coef:  float,
    entropy_coef:float,
    ppo_epochs:  int,
    minibatch:   int,
    device:      torch.device,
    max_grad_norm: float = 0.5,
    train_policy:  bool  = True,   # False during critic warm-up: value head only
    target_kl:     float | None = None,  # stop the epoch loop if KL drifts past this
) -> dict[str, float]:
    dataset = TensorDataset(
        rollout.obs, rollout.actions, rollout.masks,
        rollout.log_probs, rollout.returns, rollout.advantages,
    )
    loader = DataLoader(dataset, batch_size=minibatch, shuffle=True, drop_last=False)

    stats = dict(pol=0.0, val=0.0, ent=0.0, kl=0.0, n=0)
    model.train()
    stop = False

    for _ in range(ppo_epochs):
        if stop:
            break
        for obs, actions, masks, old_lp, returns, advantages in loader:
            obs, actions, masks  = obs.to(device), actions.to(device), masks.to(device)
            old_lp               = old_lp.to(device)
            returns, advantages  = returns.to(device), advantages.to(device)

            logits, value = model(obs)
            masked_logits = logits.masked_fill(masks == 0, float('-inf'))
            dist    = torch.distributions.Categorical(logits=masked_logits)
            new_lp  = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            log_ratio = new_lp - old_lp
            ratio     = log_ratio.exp()
            # Stable KL(π_old ‖ π_new) estimator (Schulman's k3).
            approx_kl = ((ratio - 1) - log_ratio).mean()

            surr1       = ratio * advantages
            surr2       = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss  = F.mse_loss(value.squeeze(-1), returns)

            if train_policy:
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            else:
                loss = value_coef * value_loss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()

            b = len(obs)
            stats["pol"] += policy_loss.item() * b
            stats["val"] += value_loss.item()  * b
            stats["ent"] += entropy.item()     * b
            stats["kl"]  += approx_kl.item()   * b
            stats["n"]   += b

            # Drifting too far from the behaviour policy invalidates the
            # clipped-surrogate approximation — and, warm-started from BC,
            # silently destroys the pretrained policy. Stop this batch early.
            if train_policy and target_kl is not None and approx_kl.item() > 1.5 * target_kl:
                stop = True
                break

    n = stats["n"] or 1
    return {k: v / n for k, v in stats.items() if k != "n"}


# ── Greedy win-rate eval ──────────────────────────────────────────────────────

def eval_vs_greedy(
    model:        CatanNet,
    pool:         WorkerPool,
    rollout_onnx: str,
    n_games:      int = 100,
    seed_base:    int = 900_000,
) -> float:
    """Deterministic (argmax) eval vs greedy, alternating seats.
    Returns net win rate over decided games."""
    # Re-export: the PPO update has changed the weights since collection.
    export_rollout_onnx(model, rollout_onnx)
    pool.set_model(rollout_onnx)
    jobs = [{"seed": seed_base + g, "opponent": "greedy",
             "netSeat": g % 2, "mode": "argmax"}
            for g in range(n_games)]
    games = pool.play_games(jobs)
    wins = decided = 0
    for job, traj in zip(jobs, games):
        if traj.winner is None:
            continue
        decided += 1
        wins += int(traj.winner == job["netSeat"])
    return wins / decided if decided else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CatanNet(hidden=args.hidden).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0) + 1
    elif args.bc_checkpoint:
        print(f"Warm-starting from BC checkpoint: {args.bc_checkpoint}")
        load_bc_warmstart(model, args.bc_checkpoint, device)
        start_iter = 1
    else:
        print("Training from scratch (no BC warm-start)")
        start_iter = 1

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimiser = make_optimizer(
        model,
        lr_body   = args.lr_body,
        lr_policy = args.lr_policy,
        lr_value  = args.lr_value,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    pool = WorkerPool(args.workers)
    rng  = random.Random(args.seed)
    rollout_onnx = os.path.join(args.out_dir, "rollout.onnx")
    print(f"Rollout workers: {args.workers}")
    if args.policy_warmup > 0 and start_iter <= args.policy_warmup:
        print(f"Critic warm-up: value head only for iterations 1-{args.policy_warmup}")

    # Baseline before any PPO touches the weights — every later eval should be
    # judged against this number, not against zero.
    wr0 = eval_vs_greedy(model, pool, rollout_onnx, n_games=args.eval_games)
    print(f"Baseline eval vs greedy (argmax, {args.eval_games} games): {wr0 * 100:.1f}%")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Iter':>5}  {'games':>6}  {'pol':>8}  {'val':>8}  {'ent':>8}  "
          f"{'kl':>7}  {'win%':>6}  {'steps':>9}  {'s':>5}")
    print("-" * 81)

    recent_wins: deque[float] = deque(maxlen=200)
    total_steps = 0

    frozen = False

    def set_warmup_freeze(freeze: bool) -> None:
        # During critic warm-up the value head must learn on FROZEN features:
        # the body is shared, so even a value-only loss would otherwise push
        # gradients through it and scramble the pretrained policy.
        for p in model.body.parameters():
            p.requires_grad_(not freeze)
        for p in model.policy_head.parameters():
            p.requires_grad_(not freeze)

    try:
        for it in range(start_iter, args.iterations + 1):
            t0 = time.time()
            warmup = it <= args.policy_warmup
            if warmup != frozen:
                set_warmup_freeze(warmup)
                frozen = warmup
                if not warmup:
                    print(f"  -- warm-up done: body + policy unfrozen --")

            rollout, info = collect_rollout(
                model        = model,
                pool         = pool,
                n_games      = args.games_per_iter,
                gamma        = args.gamma,
                lam          = args.lam,
                device       = device,
                rollout_onnx = rollout_onnx,
                seed_offset  = (it - 1) * args.games_per_iter,
                opponent_mix = args.opponent_mix,
                temperature  = args.temperature,
                rng          = rng,
            )
            total_steps += len(rollout.obs)
            if not np.isnan(info["greedy_winrate"]):
                recent_wins.append(info["greedy_winrate"])

            stats = ppo_update(
                model        = model,
                optimiser    = optimiser,
                rollout      = rollout,
                clip_eps     = args.clip_eps,
                value_coef   = args.value_coef,
                entropy_coef = args.entropy_coef,
                ppo_epochs   = args.ppo_epochs,
                minibatch    = args.minibatch,
                device       = device,
                train_policy = not warmup,
                target_kl    = args.target_kl or None,
            )

            elapsed = time.time() - t0
            win = (100 * sum(recent_wins) / len(recent_wins)) if recent_wins else float("nan")
            tag = " [warmup]" if warmup else ""
            print(f"{it:5d}  {args.games_per_iter:6d}  "
                  f"{stats['pol']:8.4f}  {stats['val']:8.4f}  {stats['ent']:8.4f}  "
                  f"{stats['kl']:7.4f}  {win:6.1f}  {total_steps:9,d}  {elapsed:5.1f}{tag}")

            # ── Checkpoint ────────────────────────────────────────────────────
            ckpt = {"model": model.state_dict(), "iteration": it}
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
            if it % args.save_interval == 0:
                torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_{it:05d}.pt"))

            # ── ONNX export (for TS eval) ──────────────────────────────────────
            if it % args.export_interval == 0:
                _export_onnx(model, args.out_dir, it, device)
                print(f"  -> exported catan_net_{it:05d}.onnx  (also updated catan_net.onnx)")

            # ── Deterministic eval vs greedy ───────────────────────────────────
            if it % args.eval_interval == 0:
                wr = eval_vs_greedy(model, pool, rollout_onnx,
                                    n_games=args.eval_games)
                print(f"  -> eval vs greedy (argmax, {args.eval_games} games): {wr * 100:.1f}%")

        print("\nDone.")
        _export_onnx(model, args.out_dir, args.iterations, device)
    finally:
        pool.close()


def _export_onnx(model: CatanNet, out_dir: str, it: int, device: torch.device) -> None:
    model.eval()
    cpu_model = copy.deepcopy(model).cpu()
    dummy     = torch.zeros(1, OBS_SIZE)
    versioned = os.path.join(out_dir, f"catan_net_{it:05d}.onnx")
    latest    = os.path.join(out_dir, "catan_net.onnx")
    for path in (versioned, latest):
        torch.onnx.export(
            cpu_model, dummy, path,
            input_names  = ["obs"],
            output_names = ["policy_logits", "value"],
            dynamic_axes = {
                "obs":           {0: "batch"},
                "policy_logits": {0: "batch"},
                "value":         {0: "batch"},
            },
            opset_version = 17,
        )
    model.to(device)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--i-know-this-is-deprecated" not in sys.argv:
        sys.exit(
            "train_ppo.py is incompatible with the 4-seat model (per-seat value "
            "head, cross-entropy). Use train_exit.py. To run anyway for "
            "reference, pass --i-know-this-is-deprecated.")
    sys.argv = [a for a in sys.argv if a != "--i-know-this-is-deprecated"]

    p = argparse.ArgumentParser(
        description="PPO self-play for CatanNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoints
    p.add_argument("--bc-checkpoint", default=None,
                   help="BC best.pt to warm-start from (skip if --resume)")
    p.add_argument("--resume",        default=None,
                   help="PPO latest.pt to resume training")
    p.add_argument("--out-dir",       default="models/ppo_v1")

    # Architecture
    p.add_argument("--hidden",        type=int,   default=256)

    # PPO hypers
    p.add_argument("--iterations",    type=int,   default=1000,
                   help="Number of collect→update iterations")
    p.add_argument("--games-per-iter",type=int,   default=128,
                   help="Games to collect per iteration")
    p.add_argument("--workers",       type=int,
                   default=min(32, 2 * (os.cpu_count() or 8)),
                   help="Parallel Node rollout workers (served by batched "
                        "inference). Workers idle while awaiting actions, so "
                        "~2x the core count is a good default")
    p.add_argument("--ppo-epochs",    type=int,   default=4,
                   help="Gradient epochs per collected batch")
    p.add_argument("--minibatch",     type=int,   default=512)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--gamma",         type=float, default=1.0,
                   help="Discount; 1.0 (undiscounted outcome) is right for "
                        "win/loss board games")
    p.add_argument("--lam",           type=float, default=0.97,
                   help="GAE lambda (credit-assignment horizon vs variance)")
    p.add_argument("--value-coef",    type=float, default=0.5)
    p.add_argument("--entropy-coef",  type=float, default=0.003)
    p.add_argument("--policy-warmup", type=int,   default=30,
                   help="Iterations of value-head-only training before any "
                        "policy update (protects a BC warm-start from a "
                        "random critic)")
    p.add_argument("--target-kl",     type=float, default=0.02,
                   help="Stop the PPO epoch loop early when approx KL exceeds "
                        "1.5x this (0 disables)")
    p.add_argument("--temperature",   type=float, default=1.0,
                   help="Sampling temperature during collection (<1 keeps the "
                        "behaviour policy closer to argmax strength)")
    p.add_argument("--lr-body",       type=float, default=1e-4)
    p.add_argument("--lr-policy",     type=float, default=5e-5)
    p.add_argument("--lr-value",      type=float, default=1e-3)
    p.add_argument("--opponent-mix",  type=float, default=0.5,
                   help="Fraction of rollout games vs frozen greedy "
                        "(rest are self-play); lower it as the agent improves")
    p.add_argument("--seed",          type=int,   default=0,
                   help="RNG seed for the opponent-mix coin flips")

    # Logging / saving
    p.add_argument("--save-interval",  type=int,  default=50,
                   help="Save a versioned checkpoint every N iterations")
    p.add_argument("--export-interval",type=int,  default=50,
                   help="Export ONNX every N iterations (for TS eval)")
    p.add_argument("--eval-interval",  type=int,  default=100,
                   help="Run deterministic greedy eval every N iterations")
    p.add_argument("--eval-games",     type=int,  default=100,
                   help="Games per in-loop greedy eval")

    train(p.parse_args())
