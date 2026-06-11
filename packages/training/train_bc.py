#!/usr/bin/env python3
"""
train_bc.py — Behavioural Cloning: train CatanNet to imitate greedy decisions.

This is Phase 1 of the RL pipeline.  The goal is not a finished agent; it is a
network that already knows the game mechanics so that self-play RL in Phase 2
converges quickly rather than rediscovering the rules from scratch.

What BC teaches:
  • Which actions are typically chosen (policy head mimics greedy)
  • Whether a position tends to win (value head predicts game outcome)

What BC cannot teach (learned in Phase 2 via self-play):
  • Situations where greedy makes systematic mistakes
  • Strategic depth beyond greedy's heuristic horizon

Usage:
    python train_bc.py --data-dir ../../data/greedy --out-dir models/bc_v1
    python train_bc.py --data-dir ../../data/greedy --out-dir models/bc_v1 --epochs 50 --batch 2048

After training, catan_net.onnx in --out-dir is ready to import into TS via
onnxruntime-node.  See packages/ai/src/net-policy.ts (to be written next).
"""

from __future__ import annotations
import os
import sys
import time
import argparse

# Windows consoles default to cp1252, which cannot encode the unicode glyphs
# in the log header / torch's ONNX exporter output. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import load_dataset, OBS_SIZE
from model_gnn import build_model


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    full_ds = load_dataset(args.data_dir)
    n_val   = max(2_000, int(0.05 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: {n_train:,} train  {n_val:,} val")

    loader_kw = dict(batch_size=args.batch, pin_memory=(device.type == "cuda"), num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.arch, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    policy_loss_fn = nn.CrossEntropyLoss()
    # Value head outputs a RAW logit (no Sigmoid — see model.py), so the
    # with-logits form is required; plain BCELoss would reject the inputs.
    value_loss_fn  = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\n{'Epoch':>5}  {'pol↓':>8}  {'val↓':>8}  {'acc↑':>7}  "
          f"{'v_pol↓':>8}  {'v_val↓':>8}  {'v_acc↑':>7}  {'lr':>8}  {'s':>5}")
    print("─" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        tp = tv = ta = tn = 0.0
        for obs, actions, outcomes in train_loader:
            obs, actions, outcomes = obs.to(device), actions.to(device), outcomes.to(device)
            pol, val = model(obs)

            ploss = policy_loss_fn(pol, actions)
            vloss = value_loss_fn(val.squeeze(1), outcomes)
            loss  = ploss + args.value_weight * vloss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            with torch.no_grad():
                acc = (pol.argmax(1) == actions).float().mean().item()
            b = len(obs)
            tp += ploss.item() * b
            tv += vloss.item() * b
            ta += acc * b
            tn += b

        # ── Validation epoch ──────────────────────────────────────────────────
        model.eval()
        vp = vv = va = vn = 0.0
        with torch.no_grad():
            for obs, actions, outcomes in val_loader:
                obs, actions, outcomes = obs.to(device), actions.to(device), outcomes.to(device)
                pol, val = model(obs)
                b = len(obs)
                vp += policy_loss_fn(pol, actions).item() * b
                vv += value_loss_fn(val.squeeze(1), outcomes).item() * b
                va += (pol.argmax(1) == actions).float().mean().item() * b
                vn += b

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        train_pol = tp / tn
        train_val = tv / tn
        train_acc = ta / tn
        val_pol   = vp / vn
        val_val   = vv / vn
        val_acc   = va / vn
        val_loss  = val_pol + args.value_weight * val_val

        print(f"{epoch:5d}  {train_pol:8.4f}  {train_val:8.4f}  {train_acc:7.3f}  "
              f"{val_pol:8.4f}  {val_val:8.4f}  {val_acc:7.3f}  {lr_now:8.2e}  {elapsed:5.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    print(f"\nBest epoch: {best_epoch}  val_loss={best_val_loss:.4f}")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print("\nExporting to ONNX ...")
    state_dict = torch.load(os.path.join(args.out_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    onnx_path = os.path.join(args.out_dir, "catan_net.onnx")
    # Shared legacy exporter (CPU copy, batch 1 — how the TS side runs it).
    # The dynamo exporter chokes on CUDA-resident models and on the GNN.
    from worker_pool import export_rollout_onnx
    export_rollout_onnx(model, onnx_path)
    print(f"  Saved: {onnx_path}")

    # Quick sanity check with onnxruntime (if installed).
    try:
        import onnxruntime as ort
        sess   = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, OBS_SIZE), dtype=np.float32)
        pol_out, val_out = sess.run(None, {"obs": dummy_np})
        print(f"  ONNX forward:  policy={pol_out.shape}  value={val_out.shape}  ✓")
        print(f"  Policy top-5 actions: {pol_out[0].argsort()[-5:][::-1].tolist()}")
    except ImportError:
        print("  (onnxruntime not installed — skipping inference test)")

    print(f"\nDone. Next: load {onnx_path} in TS via onnxruntime-node.")
    print("  See packages/ai/src/net-policy.ts for the inference wrapper.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Behavioural cloning for CatanNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     default="data",    help="Directory containing shard_*.bin files")
    p.add_argument("--out-dir",      default="models",  help="Directory for checkpoints and ONNX export")
    p.add_argument("--epochs",       type=int,   default=30,   help="Training epochs")
    p.add_argument("--batch",        type=int,   default=1024, help="Batch size")
    p.add_argument("--lr",           type=float, default=3e-4, help="Initial learning rate (AdamW)")
    p.add_argument("--hidden",       type=int,   default=256,  help="Hidden layer width")
    p.add_argument("--arch",         choices=["mlp", "gnn"], default="mlp",
                   help="Network architecture (gnn = graph net over the board)")
    p.add_argument("--value-weight", type=float, default=0.5,
                   help="Weight of value loss relative to policy loss (L = L_pol + w * L_val)")
    train(p.parse_args())