"""
worker_pool.py — pool of persistent Node self-play workers, shared by
train_ppo.py (PPO) and train_exit.py (Expert Iteration).

The game engine lives in TypeScript.  N long-lived Node subprocesses
(packages/ai/scripts/rollout_worker.ts) play games fully in-process: the
trainer exports the current policy to ONNX, broadcasts the path, and workers
run inference (and, in 'mcts' mode, full PUCT search) locally.  One JSON line
per command in; one trajectory message per finished game out, with binary
payloads base64-encoded.

Protocol:
  py → js : {"cmd":"model","path":...}                       → {"t":"model_ok"}
  py → js : {"cmd":"start","seed":N,"opponent":...,"netSeat":k,"mode":...,
             ...mode-specific knobs (temp / sims / tempMoves / noise)}
  js → py : {"t":"end","winner":0..3|null,"turns":T,"steps":S,"n":K,
             "obs":<b64 f32[K*1328]>,"mask":<b64 u8[K*300]>,
             "actions":<b64 u16[K]>,"seats":<b64 u8[K]>,
             "policy":<b64 f32[K*300]>}        (policy only in 'mcts' mode)
"""

from __future__ import annotations
import base64
import copy
import json
import os
import subprocess
import threading
import queue as queue_mod
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from model import CatanNet, OBS_SIZE, ACT_SIZE


def default_worker_cmd() -> tuple[list[str], str]:
    """Returns (cmd, cwd) that launches the rollout worker via the tsx CLI
    installed in packages/ai (avoids npx/.cmd shims, which break Popen on
    Windows)."""
    ai_dir  = Path(__file__).resolve().parents[1] / "ai"
    tsx_cli = ai_dir / "node_modules" / "tsx" / "dist" / "cli.mjs"
    if not tsx_cli.exists():
        raise FileNotFoundError(
            f"tsx CLI not found at {tsx_cli} — run `pnpm install` first")
    return ["node", str(tsx_cli), "scripts/rollout_worker.ts"], str(ai_dir)


class GameTrajectory(NamedTuple):
    """One finished game's recorded net decisions, in play order."""
    obs:     np.ndarray          # [K, OBS_SIZE] float32
    mask:    np.ndarray          # [K, ACT_SIZE] uint8
    actions: np.ndarray          # [K] uint16
    seats:   np.ndarray          # [K] uint8
    policy:  np.ndarray | None   # [K, ACT_SIZE] float32 MCTS visit dists, or None
    winner:  int | None          # seat index, or None if the game hit the step cap
    turns:   int                 # endTurn count (game length in turns)
    steps:   int                 # total engine moves (incl. forced/discards)


class WorkerPool:
    """N persistent Node self-play workers (in-process ONNX inference)."""

    def __init__(self, n_workers: int) -> None:
        cmd, cwd = default_worker_cmd()
        self.queue: queue_mod.Queue[tuple[int, dict | None]] = queue_mod.Queue()
        self.procs: list[subprocess.Popen] = []
        for i in range(n_workers):
            # stderr inherited so engine errors are visible in the console.
            proc = subprocess.Popen(
                cmd, cwd=cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1,
            )
            self.procs.append(proc)
            threading.Thread(target=self._reader, args=(i, proc),
                             daemon=True).start()
        self._expect_all("ready")

    def _reader(self, idx: int, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.queue.put((idx, json.loads(line)))
        self.queue.put((idx, None))             # EOF sentinel

    def _send(self, idx: int, obj: dict) -> None:
        self.procs[idx].stdin.write(json.dumps(obj) + "\n")
        self.procs[idx].stdin.flush()

    def _expect_all(self, t: str) -> None:
        for _ in range(len(self.procs)):
            widx, msg = self.queue.get()
            if msg is None or msg.get("t") != t:
                raise RuntimeError(
                    f"worker {widx}: expected '{t}', got {msg} "
                    "(see its stderr above)")

    def set_model(self, onnx_path: str | os.PathLike) -> None:
        """Broadcasts a new policy; blocks until every worker has loaded it."""
        path = str(Path(onnx_path).resolve())
        for i in range(len(self.procs)):
            self._send(i, {"cmd": "model", "path": path})
        self._expect_all("model_ok")

    def set_model2(self, onnx_path: str | os.PathLike) -> None:
        """Loads the 'net2' opponent model (checkpoint-ladder evals)."""
        path = str(Path(onnx_path).resolve())
        for i in range(len(self.procs)):
            self._send(i, {"cmd": "model2", "path": path})
        self._expect_all("model2_ok")

    def play_games(self, jobs: list[dict],
                   on_progress=None) -> list[GameTrajectory]:
        """
        Plays all jobs across the pool; workers run their games autonomously.
        Each job is sent verbatim (plus cmd:"start") — keys use the worker's
        protocol names: seed, opponent, netSeat, mode, temp, sims, tempMoves,
        noise.  Returns trajectories aligned with `jobs`.
        """
        results: list[GameTrajectory | None] = [None] * len(jobs)
        job_of: dict[int, int] = {}             # worker idx → job index
        next_job = 0
        done = 0

        def assign(widx: int) -> None:
            nonlocal next_job
            if next_job >= len(jobs):
                return
            self._send(widx, {"cmd": "start", **jobs[next_job]})
            job_of[widx] = next_job
            next_job += 1

        for w in range(len(self.procs)):
            assign(w)

        while done < len(jobs):
            widx, msg = self.queue.get()
            if msg is None:
                raise RuntimeError(
                    f"rollout worker {widx} exited unexpectedly "
                    "(see its stderr above)")
            assert msg["t"] == "end", f"unexpected message: {msg.get('t')}"

            k = msg["n"]
            if k > 0:
                obs  = np.frombuffer(base64.b64decode(msg["obs"]),
                                     dtype=np.float32).reshape(k, OBS_SIZE)
                mask = np.frombuffer(base64.b64decode(msg["mask"]),
                                     dtype=np.uint8).reshape(k, ACT_SIZE)
                actions = np.frombuffer(base64.b64decode(msg["actions"]), dtype=np.uint16)
                seats   = np.frombuffer(base64.b64decode(msg["seats"]),   dtype=np.uint8)
                policy  = (np.frombuffer(base64.b64decode(msg["policy"]),
                                         dtype=np.float32).reshape(k, ACT_SIZE)
                           if "policy" in msg else None)
            else:
                obs  = np.empty((0, OBS_SIZE), dtype=np.float32)
                mask = np.empty((0, ACT_SIZE), dtype=np.uint8)
                actions = np.empty(0, dtype=np.uint16)
                seats   = np.empty(0, dtype=np.uint8)
                policy  = None

            results[job_of[widx]] = GameTrajectory(
                obs, mask, actions, seats, policy, msg["winner"],
                msg.get("turns", 0), msg.get("steps", 0))
            done += 1
            if on_progress:
                on_progress(done, len(jobs))
            assign(widx)

        return results  # type: ignore[return-value]

    def close(self) -> None:
        for i in range(len(self.procs)):
            try:
                self._send(i, {"cmd": "stop"})
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


# ── ONNX export for the workers (fast, quiet, every iteration) ────────────────

def export_rollout_onnx(model: CatanNet, path: str) -> None:
    """Exports the current policy for the workers.  Fixed batch size 1 (that
    is how the workers run it) and the legacy TorchScript exporter when
    available — it is much faster and quieter than the dynamo exporter, which
    matters when exporting every iteration."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(1, OBS_SIZE)
    try:
        torch.onnx.export(
            cpu_model, dummy, path,
            input_names=["obs"], output_names=["policy_logits", "value"],
            opset_version=17, dynamo=False,
        )
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(
            cpu_model, dummy, path,
            input_names=["obs"], output_names=["policy_logits", "value"],
            opset_version=17,
        )


def load_bc_warmstart(model: CatanNet, path: str, device) -> None:
    """Loads a BC checkpoint into the policy side of CatanNet, failing loudly
    if anything but value-side keys mismatches (a silent partial load
    scrambles the pretrained policy)."""
    bc  = torch.load(path, map_location=device)
    res = model.load_state_dict(bc, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("value")]
    if bad:
        raise RuntimeError(
            f"BC checkpoint does not match CatanNet: missing {bad} "
            f"(unexpected: {res.unexpected_keys}). Retrain BC or fix model.py.")
