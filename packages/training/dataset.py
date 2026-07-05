"""
dataset.py — reads the binary shard files produced by collect.ts.

Shard format (little-endian):
    bytes  0- 3  uint32  magic = 0x34415443  ('CTA4')
    bytes  4- 7  uint32  num_samples
    bytes  8-11  uint32  obs_size  (1659)
    bytes 12-15  uint32  act_size  (396)
    then three contiguous blocks:
      float32[num_samples × obs_size]  observations
      uint16[num_samples]              action indices
      uint8[num_samples]               outcomes: seat-relative winner offset
                                       (0 = acting seat won) — a class label
"""

from __future__ import annotations
import os
import glob
import struct
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

OBS_SIZE = 1659
ACT_SIZE = 396
MAGIC    = 0x34415443   # 'CTA4' — 4-seat + trade shards


def read_shard(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        obs      float32[N, 1328]
        actions  int64[N]
        outcomes float32[N]   (0.0 or 1.0)
    """
    raw = np.fromfile(path, dtype=np.uint8)
    magic, n, obs_size, act_size = struct.unpack_from("<4I", raw.tobytes(), offset=0)
    assert magic    == MAGIC,    f"Bad magic in {path}: got {hex(magic)}"
    assert obs_size == OBS_SIZE, f"obs_size mismatch: {obs_size} != {OBS_SIZE}"
    assert act_size == ACT_SIZE, f"act_size mismatch: {act_size} != {ACT_SIZE}"

    payload   = raw[16:]
    e_obs     = n * obs_size * 4          # end of obs block (bytes)
    e_act     = e_obs + n * 2             # end of action block
    e_out     = e_act + n                 # end of outcome block

    obs     = payload[:e_obs].view("<f4").reshape(n, obs_size).copy()
    actions = payload[e_obs:e_act].view("<u2").astype(np.int64).copy()
    outcomes = payload[e_act:e_out].astype(np.int64).copy()   # seat-relative winner class

    return obs, actions, outcomes


class ShardDataset(Dataset):
    """Wraps one binary shard as a PyTorch Dataset."""

    def __init__(self, path: str) -> None:
        self.obs, self.actions, self.outcomes = read_shard(path)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.obs[idx],      dtype=torch.float32),
            torch.tensor(self.actions[idx],  dtype=torch.long),
            torch.tensor(self.outcomes[idx], dtype=torch.long),   # winner class for CE
        )


def load_dataset(data_dir: str, pattern: str = "shard_*.bin") -> ConcatDataset:
    """Load all shards matching pattern from data_dir."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No shards found at {os.path.join(data_dir, pattern)}\n"
            "Run: pnpm --filter @catan/ai collect <games> <out_dir>"
        )
    print(f"Loading {len(paths)} shard(s) from {data_dir} ...")
    shards = [ShardDataset(p) for p in paths]
    total  = sum(len(s) for s in shards)
    print(f"  {total:,} total samples across {len(paths)} shards")
    return ConcatDataset(shards)