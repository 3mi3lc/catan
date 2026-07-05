# packages/training/merge_shards.py
import glob, os, struct, sys
import numpy as np

data_dir  = sys.argv[1]
out_path  = sys.argv[2]  # e.g. ../../data/greedy4/merged.bin

OBS_SIZE = 1603
ACT_SIZE = 376
MAGIC    = 0x34415443

paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.bin")))
print(f"Merging {len(paths)} shards...")

total = 0
for p in paths:
    with open(p, "rb") as f:
        _, n, _, _ = struct.unpack("<4I", f.read(16))
    total += n
print(f"Total samples: {total:,}")

obs_all     = np.memmap(out_path + ".obs", dtype="<f4", mode="w+", shape=(total, OBS_SIZE))
actions_all = np.memmap(out_path + ".act", dtype="<u2", mode="w+", shape=(total,))
outcomes_all= np.memmap(out_path + ".out", dtype="u1",  mode="w+", shape=(total,))

offset = 0
for i, p in enumerate(paths):
    with open(p, "rb") as f:
        header = f.read(16)
    _, n, _, _ = struct.unpack("<4I", header)
    raw     = np.fromfile(p, dtype=np.uint8)[16:]
    e_obs   = n * OBS_SIZE * 4
    e_act   = e_obs + n * 2
    obs     = raw[:e_obs].view("<f4").reshape(n, OBS_SIZE)
    actions = raw[e_obs:e_obs + n*2].view("<u2")
    outcomes= raw[e_obs + n*2:e_act + n]
    obs_all[offset:offset+n]      = obs
    actions_all[offset:offset+n]  = actions
    outcomes_all[offset:offset+n] = outcomes
    offset += n
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(paths)}")

obs_all.flush(); actions_all.flush(); outcomes_all.flush()
print(f"Done. Written to {out_path}.obs / .act / .out")