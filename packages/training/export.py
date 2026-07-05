# export.py
import torch
import sys
from model_gnn import build_model
from dataset import OBS_SIZE

# Torch's ONNX exporter prints unicode (✅) that crashes the Windows cp1252
# console; force UTF-8 so the export's own logging can't kill the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

out_dir   = sys.argv[1] if len(sys.argv) > 1 else "models/exit_v1"
ckpt_file = sys.argv[2] if len(sys.argv) > 2 else "latest.pt"
arch      = sys.argv[3] if len(sys.argv) > 3 else "gnn"
hidden    = int(sys.argv[4]) if len(sys.argv) > 4 else 256

ckpt_path = f"{out_dir}/{ckpt_file}"
onnx_path = f"{out_dir}/{ckpt_file.replace('.pt', '.onnx')}"

ckpt = torch.load(ckpt_path, map_location="cpu")
model = build_model(arch, hidden=hidden)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

dummy = torch.zeros(1, OBS_SIZE)
# dynamo=False → legacy TorchScript exporter, which embeds the weights in a
# single .onnx file. The default dynamo exporter spills weights to a sidecar
# .onnx.data, which breaks single-file deployment (e.g. the web app asset).
export_kwargs = dict(
    input_names=["obs"],
    output_names=["policy_logits", "value"],
    dynamic_axes={"obs": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
    opset_version=17,
)
try:
    torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
except TypeError:  # older torch without the dynamo kwarg
    torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
print(f"Saved: {onnx_path}")