# export.py
import torch
from model import CatanNet
from dataset import OBS_SIZE
import sys

out_dir   = sys.argv[1] if len(sys.argv) > 1 else "models/bc_v1"
onnx_path = f"{out_dir}/catan_net.onnx"

model = CatanNet()
state_dict = torch.load(f"{out_dir}/best.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy = torch.zeros(1, OBS_SIZE)

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names  = ["obs"],
    output_names = ["policy_logits", "value"],
    dynamic_axes = {
        "obs":           {0: "batch"},
        "policy_logits": {0: "batch"},
        "value":         {0: "batch"},
    },
    opset_version = 17,
)
print(f"Saved: {onnx_path}")