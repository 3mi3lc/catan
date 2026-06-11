import torch
bc = torch.load('models/bc_v1/best.pt', map_location='cpu')
for k, v in bc.items():
    print(k, tuple(v.shape))
