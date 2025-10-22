"""Check if checkpoint parameters contain NaN/Inf"""
import torch
from pathlib import Path

DEVICE = 'cpu'  # Use CPU to avoid CUDA issues

ckpt_path = Path('checkpoints/gating_map/cifar100_lt_if100/best_gating.pth')
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

print("Checking checkpoint parameters...")
print(f"Epoch: {ckpt['epoch']}\n")

state_dict = ckpt['model_state_dict']

for name, param in state_dict.items():
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    
    status = "✅" if not (has_nan or has_inf) else "❌"
    
    print(f"{status} {name}")
    print(f"   Shape: {param.shape}, Range: [{param.min():.4f}, {param.max():.4f}]")
    
    if has_nan:
        print(f"   ⚠️  Contains NaN: {torch.isnan(param).sum()}/{param.numel()}")
    if has_inf:
        print(f"   ⚠️  Contains Inf: {torch.isinf(param).sum()}/{param.numel()}")
    print()
