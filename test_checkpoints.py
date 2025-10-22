"""Quick test both checkpoints"""
import torch
from pathlib import Path
from src.models.gating_network_map import GatingNetwork

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for ckpt_name in ['best_gating.pth', 'final_gating.pth']:
    print(f"\n{'='*60}")
    print(f"Testing: {ckpt_name}")
    print(f"{'='*60}")
    
    ckpt_path = Path(f'checkpoints/gating_map/cifar100_lt_if100/{ckpt_name}')
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    
    # Create and load model
    gating = GatingNetwork(num_experts=3, num_classes=100, routing='dense').to(DEVICE)
    gating.load_state_dict(ckpt['model_state_dict'])
    gating.eval()
    
    # Test with dummy data
    with torch.no_grad():
        dummy = torch.randn(10, 3, 100, device=DEVICE)
        out = gating(dummy)
        weights = out[0] if isinstance(out, tuple) else out
        
        has_nan = torch.isnan(weights).any().item()
        has_inf = torch.isinf(weights).any().item()
        
        if has_nan:
            print("❌ Produces NaN!")
        elif has_inf:
            print("❌ Produces Inf!")
        else:
            print(f"✅ Works! Range: [{weights.min():.3f}, {weights.max():.3f}], Sum: {weights.sum(dim=-1).mean():.3f}")

print(f"\n{'='*60}")
