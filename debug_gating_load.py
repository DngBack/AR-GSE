"""
Debug script to check gating network loading and forward pass
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from src.models.gating_network_map import GatingNetwork

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("🔍 DEBUGGING GATING NETWORK")
print("="*70)

# ============================================================================
# 1. CHECK CHECKPOINT FILE
# ============================================================================
print("\n1. Checking checkpoint file...")
checkpoint_path = Path('./checkpoints/gating_map/cifar100_lt_if100/best_gating.pth')

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

print(f"✅ Checkpoint exists: {checkpoint_path}")
print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")

# ============================================================================
# 2. LOAD CHECKPOINT
# ============================================================================
print("\n2. Loading checkpoint...")

# Try both checkpoints
for ckpt_name in ['best_gating.pth', 'final_gating.pth']:
    print(f"\n📦 Testing {ckpt_name}:")
    checkpoint_path = Path(f'./checkpoints/gating_map/cifar100_lt_if100/{ckpt_name}')
    
    if not checkpoint_path.exists():
        print(f"   ❌ Not found")
        continue
    
    print(f"   ✅ Exists ({checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        if 'val_metrics' in checkpoint and checkpoint['val_metrics']:
            print(f"   Val metrics: {checkpoint['val_metrics']}")
        
        # Create fresh gating network
        gating = GatingNetwork(
            num_experts=3,
            num_classes=100,
            routing='dense'
        ).to(DEVICE)
        
        # Load state dict
        gating.load_state_dict(checkpoint['model_state_dict'])
        gating.eval()
        
        # Test with dummy data
        with torch.no_grad():
            dummy_logits = torch.randn(10, 3, 100, device=DEVICE)
            output = gating(dummy_logits)
            weights = output[0] if isinstance(output, tuple) else output
            
            has_nan = torch.isnan(weights).any().item()
            has_inf = torch.isinf(weights).any().item()
            
            if has_nan:
                print(f"   ❌ Produces NaN!")
            elif has_inf:
                print(f"   ❌ Produces Inf!")
            else:
                print(f"   ✅ Works! Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
                print(f"      Weights sum: {weights.sum(dim=-1).mean():.3f}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*70)
exit(0)

# ============================================================================
# 3. CREATE GATING NETWORK
# ============================================================================
print("\n3. Creating gating network...")
try:
    gating = GatingNetwork(
        num_experts=3,
        num_classes=100,
        routing='dense'
    ).to(DEVICE)
    print(f"✅ Gating network created")
    print(f"   Parameters: {sum(p.numel() for p in gating.parameters()):,}")
    
except Exception as e:
    print(f"❌ Error creating gating network: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 4. LOAD STATE DICT
# ============================================================================
print("\n4. Loading state dict into model...")
try:
    gating.load_state_dict(checkpoint['model_state_dict'])
    gating.eval()
    print(f"✅ State dict loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading state dict: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 5. TEST WITH DUMMY DATA
# ============================================================================
print("\n5. Testing with dummy data...")
try:
    B, E, C = 10, 3, 100
    dummy_logits = torch.randn(B, E, C, device=DEVICE)
    print(f"   Input shape: {dummy_logits.shape}")
    print(f"   Input range: [{dummy_logits.min():.3f}, {dummy_logits.max():.3f}]")
    
    with torch.no_grad():
        output = gating(dummy_logits)
    
    print(f"✅ Forward pass successful")
    print(f"   Output type: {type(output)}")
    
    if isinstance(output, tuple):
        print(f"   Tuple length: {len(output)}")
        weights = output[0]
        print(f"   Weights shape: {weights.shape}")
        print(f"   Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Weights sum per sample: {weights.sum(dim=-1)}")
        
        if torch.isnan(weights).any():
            print(f"   ⚠️  Weights contain NaN!")
        if torch.isinf(weights).any():
            print(f"   ⚠️  Weights contain Inf!")
    else:
        weights = output
        print(f"   Weights shape: {weights.shape}")
        print(f"   Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Weights sum per sample: {weights.sum(dim=-1)}")
        
        if torch.isnan(weights).any():
            print(f"   ⚠️  Weights contain NaN!")
        if torch.isinf(weights).any():
            print(f"   ⚠️  Weights contain Inf!")
    
except Exception as e:
    print(f"❌ Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 6. TEST WITH REAL VAL DATA
# ============================================================================
print("\n6. Testing with real val data...")
try:
    from train_map_cost_sweep import load_expert_logits
    
    expert_logits_val = load_expert_logits(
        ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        './outputs/logits/cifar100_lt_if100/',
        'tunev',
        DEVICE
    )
    
    print(f"   Expert logits shape: {expert_logits_val.shape}")
    print(f"   Expert logits range: [{expert_logits_val.min():.3f}, {expert_logits_val.max():.3f}]")
    
    # Check for extreme values
    if expert_logits_val.max() > 100:
        print(f"   ⚠️  WARNING: Very large logits detected!")
    if expert_logits_val.min() < -100:
        print(f"   ⚠️  WARNING: Very small logits detected!")
    
    with torch.no_grad():
        output = gating(expert_logits_val)
    
    print(f"✅ Forward pass with real data successful")
    
    if isinstance(output, tuple):
        weights = output[0]
    else:
        weights = output
    
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"   Weights mean per expert: {weights.mean(dim=0)}")
    print(f"   Weights sum per sample (first 10): {weights.sum(dim=-1)[:10]}")
    
    if torch.isnan(weights).any():
        print(f"   ❌ Weights contain NaN! Count: {torch.isnan(weights).sum()}/{weights.numel()}")
        
        # Find which samples have NaN
        nan_mask = torch.isnan(weights).any(dim=1)
        nan_indices = torch.where(nan_mask)[0]
        print(f"   Samples with NaN: {nan_indices[:10]}")
        
        if len(nan_indices) > 0:
            idx = nan_indices[0].item()
            print(f"\n   Investigating sample {idx}:")
            print(f"     Input logits: {expert_logits_val[idx]}")
            print(f"     Output weights: {weights[idx]}")
    
    if torch.isinf(weights).any():
        print(f"   ❌ Weights contain Inf! Count: {torch.isinf(weights).sum()}/{weights.numel()}")
    
    if not torch.isnan(weights).any() and not torch.isinf(weights).any():
        print(f"   ✅ All weights are valid (no NaN/Inf)")
    
except Exception as e:
    print(f"❌ Error with real data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ DEBUGGING COMPLETE")
print("="*70)
