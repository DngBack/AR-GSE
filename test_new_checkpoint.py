"""
Test newly trained gating checkpoint (without LayerNorm).
"""
import torch
import numpy as np
from src.models.gating_network_map import GatingNetwork

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/gating_map/cifar100_lt_if100/best_gating.pth'

print("="*70)
print("🧪 TESTING NEW GATING CHECKPOINT (No LayerNorm)")
print("="*70)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
print(f"\n✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"   Val balanced acc: {checkpoint['val_metrics']['balanced_acc']:.4f}")

# Create model (normalize_features=False by default now)
model = GatingNetwork(
    num_experts=3,
    num_classes=100,
    routing='dense',
    hidden_dims=[256, 128]
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"   normalize_features: {model.feature_extractor.normalize_features}")

# Test with random posteriors
print("\n" + "="*70)
print("Testing with random posteriors...")
print("="*70)

with torch.no_grad():
    posteriors = torch.randn(16, 3, 100, device=DEVICE).softmax(dim=-1)
    output = model(posteriors)
    
    if isinstance(output, tuple):
        weights, logits = output
    else:
        weights = output
        logits = None
    
    print(f"Input shape: {posteriors.shape}")
    print(f"Output weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"Weights sum: {weights.sum(dim=-1).mean():.6f} (should be 1.0)")
    
    if torch.isnan(weights).any():
        print("❌ STILL PRODUCES NaN!")
        print(f"   NaN count: {torch.isnan(weights).sum()}/{weights.numel()}")
    else:
        print("✅ NO NaN - Checkpoint is working!")
        print(f"   Mean weights per expert: {weights.mean(dim=0)}")

# Test with real expert logits
print("\n" + "="*70)
print("Testing with REAL expert logits...")
print("="*70)

logits_val = torch.load('outputs/logits_fixed/cifar100_lt_if100/val_logits.pt', map_location=DEVICE, weights_only=False)
print(f"Loaded val logits: {logits_val.shape}")
print(f"Logits range: [{logits_val.min():.3f}, {logits_val.max():.3f}]")

with torch.no_grad():
    real_posteriors = logits_val[:16].softmax(dim=-1)  # [16, 3, 100]
    print(f"Real posteriors range: [{real_posteriors.min():.6f}, {real_posteriors.max():.6f}]")
    
    output = model(real_posteriors)
    
    if isinstance(output, tuple):
        weights, logits = output
    else:
        weights = output
    
    print(f"Output weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"Weights sum: {weights.sum(dim=-1).mean():.6f}")
    
    if torch.isnan(weights).any():
        print("❌ PRODUCES NaN WITH REAL LOGITS!")
        print(f"   NaN count: {torch.isnan(weights).sum()}/{weights.numel()}")
    else:
        print("✅ WORKS WITH REAL LOGITS!")
        print(f"   Mean weights per expert: {weights.mean(dim=0)}")

print("\n" + "="*70)
print("🎉 TEST COMPLETE")
print("="*70)
