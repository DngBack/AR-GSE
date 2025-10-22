"""
Debug expert logits để tìm nguyên nhân Val NLL cao
"""

import torch
import torch.nn.functional as F
from pathlib import Path

print("="*70)
print("DEBUGGING EXPERT LOGITS")
print("="*70)

# Load expert logits for val split
logits_dir = './outputs/logits/cifar100_lt_if100/'
experts = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']

print("\nLoading val split logits...")
val_logits_list = []
for expert in experts:
    path = Path(logits_dir) / expert / 'val_logits.pt'
    logits = torch.load(path, map_location='cpu').float()
    val_logits_list.append(logits)
    print(f"  {expert}: {logits.shape}")

# Stack: [E, N, C] → [N, E, C]
val_logits = torch.stack(val_logits_list, dim=0).transpose(0, 1)
print(f"\nCombined shape: {val_logits.shape}")

# Convert to posteriors
posteriors = torch.softmax(val_logits, dim=-1)  # [N, E, C]

# Load labels
import json
import torchvision

with open('./data/cifar100_lt_if100_splits_fixed/val_indices.json', 'r') as f:
    val_indices = json.load(f)

dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
labels = torch.tensor([dataset.targets[i] for i in val_indices])

print(f"Labels shape: {labels.shape}")

# Analyze expert posteriors
print("\n" + "="*70)
print("EXPERT POSTERIOR ANALYSIS")
print("="*70)

for e, expert in enumerate(experts):
    post_e = posteriors[:, e, :]  # [N, C]
    true_probs = torch.gather(post_e, dim=1, index=labels.unsqueeze(1)).squeeze(1)
    
    nll_e = -torch.log(true_probs + 1e-8).mean()
    
    print(f"\n{expert}:")
    print(f"  Posterior range: [{post_e.min():.6f}, {post_e.max():.6f}]")
    print(f"  True class prob: min={true_probs.min():.6f}, max={true_probs.max():.6f}, mean={true_probs.mean():.6f}")
    print(f"  NLL: {nll_e:.6f}")
    print(f"  Accuracy: {(post_e.argmax(dim=1) == labels).float().mean():.4f}")

# Uniform mixture
print("\n" + "="*70)
print("UNIFORM MIXTURE")
print("="*70)

uniform_mixture = posteriors.mean(dim=1)  # [N, C]
true_probs_mix = torch.gather(uniform_mixture, dim=1, index=labels.unsqueeze(1)).squeeze(1)
nll_mix = -torch.log(true_probs_mix + 1e-8).mean()

print(f"Mixture posterior range: [{uniform_mixture.min():.6f}, {uniform_mixture.max():.6f}]")
print(f"True class prob: min={true_probs_mix.min():.6f}, max={true_probs_mix.max():.6f}, mean={true_probs_mix.mean():.6f}")
print(f"NLL: {nll_mix:.6f}")
print(f"Accuracy: {(uniform_mixture.argmax(dim=1) == labels).float().mean():.4f}")

# Check if there are any issues
print("\n" + "="*70)
print("POTENTIAL ISSUES")
print("="*70)

# Issue 1: Very low probabilities
very_low_probs = (true_probs_mix < 1e-3).sum()
print(f"\n1. Samples with very low prob (<0.001): {very_low_probs} / {len(labels)}")

if very_low_probs > 0:
    low_indices = torch.where(true_probs_mix < 1e-3)[0]
    print(f"   Example low prob samples:")
    for idx in low_indices[:5]:
        print(f"     Sample {idx}: prob={true_probs_mix[idx]:.2e}, label={labels[idx]}, "
              f"pred={uniform_mixture[idx].argmax()}")

# Issue 2: Check logits scale
print(f"\n2. Logits scale check:")
print(f"   Logits range: [{val_logits.min():.2f}, {val_logits.max():.2f}]")
print(f"   Logits mean: {val_logits.mean():.2f}, std: {val_logits.std():.2f}")

# Issue 3: Any NaN or Inf?
print(f"\n3. Numerical issues:")
print(f"   NaN in logits: {torch.isnan(val_logits).any()}")
print(f"   Inf in logits: {torch.isinf(val_logits).any()}")
print(f"   NaN in posteriors: {torch.isnan(posteriors).any()}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

expected_nll = nll_mix.item()
observed_val_nll = 2.02

print(f"\nExpected NLL (from logits): {expected_nll:.4f}")
print(f"Observed Val NLL (in training): {observed_val_nll:.4f}")

if abs(expected_nll - observed_val_nll) > 0.1:
    print("\n⚠️  MISMATCH! Val NLL in training không khớp với tính trực tiếp")
    print("   → Có vấn đề trong code tính loss hoặc data loading")
else:
    print("\n✓ Val NLL khớp với expected value")
    if expected_nll > 1.0:
        print("   → NLL cao là do expert predictions kém trên val set")
        print("   → Không phải bug, mà là chất lượng expert")
