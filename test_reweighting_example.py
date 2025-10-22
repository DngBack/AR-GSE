"""
Test reweighting computation with numerical example.
"""
import torch
import numpy as np
import json
from pathlib import Path

# Load inverse weights
with open('data/cifar100_lt_if100_splits_fixed/inverse_class_weights.json') as f:
    inv_weights = torch.tensor(json.load(f), dtype=torch.float32)

print("="*70)
print("NUMERICAL EXAMPLE: REWEIGHTING IN GROUP ERROR")
print("="*70)

# Example: 10 samples from test set (balanced)
# 6 from head (classes 0-68), 4 from tail (classes 69-99)
labels = torch.tensor([5, 10, 20, 30, 40, 50,    # 6 head samples
                      70, 75, 80, 99])            # 4 tail samples

predictions = torch.tensor([5, 10, 22, 30, 42, 51,    # head: 2 wrong (indices 2, 4, 5)
                           71, 75, 82, 99])            # tail: 2 wrong (indices 6, 8)

accept = torch.tensor([1, 1, 1, 1, 1, 1,    # head: all accepted
                      1, 1, 1, 0], dtype=torch.bool)  # tail: 1 rejected (index 9)

# Get sample weights
sample_weights = inv_weights[labels]

print(f"\nSetup:")
print(f"  Head samples (0-68): {(labels < 69).sum()} samples")
print(f"  Tail samples (69-99): {(labels >= 69).sum()} samples")
print(f"  Accepted: {accept.sum()}/{len(accept)}")

print(f"\nSample weights:")
for i in range(len(labels)):
    group = "head" if labels[i] < 69 else "tail"
    status = "accept" if accept[i] else "reject"
    correct = "✓" if predictions[i] == labels[i] else "✗"
    print(f"  [{i}] class={labels[i]:2d} ({group:4s}), w={sample_weights[i]:.4f}, {status:6s}, {correct}")

# Compute HEAD group error
head_mask = labels < 69
head_weights = sample_weights[head_mask]
head_accept = accept[head_mask]
head_correct = (predictions[head_mask] == labels[head_mask])

print(f"\n{'='*70}")
print("HEAD GROUP (classes 0-68):")
print(f"{'='*70}")

head_errors = (~head_correct).float() * head_accept.float() * head_weights
head_total_weight = (head_accept.float() * head_weights).sum()

print(f"  Numerator (weighted errors on accepted):")
for i, idx in enumerate(torch.where(head_mask)[0]):
    if head_accept[i]:
        contrib = head_errors[i].item()
        print(f"    Sample {idx}: w={head_weights[i]:.4f}, error={not head_correct[i]}, contrib={contrib:.4f}")

print(f"  Sum numerator: {head_errors.sum():.4f}")
print(f"  Sum denominator (total weight accepted): {head_total_weight:.4f}")

e_head_weighted = (head_errors.sum() / head_total_weight).item()
print(f"  → e_head(w) = {e_head_weighted:.6f}")

# Without reweighting (uniform)
e_head_uniform = (~head_correct & head_accept).float().mean().item()
print(f"  → e_head(uniform) = {e_head_uniform:.6f}")

# Compute TAIL group error
tail_mask = labels >= 69
tail_weights = sample_weights[tail_mask]
tail_accept = accept[tail_mask]
tail_correct = (predictions[tail_mask] == labels[tail_mask])

print(f"\n{'='*70}")
print("TAIL GROUP (classes 69-99):")
print(f"{'='*70}")

tail_errors = (~tail_correct).float() * tail_accept.float() * tail_weights
tail_total_weight = (tail_accept.float() * tail_weights).sum()

print(f"  Numerator (weighted errors on accepted):")
for i, idx in enumerate(torch.where(tail_mask)[0]):
    if tail_accept[i]:
        contrib = tail_errors[i].item()
        print(f"    Sample {idx}: w={tail_weights[i]:.4f}, error={not tail_correct[i]}, contrib={contrib:.4f}")

print(f"  Sum numerator: {tail_errors.sum():.4f}")
print(f"  Sum denominator (total weight accepted): {tail_total_weight:.4f}")

e_tail_weighted = (tail_errors.sum() / tail_total_weight).item()
print(f"  → e_tail(w) = {e_tail_weighted:.6f}")

# Without reweighting
e_tail_uniform = (~tail_correct & tail_accept).float().mean().item()
print(f"  → e_tail(uniform) = {e_tail_uniform:.6f}")

# Overall metrics
print(f"\n{'='*70}")
print("OBJECTIVES:")
print(f"{'='*70}")

R_bal_weighted = (e_head_weighted + e_tail_weighted) / 2
R_bal_uniform = (e_head_uniform + e_tail_uniform) / 2

R_worst_weighted = max(e_head_weighted, e_tail_weighted)
R_worst_uniform = max(e_head_uniform, e_tail_uniform)

print(f"  Balanced:")
print(f"    With reweight:    {R_bal_weighted:.6f}")
print(f"    Without reweight: {R_bal_uniform:.6f}")
print(f"    Difference: {abs(R_bal_weighted - R_bal_uniform):.6f}")

print(f"\n  Worst-group:")
print(f"    With reweight:    {R_worst_weighted:.6f}")
print(f"    Without reweight: {R_worst_uniform:.6f}")
print(f"    Difference: {abs(R_worst_weighted - R_worst_uniform):.6f}")

print(f"\n{'='*70}")
print("KEY INSIGHT:")
print(f"{'='*70}")
print("""
Reweighting changes metric values because:
  1. Tail errors get HIGHER weight (rare classes more important)
  2. Head errors get LOWER weight (common classes less important)
  3. This reflects performance under IMBALANCED train distribution
  4. Critical for fair evaluation on long-tail data!
""")
