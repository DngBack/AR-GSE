#!/usr/bin/env python3
"""
Quick test entropy regularizer fix
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/duong.xuan.bach/AR-GSE')

from src.models.gating_losses import EntropyRegularizer

print("="*70)
print("TESTING ENTROPY REGULARIZER FIX")
print("="*70)

# Mock weights
B, E = 128, 3
weights = F.softmax(torch.randn(B, E), dim=-1)

print(f"\nWeights shape: {weights.shape}")
print(f"Example weights: {weights[0]}")
print(f"Sum: {weights[0].sum():.6f}")

# Test maximize mode
print("\n" + "="*70)
print("MODE: MAXIMIZE (encourage diversity)")
print("="*70)

ent_max = EntropyRegularizer(mode='maximize', num_experts=E)
loss_max = ent_max(weights)

# Compute entropy manually
entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
max_entropy = torch.log(torch.tensor(float(E)))

print(f"Entropy: {entropy:.6f}")
print(f"Max entropy (log({E})): {max_entropy:.6f}")
print(f"Loss (max_entropy - entropy): {loss_max:.6f}")
print(f"Manual: {(max_entropy - entropy):.6f}")
print(f"Loss >= 0? {loss_max >= 0}")

# Test với uniform weights (should have loss ≈ 0)
uniform_weights = torch.ones(B, E) / E
loss_uniform = ent_max(uniform_weights)
print(f"\nUniform weights loss: {loss_uniform:.6f} (should be ≈ 0)")

# Test với peaked weights (should have high loss)
peaked_weights = torch.zeros(B, E)
peaked_weights[:, 0] = 1.0
loss_peaked = ent_max(peaked_weights)
print(f"Peaked weights loss: {loss_peaked:.6f} (should be high)")

# Test minimize mode
print("\n" + "="*70)
print("MODE: MINIMIZE (encourage sparsity)")
print("="*70)

ent_min = EntropyRegularizer(mode='minimize', num_experts=E)
loss_min = ent_min(weights)

print(f"Loss (entropy): {loss_min:.6f}")
print(f"Manual entropy: {entropy:.6f}")
print(f"Loss >= 0? {loss_min >= 0}")

# Test với uniform weights (should have high loss)
loss_uniform_min = ent_min(uniform_weights)
print(f"\nUniform weights loss: {loss_uniform_min:.6f} (should be high)")

# Test với peaked weights (should have low loss ≈ 0)
loss_peaked_min = ent_min(peaked_weights)
print(f"Peaked weights loss: {loss_peaked_min:.6f} (should be ≈ 0)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if loss_max >= 0 and loss_min >= 0:
    print("✓ All losses are non-negative")
else:
    print("✗ Some losses are negative!")

if loss_peaked > loss_uniform:
    print("✓ Maximize mode correctly penalizes peaked distribution")
else:
    print("✗ Maximize mode logic wrong")

if loss_uniform_min > loss_peaked_min:
    print("✓ Minimize mode correctly penalizes uniform distribution")
else:
    print("✗ Minimize mode logic wrong")
