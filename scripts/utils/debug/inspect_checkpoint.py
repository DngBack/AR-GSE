#!/usr/bin/env python3
"""
Inspect checkpoint structure to understand the model architecture.
"""

import torch
from pathlib import Path

checkpoint_path = "checkpoints/experts/cifar100_lt_if100/best_ce_baseline.pth"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint keys: {checkpoint.keys()}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print(f"\nNumber of parameters: {len(state_dict)}")
print(f"\nFirst 20 parameter names:")
for i, (k, v) in enumerate(list(state_dict.items())[:20]):
    print(f"  {k}: {v.shape}")

print(f"\nLast 5 parameter names:")
for k, v in list(state_dict.items())[-5:]:
    print(f"  {k}: {v.shape}")

# Find fc layer
print(f"\nClassifier layers (fc/linear):")
for k, v in state_dict.items():
    if 'fc' in k.lower() or 'linear' in k.lower() or 'classifier' in k.lower():
        print(f"  {k}: {v.shape}")

# Check for special keys
if 'temperature' in state_dict:
    print(f"\n✓ Has temperature parameter: {state_dict['temperature']}")
