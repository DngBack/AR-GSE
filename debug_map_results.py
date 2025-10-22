"""
Quick debug script to check MAP results
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_dir_simple = Path('./results/map_simple/cifar100_lt_if100')
results_dir_old = Path('./results/map_plugin/cifar100_lt_if100')

# Try simplified first
if results_dir_simple.exists():
    results_dir = results_dir_simple
    params_dir = Path('./checkpoints/map_simple/cifar100_lt_if100')
    print(f"Using SIMPLIFIED results from: {results_dir}")
elif results_dir_old.exists():
    results_dir = results_dir_old  
    params_dir = Path('./checkpoints/map_plugin/cifar100_lt_if100')
    print(f"Using OLD results from: {results_dir}")
else:
    print("ERROR: No results found!")
    exit(1)

print("="*70)
print("🔍 MAP RESULTS DEBUG")
print("="*70)

# RC curve
rc_path = results_dir / 'rc_curve.json'
with open(rc_path, 'r') as f:
    rc_data = json.load(f)

print("\nRC CURVE DATA:")
print(f"  Keys: {rc_data.keys()}")
print(f"  AURC: {rc_data['aurc']}")
print(f"  Number of points: {len(rc_data['rejection_rates'])}")

rej = np.array(rc_data['rejection_rates'])
err = np.array(rc_data['selective_errors'])

print(f"\n  Rejection rates:")
print(f"    Min: {rej.min():.4f}")
print(f"    Max: {rej.max():.4f}")
print(f"    Range: [{rej[0]:.4f}, {rej[-1]:.4f}]")

print(f"\n  Selective errors:")
print(f"    Min: {err.min():.4f}")
print(f"    Max: {err.max():.4f}")
print(f"    Mean: {err.mean():.4f}")
print(f"    Std: {err.std():.4f}")

# Check if all zeros
if err.max() == 0.0:
    print("\n  ⚠️  WARNING: All errors are ZERO! This is a BUG.")
    print("  Possible causes:")
    print("    - All samples rejected")
    print("    - Wrong margin computation")
    print("    - Bug in selective_error calculation")

# Sample points
print("\n  Sample points:")
for target_rej in [0.0, 0.1, 0.2, 0.5]:
    idx = np.argmin(np.abs(rej - target_rej))
    print(f"    Rej≈{target_rej:.1f}: error={err[idx]:.4f}")

# Group errors
if 'group_errors' in rc_data:
    group_errors = rc_data['group_errors']
    print(f"\n  Group errors (first point):")
    print(f"    Head: {group_errors[0][0]:.4f}")
    print(f"    Tail: {group_errors[0][1]:.4f}")

# Evaluation
eval_path = results_dir / 'evaluation_test.json'
with open(eval_path, 'r') as f:
    eval_data = json.load(f)

print("\n" + "="*70)
print("EVALUATION DATA:")
print("="*70)

cm = eval_data['classification_metrics']
print(f"\nClassification (no rejection):")
print(f"  Accuracy: {cm['accuracy']:.4f}")
print(f"  Top-5 Acc: {cm['top5_accuracy']:.4f}")

print(f"\nGroup-wise:")
for g_id, gm in eval_data['group_metrics'].items():
    print(f"  Group {g_id}: Acc={gm['accuracy']:.4f}, Count={gm['count']}")

print(f"\nRC curve from eval:")
print(f"  AURC: {eval_data['rc_curve']['aurc']}")

# Parameters
with open(params_dir / 'map_parameters.json', 'r') as f:
    params = json.load(f)

print("\n" + "="*70)
print("PARAMETERS:")
print("="*70)

if 'threshold' in params:
    # Simplified version
    print(f"\n  threshold = {params['threshold']:.3f}")
    print(f"  γ = {params['gamma']:.3f}")
    print(f"  Objective: {params['objective']}")
else:
    # Old version
    print(f"\n  λ = {params['lambda']:.3f}")
    print(f"  γ = {params['gamma']:.3f}")
    print(f"  ν = {params['nu']:.3f}")
    print(f"  Objective: {params['objective']}")
    
    alpha = np.array(params['alpha'])
    mu = np.array(params['mu'])
    
    print(f"\n  α (per-class weights):")
    print(f"    Shape: {alpha.shape}")
    print(f"    Range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"    Mean: {alpha.mean():.4f}")
    print(f"    Head mean (0-49): {alpha[:50].mean():.4f}")
    print(f"    Tail mean (50-99): {alpha[50:].mean():.4f}")
    
    print(f"\n  μ (thresholds):")
    print(f"    μ[0] (head): {mu[0]:.4f}")
    print(f"    μ[1] (tail): {mu[1]:.4f}")
    print(f"    λ = μ[0] - μ[1]: {mu[0] - mu[1]:.4f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if err.max() == 0.0:
    print("\n❌ CRITICAL BUG: All selective errors are ZERO!")
    print("\nPossible issues:")
    print("  1. All samples are rejected (coverage = 0)")
    print("  2. Margin computation is wrong (all margins > 0)")
    print("  3. Bug in compute_selective_metrics()")
    print("\nNext steps:")
    print("  - Check margin distribution")
    print("  - Check acceptance rates")
    print("  - Add debug prints in map_selector.py")
else:
    print("\n✅ Errors look reasonable")

print("\n🎉 Done!")
