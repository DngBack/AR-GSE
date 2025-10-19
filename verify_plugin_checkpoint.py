"""
Verify plugin checkpoint and analyze parameters
"""
import torch
import json
from pathlib import Path

# Load checkpoint
ckpt_path = "checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt"
print(f"📂 Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("\n" + "="*60)
print("CHECKPOINT CONTENTS")
print("="*60)
for key in ckpt.keys():
    if isinstance(ckpt[key], torch.Tensor):
        print(f"{key:30s}: shape={list(ckpt[key].shape)}, dtype={ckpt[key].dtype}")
    else:
        print(f"{key:30s}: type={type(ckpt[key])}")

print("\n" + "="*60)
print("PLUGIN PARAMETERS")
print("="*60)

# Alpha (acceptance parameters)
alpha = ckpt['alpha']
print(f"α (alpha): {alpha.tolist()}")
print(f"  → Group 0 (head): α={alpha[0]:.4f}")
print(f"  → Group 1 (tail): α={alpha[1]:.4f}")
print(f"  → Interpretation: Higher α means more selective (higher threshold)")

# Mu (threshold adjustments)
mu = ckpt['mu']
print(f"\nμ (mu): {mu.tolist()}")
print(f"  → Group 0 (head): μ={mu[0]:.4f}")
print(f"  → Group 1 (tail): μ={mu[1]:.4f}")
print(f"  → Interpretation: Positive μ increases threshold (more selective)")

# Class to group mapping
class_to_group = ckpt['class_to_group']
num_head = (class_to_group == 0).sum().item()
num_tail = (class_to_group == 1).sum().item()
print(f"\nClass to Group mapping:")
print(f"  → Group 0 (head): {num_head} classes")
print(f"  → Group 1 (tail): {num_tail} classes")

# Best score
if 'best_score' in ckpt:
    print(f"\nBest validation score: {ckpt['best_score']:.4f}")
    print(f"  → This is the error achieved during plugin training")

# Threshold
if 'threshold' in ckpt:
    threshold = ckpt['threshold']
    if isinstance(threshold, list):
        print(f"\nThreshold c*: {threshold}")
    else:
        print(f"\nThreshold c*: {threshold:.4f}")
    print(f"  → This is the optimal rejection cost found during training")
if 't_group' in ckpt:
    t_group = ckpt['t_group']
    print(f"\nPer-group thresholds: {t_group}")
    print(f"  → Group 0 (head): {t_group[0] if isinstance(t_group, list) else t_group}")
    print(f"  → Group 1 (tail): {t_group[1] if isinstance(t_group, list) and len(t_group) > 1 else 'N/A'}")

print("\n" + "="*60)
print("PARAMETER ANALYSIS")
print("="*60)

# Check if parameters are identity (not optimized)
if torch.allclose(alpha, torch.ones(2)):
    print("⚠️  WARNING: α = [1, 1] → Identity, not optimized!")
else:
    print("✅ α has been optimized (not identity)")

if torch.allclose(mu, torch.zeros(2)):
    print("⚠️  WARNING: μ = [0, 0] → No adjustment!")
else:
    print("✅ μ has been optimized")
    if mu[0] < 0 and mu[1] > 0:
        print("   → Head group: more lenient (μ < 0)")
        print("   → Tail group: more selective (μ > 0)")
        print("   → This is EXPECTED for long-tail (protect tail from errors)")

# Compute effective threshold difference
# For sample in group k: threshold = Σ_y [(1/α_k - μ_k) * η_y]
head_factor = 1.0/alpha[0].item() - mu[0].item()
tail_factor = 1.0/alpha[1].item() - mu[1].item()
print(f"\nEffective threshold factors:")
print(f"  → Head (1/α - μ): {head_factor:.4f}")
print(f"  → Tail (1/α - μ): {tail_factor:.4f}")
print(f"  → Difference: {tail_factor - head_factor:.4f}")
if tail_factor > head_factor:
    print("  → Tail has HIGHER threshold → More selective (reject more)")
    print("  → This may cause low coverage for tail group!")

print("\n" + "="*60)
print("POTENTIAL ISSUES")
print("="*60)

issues = []

# Check if α = 1 (no optimization)
if torch.allclose(alpha, torch.ones(2)):
    issues.append("α not optimized (identity)")

# Check if μ causes extreme selectivity
if abs(mu[0].item()) > 1.0 or abs(mu[1].item()) > 1.0:
    issues.append(f"Large μ values (|μ| > 1.0): may cause extreme behavior")

# Check if tail is too selective
if tail_factor > head_factor + 0.5:
    issues.append("Tail group significantly more selective than head")
    issues.append("→ May explain why worst-group has low coverage/high risk")

if len(issues) == 0:
    print("✅ No major issues detected")
else:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if torch.allclose(alpha, torch.ones(2)) and not torch.allclose(mu, torch.zeros(2)):
    print("1. α = [1, 1] but μ ≠ [0, 0]")
    print("   → Plugin training may have fixed α and only optimized μ")
    print("   → This is OK if intended, but limits flexibility")

if tail_factor > head_factor + 0.5:
    print("2. Tail group is much more selective than head")
    print("   → Try different plugin training with:")
    print("      - Balanced objective (instead of worst)")
    print("      - Lower target coverage")
    print("      - Different initialization")

print("\n📝 Next steps:")
print("1. Check if plugin training converged properly")
print("2. Compare with gating-only checkpoint (before plugin)")
print("3. Try re-training plugin with different hyperparameters")
