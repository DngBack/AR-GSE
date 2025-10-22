"""
Test loss computation để debug val loss issue
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/duong.xuan.bach/AR-GSE')

from src.models.gating_losses import GatingLoss, MixtureNLLLoss, LoadBalancingLoss

print("="*70)
print("TESTING LOSS COMPUTATION")
print("="*70)

# Mock data
B, E, C = 128, 3, 100
posteriors = F.softmax(torch.randn(B, E, C) + 2, dim=-1)  # Shift để có prob cao
targets = torch.randint(0, C, (B,))
weights = F.softmax(torch.randn(B, E) * 0.5, dim=-1)

print(f"\nData: B={B}, E={E}, C={C}")
print(f"Weights stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

# Test 1: Pure NLL
print("\n" + "="*70)
print("TEST 1: Mixture NLL")
print("="*70)

nll_fn = MixtureNLLLoss()
nll = nll_fn(posteriors, weights, targets)
print(f"NLL: {nll.item():.6f}")

# Verify manually
mixture_post = torch.sum(weights.unsqueeze(-1) * posteriors, dim=1)
true_probs = torch.gather(mixture_post, dim=1, index=targets.unsqueeze(1)).squeeze(1)
manual_nll = -torch.log(true_probs + 1e-8).mean()
print(f"Manual NLL: {manual_nll.item():.6f}")
print(f"Match: {torch.allclose(nll, manual_nll, atol=1e-5)}")

# Test 2: Load-balancing (top-1 for dense)
print("\n" + "="*70)
print("TEST 2: Load-Balancing Loss")
print("="*70)

lb_fn = LoadBalancingLoss(alpha=1e-2)
lb = lb_fn(weights, top_k=1)
print(f"LB loss (top_k=1): {lb.item():.6f}")

# Manual computation
expert_indices = weights.argmax(dim=-1)
f = torch.zeros(E)
for i in range(E):
    f[i] = (expert_indices == i).float().mean()
P = weights.mean(dim=0)
manual_lb = 1e-2 * E * torch.sum(f * P)
print(f"Manual LB: {manual_lb.item():.6f}")
print(f"f (routing fraction): {f}")
print(f"P (avg weight): {P}")
print(f"Match: {torch.allclose(lb, manual_lb, atol=1e-5)}")

# Test 3: Combined loss (without LB)
print("\n" + "="*70)
print("TEST 3: Combined Loss (Dense Routing - NO LB)")
print("="*70)

loss_fn = GatingLoss(
    lambda_lb=1e-2,
    lambda_h=0.01,
    use_load_balancing=False,  # Dense routing
    use_entropy_reg=True,
    top_k=1,
    entropy_mode='maximize'
)

total_loss, components = loss_fn(posteriors, weights, targets, return_components=True)
print(f"Total loss: {total_loss.item():.6f}")
print(f"Components:")
for k, v in components.items():
    print(f"  {k}: {v:.6f}")

# Test 4: Combined loss (with LB - for sparse)
print("\n" + "="*70)
print("TEST 4: Combined Loss (Sparse Routing - WITH LB)")
print("="*70)

loss_fn_sparse = GatingLoss(
    lambda_lb=1e-2,
    lambda_h=0.01,
    use_load_balancing=True,  # Sparse routing
    use_entropy_reg=True,
    top_k=2,
    entropy_mode='maximize'
)

total_loss_sparse, components_sparse = loss_fn_sparse(posteriors, weights, targets, return_components=True)
print(f"Total loss: {total_loss_sparse.item():.6f}")
print(f"Components:")
for k, v in components_sparse.items():
    print(f"  {k}: {v:.6f}")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\n1. Pure NLL: {nll.item():.6f}")
print(f"   → This should be in range [0.01, 5.0] for reasonable training")
print(f"   → Min true_prob: {true_probs.min():.6f}, Max: {true_probs.max():.6f}")

print(f"\n2. Load-balancing (if enabled): {lb.item():.6f}")
print(f"   → For dense routing, this should be DISABLED")
print(f"   → If enabled, typical range: [0.001, 0.1]")

print(f"\n3. Entropy regularization: {components.get('entropy', 0.0):.6f}")
print(f"   → Should be small: [-0.01, 0.01]")

print(f"\n4. Dense routing total: {total_loss.item():.6f}")
print(f"   → Should be close to NLL + small entropy term")

print(f"\n5. Sparse routing total: {total_loss_sparse.item():.6f}")
print(f"   → NLL + LB + Entropy")

if total_loss.item() > 5.0:
    print("\n⚠️  WARNING: Loss is too high!")
    print("   Check for numerical issues or incorrect loss weights")
