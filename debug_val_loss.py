"""
Debug validation loss issue
"""

import torch
import torch.nn.functional as F

# Mock data giống như trong training
B, E, C = 128, 3, 100

# Tạo posterior giả (giống expert outputs)
posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
targets = torch.randint(0, C, (B,))

# Tạo gating weights (uniform-ish)
weights = F.softmax(torch.randn(B, E) * 0.1, dim=-1)

print("="*70)
print("DEBUGGING VAL LOSS ISSUE")
print("="*70)
print(f"Batch: {B}, Experts: {E}, Classes: {C}")
print(f"Weights shape: {weights.shape}")
print(f"Weights mean per expert: {weights.mean(dim=0)}")
print(f"Weights sum per sample: {weights.sum(dim=1).mean():.4f}")

# 1. Compute mixture NLL
eps = 1e-8
mixture_posterior = torch.sum(weights.unsqueeze(-1) * posteriors, dim=1)  # [B, C]
true_probs = torch.gather(mixture_posterior, dim=1, index=targets.unsqueeze(1)).squeeze(1)
nll = -torch.log(true_probs + eps).mean()

print(f"\n1. Mixture NLL: {nll.item():.6f}")
print(f"   Min prob: {true_probs.min().item():.6f}")
print(f"   Max prob: {true_probs.max().item():.6f}")
print(f"   Mean prob: {true_probs.mean().item():.6f}")

# 2. Load-balancing loss (dense routing)
alpha = 1e-2

# f_i: fraction routed to each expert (dense = tất cả)
# Với dense routing, không có "routing" rõ ràng
# Switch paper dùng cho sparse routing (top-1)
# Với dense, f_i nên được tính dựa trên weight magnitude

# Cách 1: Dùng weight average (như hiện tại)
P = weights.mean(dim=0)  # [E]
print(f"\n2a. Load-balancing (average weights):")
print(f"   P (avg weight per expert): {P}")

# Với dense routing, không có f_i rõ ràng
# Giả sử tất cả expert đều được dùng: f_i = 1
f_dense = torch.ones(E)
lb_loss_dense = alpha * E * torch.sum(f_dense * P)
print(f"   f (assuming all used): {f_dense}")
print(f"   LB loss: {lb_loss_dense.item():.6f}")

# Cách 2: Top-1 (như trong code)
expert_indices = weights.argmax(dim=-1)  # [B]
f_top1 = torch.zeros(E)
for i in range(E):
    f_top1[i] = (expert_indices == i).float().mean()
lb_loss_top1 = alpha * E * torch.sum(f_top1 * P)
print(f"\n2b. Load-balancing (top-1 routing):")
print(f"   f (top-1 selection): {f_top1}")
print(f"   LB loss: {lb_loss_top1.item():.6f}")

# 3. Entropy regularization
entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()
lambda_h = 0.01
ent_loss = -lambda_h * entropy  # maximize → minimize negative
print(f"\n3. Entropy regularization:")
print(f"   Entropy: {entropy.item():.6f}")
print(f"   Ent loss (lambda_h * -H): {ent_loss.item():.6f}")

# 4. Total loss
total_loss = nll + lb_loss_top1 + ent_loss
print(f"\n4. TOTAL LOSS:")
print(f"   NLL: {nll.item():.6f}")
print(f"   + LB: {lb_loss_top1.item():.6f}")
print(f"   + Ent: {ent_loss.item():.6f}")
print(f"   = Total: {total_loss.item():.6f}")

print("\n" + "="*70)
print("ISSUE IDENTIFIED:")
print("="*70)

if lb_loss_top1.item() > 1.0:
    print("⚠️  Load-balancing loss is TOO LARGE!")
    print(f"   LB loss = alpha * E * sum(f * P)")
    print(f"   = {alpha} * {E} * {torch.sum(f_top1 * P).item():.4f}")
    print(f"   = {lb_loss_top1.item():.6f}")
    print("\n   For DENSE routing, load-balancing should be DISABLED")
    print("   or computed differently!")

if nll.item() > 5.0:
    print("⚠️  NLL is too high - possible numerical issue")
    print(f"   Check for very small probabilities: min={true_probs.min().item():.1e}")
