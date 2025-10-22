"""
Quick test: Check if Balanced vs Worst produce different errors.
"""
import numpy as np

# Simulate metrics with 2 groups (head=50 classes, tail=50 classes)
print("="*70)
print("Testing Balanced vs Worst Error Computation")
print("="*70)

# Example: Head group has lower error, Tail group has higher error
head_error = 0.30
tail_error = 0.70
group_errors = np.array([head_error, tail_error])

print(f"\nGroup errors: {group_errors}")
print(f"  Head (50 classes): {head_error:.2f}")
print(f"  Tail (50 classes): {tail_error:.2f}")

# Compute overall error (weighted by group sizes)
overall_error = np.mean(group_errors)  # Simple mean if groups equal size

# Compute Balanced error
balanced_error = np.mean(group_errors)

# Compute Worst error
worst_error = np.max(group_errors)

print(f"\nError by objective:")
print(f"  Overall (standard):  {overall_error:.4f}")
print(f"  Balanced (mean):     {balanced_error:.4f}")
print(f"  Worst (max):         {worst_error:.4f}")

print(f"\nDifference:")
print(f"  Worst - Balanced: {worst_error - balanced_error:.4f} ({(worst_error - balanced_error)/balanced_error * 100:+.1f}%)")

print("\n" + "="*70)
print("Expected behavior:")
print("  - Balanced should optimize mean of group errors → balances both groups")
print("  - Worst should optimize max → focuses on tail (worst group)")
print("  - Worst error should be HIGHER than Balanced error")
print("="*70)
