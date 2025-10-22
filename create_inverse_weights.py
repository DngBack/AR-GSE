"""
Create CORRECT inverse frequency class weights for reweighting evaluation.

For long-tail CIFAR-100 (IF=100):
- Train distribution: Imbalanced (500 samples for class 0, 5 for class 99)
- Test distribution: Balanced (~10 samples per class)

Purpose: When evaluating on balanced test, use inverse frequency weights
to reflect importance under the original imbalanced distribution.
"""
import json
import numpy as np
from pathlib import Path

# Load train class counts
splits_dir = Path('data/cifar100_lt_if100_splits_fixed')
with open(splits_dir / 'train_class_counts.json') as f:
    train_counts = json.load(f)

train_counts = np.array(train_counts)
N_train = train_counts.sum()

print("="*70)
print("CREATING INVERSE FREQUENCY CLASS WEIGHTS")
print("="*70)

print(f"\nTrain distribution:")
print(f"  Total samples: {N_train}")
print(f"  Class 0 (most frequent): {train_counts[0]} samples")
print(f"  Class 99 (least frequent): {train_counts[-1]} samples")
print(f"  Imbalance factor: {train_counts[0] / train_counts[-1]:.1f}")

# Compute inverse frequency weights
freq = train_counts / N_train
inv_freq = 1.0 / freq

# Normalize to sum = num_classes (common practice)
# This keeps weights on reasonable scale
inv_freq_normalized = inv_freq / inv_freq.mean()

print(f"\nInverse frequency weights:")
print(f"  Class 0 (head): {inv_freq_normalized[0]:.6f} (low weight, frequent)")
print(f"  Class 99 (tail): {inv_freq_normalized[-1]:.6f} (high weight, rare)")
print(f"  Ratio tail/head: {inv_freq_normalized[-1] / inv_freq_normalized[0]:.2f}x")
print(f"  Mean: {inv_freq_normalized.mean():.6f}")
print(f"  Sum: {inv_freq_normalized.sum():.2f}")

# Define groups: 69 head, 31 tail (as user specified)
NUM_HEAD = 69
NUM_TAIL = 31

head_weights = inv_freq_normalized[:NUM_HEAD]
tail_weights = inv_freq_normalized[NUM_HEAD:]

print(f"\nGroup statistics:")
print(f"  Head (classes 0-{NUM_HEAD-1}): {len(head_weights)} classes")
print(f"    Mean weight: {head_weights.mean():.6f}")
print(f"    Range: [{head_weights.min():.6f}, {head_weights.max():.6f}]")
print(f"  Tail (classes {NUM_HEAD}-99): {len(tail_weights)} classes")
print(f"    Mean weight: {tail_weights.mean():.6f}")
print(f"    Range: [{tail_weights.min():.6f}, {tail_weights.max():.6f}]")
print(f"  Tail mean / Head mean: {tail_weights.mean() / head_weights.mean():.2f}x")

# Save
output_path = splits_dir / 'inverse_class_weights.json'
with open(output_path, 'w') as f:
    json.dump(inv_freq_normalized.tolist(), f, indent=2)

print(f"\n✅ Saved to: {output_path}")

# Also save group boundaries
group_config = {
    'num_groups': 2,
    'group_boundaries': [NUM_HEAD],  # [0, 69) = head, [69, 100) = tail
    'group_names': ['head', 'tail'],
    'group_sizes': [NUM_HEAD, NUM_TAIL]
}

config_path = splits_dir / 'group_config.json'
with open(config_path, 'w') as f:
    json.dump(group_config, f, indent=2)

print(f"✅ Saved group config to: {config_path}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
When computing metrics on BALANCED test set:

  e_k(w) = Σ(w_i * 1[wrong] * A_i) / Σ(w_i * A_i)

- w_i: Inverse frequency weight (tail classes get higher weight)
- This gives more importance to tail classes
- Reflects performance under original imbalanced distribution
- Different from uniform evaluation (which treats all classes equally)

Example:
  - Without reweight: Error on 1 head sample = Error on 1 tail sample
  - With reweight: Error on 1 tail sample = 100x more important!
""")
print("="*70)
