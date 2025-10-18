#!/usr/bin/env python3
"""
Balanced test splits for CIFAR-100-LT WITHOUT data leakage.

Strategy:
1. Train: Long-tail distribution (UNCHANGED from current implementation)
2. CIFAR-100 Test (10,000 samples, 100 per class) → Split into:
   - Test: 80% = 8,000 samples (80 per class) - BALANCED
   - Val: 10% = 1,000 samples (10 per class) - BALANCED  
   - TuneV: 10% = 1,000 samples (10 per class) - BALANCED

Key Points:
- NO duplication → NO data leakage
- All test splits are BALANCED (equal samples per class)
- Reweighting happens at METRIC computation time (not in data)
"""

import numpy as np
import json
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class CIFAR100LTDataset(Dataset):
    """Custom Dataset wrapper for CIFAR-100-LT with flexible indexing."""
    
    def __init__(self, cifar_dataset, indices, transform=None):
        self.cifar_dataset = cifar_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual CIFAR index
        cifar_idx = self.indices[idx]
        image, label = self.cifar_dataset[cifar_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_cifar100_transforms():
    """Get CIFAR-100 transforms following paper specifications."""
    
    # Training transforms (basic augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
        )
    ])
    
    # Evaluation transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    return train_transform, eval_transform


def get_exponential_counts(num_classes: int = 100, imb_factor: float = 100, max_samples: int = 500) -> List[int]:
    """Generate exponential profile counts for long-tail distribution."""
    counts = []
    for cls_idx in range(num_classes):
        # Exponential decay: n_i = n_max * (IF)^(-i/(C-1))
        num = max_samples * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    return counts


def create_longtail_train(cifar_train_dataset, imb_factor: float = 100, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create long-tail training set using exponential profile.
    
    THIS IS UNCHANGED - keeps current implementation.
    """
    print(f"Creating CIFAR-100-LT training set (IF={imb_factor})...")
    
    np.random.seed(seed)
    targets = np.array(cifar_train_dataset.targets)
    num_classes = 100
    
    # Get target counts
    target_counts = get_exponential_counts(num_classes, imb_factor, 500)
    
    # Sample indices for each class
    train_indices = []
    actual_counts = []
    
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        
        # Random sample without replacement
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        train_indices.extend(sampled.tolist())
        actual_counts.append(num_to_sample)
    
    train_targets = targets[train_indices].tolist()
    
    print(f"  Total samples: {len(train_indices):,}")
    print(f"  Head class (0): {actual_counts[0]} samples")
    print(f"  Tail class (99): {actual_counts[-1]} samples")
    print(f"  Imbalance Factor: {actual_counts[0] / actual_counts[-1]:.1f}")
    
    return train_indices, train_targets, actual_counts


def split_balanced_test_8_1_1(
    cifar_test_dataset,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split CIFAR-100 test set (10,000 samples, 100 per class) into:
    - Test: 80 per class = 8,000 samples total
    - Val: 10 per class = 1,000 samples total
    - TuneV: 10 per class = 1,000 samples total
    
    All splits are BALANCED and DISJOINT (no overlap).
    
    Args:
        cifar_test_dataset: Original CIFAR-100 test dataset
        seed: Random seed for reproducibility
        
    Returns:
        (test_indices, test_targets, val_indices, val_targets, tunev_indices, tunev_targets)
    """
    print("\nSplitting CIFAR-100 test set (8:1:1 ratio)...")
    print("  Original: 100 samples per class (10,000 total)")
    print("  Split strategy:")
    print("    - Test:  80 per class (8,000 total)")
    print("    - Val:   10 per class (1,000 total)")
    print("    - TuneV: 10 per class (1,000 total)")
    
    np.random.seed(seed)
    test_targets_array = np.array(cifar_test_dataset.targets)
    num_classes = 100
    
    # Allocate per class
    test_per_class = 80
    val_per_class = 10
    tunev_per_class = 10
    
    test_indices = []
    val_indices = []
    tunev_indices = []
    
    # Split each class independently to ensure balance
    for cls in range(num_classes):
        # Get all indices for this class (should be 100)
        cls_indices = np.where(test_targets_array == cls)[0]
        
        # Verify we have 100 samples
        assert len(cls_indices) == 100, f"Class {cls} has {len(cls_indices)} samples, expected 100"
        
        # Shuffle
        np.random.shuffle(cls_indices)
        
        # Split: test (0-79) | val (80-89) | tunev (90-99)
        test_indices.extend(cls_indices[:test_per_class].tolist())
        val_indices.extend(cls_indices[test_per_class:test_per_class+val_per_class].tolist())
        tunev_indices.extend(cls_indices[test_per_class+val_per_class:].tolist())
    
    # Get corresponding targets
    test_targets = test_targets_array[test_indices].tolist()
    val_targets = test_targets_array[val_indices].tolist()
    tunev_targets = test_targets_array[tunev_indices].tolist()
    
    # Verify splits
    print("\n  ✅ Splits created successfully:")
    print(f"    Test:  {len(test_indices):,} samples ({len(set(test_indices))} unique)")
    print(f"    Val:   {len(val_indices):,} samples ({len(set(val_indices))} unique)")
    print(f"    TuneV: {len(tunev_indices):,} samples ({len(set(tunev_indices))} unique)")
    print(f"    Total: {len(test_indices) + len(val_indices) + len(tunev_indices):,} / 10,000")
    
    # Verify no overlap
    test_set = set(test_indices)
    val_set = set(val_indices)
    tunev_set = set(tunev_indices)
    
    assert len(test_set & val_set) == 0, "Test and Val overlap!"
    assert len(test_set & tunev_set) == 0, "Test and TuneV overlap!"
    assert len(val_set & tunev_set) == 0, "Val and TuneV overlap!"
    print("  ✅ No data leakage - all splits are disjoint")
    
    # Verify balance
    for name, targets in [("Test", test_targets), ("Val", val_targets), ("TuneV", tunev_targets)]:
        counts = Counter(targets)
        expected = len(targets) // num_classes
        assert all(c == expected for c in counts.values()), f"{name} is not balanced!"
    print("  ✅ All splits are perfectly balanced")
    
    return (test_indices, test_targets, val_indices, val_targets, tunev_indices, tunev_targets)


def compute_class_weights(train_class_counts: List[int]) -> np.ndarray:
    """
    Compute class weights based on train distribution for metric reweighting.
    
    These weights are used ONLY for computing metrics, NOT for data sampling.
    
    Args:
        train_class_counts: Number of samples per class in training set
        
    Returns:
        numpy array of weights (normalized to sum to 1)
    """
    train_counts = np.array(train_class_counts)
    total_train = train_counts.sum()
    weights = train_counts / total_train
    
    print("\n" + "="*60)
    print("CLASS WEIGHTS (for metric reweighting)")
    print("="*60)
    print(f"Total training samples: {total_train:,}")
    print(f"\nWeight distribution:")
    print(f"  Head class (0):  {train_counts[0]:4d} samples → weight = {weights[0]:.6f}")
    print(f"  Class 25:        {train_counts[25]:4d} samples → weight = {weights[25]:.6f}")
    print(f"  Class 50:        {train_counts[50]:4d} samples → weight = {weights[50]:.6f}")
    print(f"  Class 75:        {train_counts[75]:4d} samples → weight = {weights[75]:.6f}")
    print(f"  Tail class (99): {train_counts[99]:4d} samples → weight = {weights[-1]:.6f}")
    print(f"\nWeight ratio (head/tail): {weights[0]/weights[-1]:.1f}x")
    print("="*60)
    
    return weights


def analyze_distribution(indices: List[int], targets: List[int], name: str):
    """Analyze and print distribution statistics."""
    print(f"\n{'='*60}")
    print(f"{name.upper()} DISTRIBUTION")
    print(f"{'='*60}")
    
    target_counts = Counter(targets)
    sorted_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = sum(sorted_counts)
    head_count = sorted_counts[0]
    tail_count = sorted_counts[99]
    
    print(f"Total samples: {total:,}")
    print(f"Samples per class:")
    print(f"  Head class (0):  {head_count}")
    print(f"  Class 25:        {sorted_counts[25]}")
    print(f"  Class 50:        {sorted_counts[50]}")
    print(f"  Class 75:        {sorted_counts[75]}")
    print(f"  Tail class (99): {tail_count}")
    
    if tail_count > 0:
        print(f"Imbalance factor: {head_count/tail_count:.1f}")
    
    # Verify balance for test splits
    if name in ["Test", "Val", "TuneV"]:
        is_balanced = len(set(sorted_counts)) == 1
        if is_balanced:
            print(f"✅ Perfectly balanced: {sorted_counts[0]} samples per class")
        else:
            print(f"⚠️ Not balanced! Min: {min(sorted_counts)}, Max: {max(sorted_counts)}")


def save_splits_to_json(splits_dict: Dict, output_dir: str):
    """Save all splits and metadata to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to: {output_dir}")
    
    for split_name, data in splits_dict.items():
        filepath = output_path / f"{split_name}.json"
        
        # Convert numpy types to Python native types
        if hasattr(data, 'tolist'):
            data_to_save = data.tolist()
        elif isinstance(data, (list, tuple)):
            data_to_save = [float(x) if isinstance(x, (np.floating, float)) else int(x) 
                           for x in data]
        else:
            data_to_save = list(data)
            
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        if isinstance(data_to_save, list):
            print(f"  ✓ {split_name}.json ({len(data_to_save):,} items)")
        else:
            print(f"  ✓ {split_name}.json")


def create_cifar100_lt_balanced_test_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits_fixed",
    seed: int = 42
):
    """
    Create CIFAR-100-LT dataset with balanced test splits (8:1:1).
    
    This function creates:
    1. Long-tail training set (exponential distribution, IF=100)
    2. Balanced test set (80 per class = 8,000 total)
    3. Balanced validation set (10 per class = 1,000 total)
    4. Balanced tuneV set (10 per class = 1,000 total)
    5. Class weights for metric reweighting
    
    NO data duplication → NO data leakage
    All test splits are disjoint and balanced
    
    Args:
        imb_factor: Imbalance factor for training (default: 100)
        output_dir: Output directory for splits
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (datasets_dict, splits_dict, class_weights)
    """
    print("="*60)
    print("CREATING CIFAR-100-LT WITH BALANCED TEST SPLITS (8:1:1)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Imbalance Factor: {imb_factor}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Random Seed: {seed}")
    
    # Load original CIFAR-100
    print("\n" + "="*60)
    print("STEP 1: Loading CIFAR-100 datasets...")
    print("="*60)
    cifar_train = torchvision.datasets.CIFAR100(
        root='data', train=True, download=True, transform=None
    )
    cifar_test = torchvision.datasets.CIFAR100(
        root='data', train=False, download=True, transform=None
    )
    print(f"  ✓ Train: {len(cifar_train):,} samples")
    print(f"  ✓ Test:  {len(cifar_test):,} samples")
    
    # Create long-tail training set
    print("\n" + "="*60)
    print("STEP 2: Creating long-tail training set...")
    print("="*60)
    train_indices, train_targets, train_counts = create_longtail_train(
        cifar_train, imb_factor, seed
    )
    
    # Split balanced test set (8:1:1)
    print("\n" + "="*60)
    print("STEP 3: Splitting test set (8:1:1)...")
    print("="*60)
    (test_indices, test_targets, 
     val_indices, val_targets, 
     tunev_indices, tunev_targets) = split_balanced_test_8_1_1(cifar_test, seed)
    
    # Compute class weights for reweighting
    print("\n" + "="*60)
    print("STEP 4: Computing class weights...")
    print("="*60)
    class_weights = compute_class_weights(train_counts)
    
    # Analyze distributions
    print("\n" + "="*60)
    print("STEP 5: Analyzing distributions...")
    print("="*60)
    analyze_distribution(train_indices, train_targets, "Train")
    analyze_distribution(test_indices, test_targets, "Test")
    analyze_distribution(val_indices, val_targets, "Val")
    analyze_distribution(tunev_indices, tunev_targets, "TuneV")
    
    # Save all splits
    print("\n" + "="*60)
    print("STEP 6: Saving splits...")
    print("="*60)
    splits = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'val_indices': val_indices,
        'tunev_indices': tunev_indices,
        'class_weights': class_weights,
        'train_class_counts': train_counts
    }
    
    save_splits_to_json(splits, output_dir)
    
    # Create dataset objects
    print("\n" + "="*60)
    print("STEP 7: Creating dataset objects...")
    print("="*60)
    train_transform, eval_transform = get_cifar100_transforms()
    
    datasets = {
        'train': CIFAR100LTDataset(cifar_train, train_indices, train_transform),
        'test': CIFAR100LTDataset(cifar_test, test_indices, eval_transform),
        'val': CIFAR100LTDataset(cifar_test, val_indices, eval_transform),
        'tunev': CIFAR100LTDataset(cifar_test, tunev_indices, eval_transform)
    }
    
    for name, dataset in datasets.items():
        print(f"  ✓ {name}: {len(dataset):,} samples")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"  Train:  {len(datasets['train']):,} samples (long-tail, IF={imb_factor})")
    print(f"  Test:   {len(datasets['test']):,} samples (balanced)")
    print(f"  Val:    {len(datasets['val']):,} samples (balanced)")
    print(f"  TuneV:  {len(datasets['tunev']):,} samples (balanced)")
    
    print("\n📁 Files saved:")
    print(f"  {output_dir}/train_indices.json")
    print(f"  {output_dir}/test_indices.json")
    print(f"  {output_dir}/val_indices.json")
    print(f"  {output_dir}/tunev_indices.json")
    print(f"  {output_dir}/class_weights.json  👈 USE for metrics!")
    print(f"  {output_dir}/train_class_counts.json")
    
    print("\n⚠️  IMPORTANT:")
    print("  - Test/Val/TuneV are BALANCED (no duplication)")
    print("  - Use class_weights.json for reweighted metrics")
    print("  - No data leakage between splits")
    
    return datasets, splits, class_weights


if __name__ == "__main__":
    # Create the dataset
    datasets, splits, class_weights = create_cifar100_lt_balanced_test_splits(
        imb_factor=100,
        output_dir="data/cifar100_lt_if100_splits_fixed",
        seed=42
    )
    
    print("\n" + "="*60)
    print("READY FOR TRAINING!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train experts with: python train_experts.py")
    print("  2. Train gating with: python train_gating.py")
    print("  3. Evaluate with reweighted metrics")
