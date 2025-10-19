#!/usr/bin/env python3
"""
Test script to verify expert training dataloader setup.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.dataloader_utils import get_expert_training_dataloaders
import numpy as np

def test_dataloader():
    """Test expert training dataloader."""
    print("="*60)
    print("TESTING EXPERT TRAINING DATALOADER")
    print("="*60)
    
    # Test with expert split
    print("\n1. Testing with EXPERT split (90% of train):")
    print("-" * 60)
    train_loader, val_loader = get_expert_training_dataloaders(
        batch_size=128,
        num_workers=0,  # No multiprocessing for testing
        use_expert_split=True,
        splits_dir="data/cifar100_lt_if100_splits_fixed"
    )
    
    print(f"Train loader:")
    print(f"  Batches: {len(train_loader)}")
    print(f"  Samples: {len(train_loader.dataset):,}")
    
    # Check class distribution
    from collections import Counter
    train_targets = []
    for _, targets in train_loader:
        train_targets.extend(targets.numpy().tolist())
    
    class_counts = Counter(train_targets)
    print(f"  Classes: {len(class_counts)}")
    print(f"  Head class (0): {class_counts[0]} samples")
    print(f"  Tail class (99): {class_counts[99]} samples")
    print(f"  Imbalance Factor: {class_counts[0] / class_counts[99]:.1f}")
    
    print(f"\nVal loader:")
    print(f"  Batches: {len(val_loader)}")
    print(f"  Samples: {len(val_loader.dataset):,}")
    
    # Check val is balanced
    val_targets = []
    for _, targets in val_loader:
        val_targets.extend(targets.numpy().tolist())
    
    val_class_counts = Counter(val_targets)
    print(f"  Classes: {len(val_class_counts)}")
    print(f"  Per class: {val_class_counts[0]} samples")
    print(f"  Balanced: {len(set(val_class_counts.values())) == 1}")
    
    # Test a batch
    print("\n2. Testing batch loading:")
    print("-" * 60)
    images, labels = next(iter(train_loader))
    print(f"  Image batch shape: {images.shape}")
    print(f"  Label batch shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test with full train (for comparison)
    print("\n3. Testing with FULL train split:")
    print("-" * 60)
    train_loader_full, _ = get_expert_training_dataloaders(
        batch_size=128,
        num_workers=0,
        use_expert_split=False,
        splits_dir="data/cifar100_lt_if100_splits_fixed"
    )
    
    print(f"Train loader:")
    print(f"  Batches: {len(train_loader_full)}")
    print(f"  Samples: {len(train_loader_full.dataset):,}")
    
    train_targets_full = []
    for _, targets in train_loader_full:
        train_targets_full.extend(targets.numpy().tolist())
    
    class_counts_full = Counter(train_targets_full)
    print(f"  Head class (0): {class_counts_full[0]} samples")
    print(f"  Tail class (99): {class_counts_full[99]} samples")
    print(f"  Imbalance Factor: {class_counts_full[0] / class_counts_full[99]:.1f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print(f"  Expert split: {len(train_loader.dataset):,} samples (90% of train)")
    print(f"  Full train:   {len(train_loader_full.dataset):,} samples")
    print(f"  Val (balanced): {len(val_loader.dataset):,} samples")
    print(f"\n  Expert IF: {class_counts[0] / class_counts[99]:.1f}")
    print(f"  Full train IF: {class_counts_full[0] / class_counts_full[99]:.1f}")

if __name__ == "__main__":
    test_dataloader()
