#!/usr/bin/env python3
"""
Create CIFAR-100-LT splits using the REWEIGHTING approach (no duplication).

This script creates:
1. Long-tail training set (unchanged)
2. Balanced val/tunev/test sets (disjoint, no duplication)
3. Class weights for metric reweighting

Usage:
    python create_splits_reweighted.py
    python create_splits_reweighted.py --imb-factor 100 --val-ratio 0.15 --tunev-ratio 0.10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.reweighted_datasets import create_reweighted_cifar100_lt_splits


def main():
    parser = argparse.ArgumentParser(
        description="Create CIFAR-100-LT splits with reweighting (no duplication)"
    )
    
    parser.add_argument(
        '--imb-factor',
        type=float,
        default=100,
        help='Imbalance factor for long-tail distribution (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cifar100_lt_if100_splits_reweighted',
        help='Output directory for splits (default: data/cifar100_lt_if100_splits_reweighted)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation ratio from test set (default: 0.15 = 15%%)'
    )
    
    parser.add_argument(
        '--tunev-ratio',
        type=float,
        default=0.10,
        help='TuneV ratio from test set (default: 0.10 = 10%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.val_ratio + args.tunev_ratio >= 1.0:
        print("ERROR: val_ratio + tunev_ratio must be < 1.0")
        sys.exit(1)
    
    # Create splits
    print(f"\nCreating CIFAR-100-LT splits with:")
    print(f"  Imbalance Factor: {args.imb_factor}")
    print(f"  Val Ratio: {args.val_ratio:.1%}")
    print(f"  TuneV Ratio: {args.tunev_ratio:.1%}")
    print(f"  Test Ratio: {1.0 - args.val_ratio - args.tunev_ratio:.1%}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Random Seed: {args.seed}")
    
    datasets, splits, class_weights = create_reweighted_cifar100_lt_splits(
        imb_factor=args.imb_factor,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        tunev_ratio=args.tunev_ratio,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train experts: python train_experts.py")
    print("2. Use class_weights.json for reweighted metrics during evaluation")
    print("\nFiles created:")
    print(f"  {args.output_dir}/train_indices.json")
    print(f"  {args.output_dir}/val_indices.json")
    print(f"  {args.output_dir}/tunev_indices.json")
    print(f"  {args.output_dir}/test_indices.json")
    print(f"  {args.output_dir}/class_weights.json  👈 USE THIS for metrics!")
    print(f"  {args.output_dir}/train_class_counts.json")


if __name__ == "__main__":
    main()
