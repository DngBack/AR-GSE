#!/usr/bin/env python3
"""
Create CIFAR-100-LT splits with BALANCED test sets (8:1:1 ratio).

This replaces the old duplication-based approach with:
- Train: Long-tail (unchanged)
- Test: 80 per class = 8,000 samples (balanced, no duplication)
- Val: 10 per class = 1,000 samples (balanced, no duplication)
- TuneV: 10 per class = 1,000 samples (balanced, no duplication)

NO data leakage, NO duplication.
Reweighting happens at metric computation time.

Usage:
    python create_splits_fixed.py
    python create_splits_fixed.py --imb-factor 100 --output data/my_splits
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.balanced_test_splits import create_cifar100_lt_balanced_test_splits


def main():
    parser = argparse.ArgumentParser(
        description="Create CIFAR-100-LT with balanced test splits (8:1:1)"
    )
    
    parser.add_argument(
        '--imb-factor',
        type=float,
        default=100,
        help='Imbalance factor for training set (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/cifar100_lt_if100_splits_fixed',
        help='Output directory (default: data/cifar100_lt_if100_splits_fixed)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CIFAR-100-LT DATA PREPARATION (BALANCED TEST - NO DUPLICATION)")
    print("="*80)
    print("\nConfiguration:")
    print(f"  • Imbalance Factor: {args.imb_factor}")
    print(f"  • Output Directory: {args.output}")
    print(f"  • Random Seed: {args.seed}")
    print(f"\nSplit Ratios (from CIFAR-100 test set):")
    print(f"  • Test:  80% (8,000 samples, 80 per class)")
    print(f"  • Val:   10% (1,000 samples, 10 per class)")
    print(f"  • TuneV: 10% (1,000 samples, 10 per class)")
    print(f"\n{'='*80}\n")
    
    try:
        # Create splits
        datasets, splits, class_weights = create_cifar100_lt_balanced_test_splits(
            imb_factor=args.imb_factor,
            output_dir=args.output,
            seed=args.seed
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        
        print("\n📊 Datasets Created:")
        for name, dataset in datasets.items():
            print(f"  {name:>6}: {len(dataset):>6,} samples")
        
        print("\n📁 Files Saved to: " + args.output)
        output_path = Path(args.output)
        if output_path.exists():
            for json_file in sorted(output_path.glob("*.json")):
                print(f"  ✓ {json_file.name}")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1️⃣  Train Expert Models:")
        print("   python train_experts.py")
        print("\n2️⃣  Train Gating Network:")
        print("   python train_gating.py --mode pretrain")
        print("   python train_gating.py --mode selective")
        print("\n3️⃣  Train AR-GSE:")
        print("   python train_argse.py")
        print("\n4️⃣  Evaluate with Reweighted Metrics:")
        print("   python evaluate_argse.py")
        print("\n⚠️  IMPORTANT:")
        print(f"   Use {args.output}/class_weights.json for reweighted metrics!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nPlease check:")
        print("  • CIFAR-100 data is accessible")
        print("  • Required dependencies are installed (numpy, torchvision)")
        print("  • Sufficient disk space")
        print("  • Write permissions for output directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
