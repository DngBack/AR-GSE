#!/usr/bin/env python3
"""
Create CIFAR-100-LT splits with BALANCED test sets (8:1:1 ratio).

This replaces the old duplication-based approach with:
- Train: Long-tail (unchanged)
  - Option to split into Expert (90%) and Gating (10%) - SAME imbalance ratio
- Test: 80 per class = 8,000 samples (balanced, no duplication)
- Val: 10 per class = 1,000 samples (balanced, no duplication)
- TuneV: 10 per class = 1,000 samples (balanced, no duplication)

NO data leakage, NO duplication.
Reweighting happens at metric computation time.

Usage:
    python create_splits_fixed.py
    python create_splits_fixed.py --imb-factor 100 --output data/my_splits
    python create_splits_fixed.py --split-train  # Split train into expert/gating
    python create_splits_fixed.py --visualize    # Create visualizations
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.balanced_test_splits import create_cifar100_lt_balanced_test_splits
from src.visualize.splits_visualizer import visualize_splits


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
        '--split-train',
        action='store_true',
        help='Split training set into Expert (90%%) and Gating (10%%) with same imbalance'
    )
    
    parser.add_argument(
        '--expert-ratio',
        type=float,
        default=0.9,
        help='Ratio for expert training when splitting (default: 0.9 = 90%%)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations of data distributions'
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
    
    if args.split_train:
        print(f"\n  • Train Split: {args.expert_ratio*100:.0f}% Expert / {(1-args.expert_ratio)*100:.0f}% Gating")
        print(f"    ➜ Expert and Gating will have SAME imbalance ratio!")
    else:
        print(f"\n  • Train Split: Single unified training set")
    
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
            seed=args.seed,
            split_train_for_experts_and_gating=args.split_train,
            expert_ratio=args.expert_ratio
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        
        print("\n📊 Datasets Created:")
        for name, dataset in datasets.items():
            print(f"  {name:>8}: {len(dataset):>6,} samples")
        
        print("\n📁 Files Saved to: " + args.output)
        output_path = Path(args.output)
        if output_path.exists():
            for json_file in sorted(output_path.glob("*.json")):
                print(f"  ✓ {json_file.name}")
        
        # Visualize if requested
        if args.visualize:
            print("\n" + "="*80)
            print("CREATING VISUALIZATIONS")
            print("="*80)
            visualize_splits(args.output, "outputs/visualizations")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if args.split_train:
            print("\n1️⃣  Train Expert Models (using expert split):")
            print("   python train_experts.py --use-expert-split")
            print("\n2️⃣  Train Gating Network (using gating split):")
            print("   python train_gating.py --mode pretrain --use-gating-split")
            print("   python train_gating.py --mode selective --use-gating-split")
        else:
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
        
        if not args.visualize:
            print("\n💡 TIP: Run with --visualize to see data distributions")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nPlease check:")
        print("  • CIFAR-100 data is accessible")
        print("  • Required dependencies are installed (numpy, torchvision, matplotlib)")
        print("  • Sufficient disk space")
        print("  • Write permissions for output directory")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
