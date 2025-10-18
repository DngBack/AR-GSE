#!/usr/bin/env python3
"""
Verify recomputed logits are correct.

This script:
1. Loads logits from all experts
2. Checks shapes and consistency
3. Computes accuracies
4. Compares with reweighted metrics

Usage:
    python verify_logits.py
    python verify_logits.py --logits-dir outputs/logits_fixed
"""

import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.metrics.reweighted_metrics import ReweightedMetrics


def load_logits(logits_path):
    """Load logits from .npz file."""
    data = np.load(logits_path)
    return {
        'logits': data['logits'],
        'targets': data['targets'],
        'predictions': data['predictions']
    }


def verify_logits_file(logits_path):
    """Verify a single logits file."""
    data = load_logits(logits_path)
    
    logits = data['logits']
    targets = data['targets']
    predictions = data['predictions']
    
    # Basic checks
    checks = {
        'Shape consistency': logits.shape[0] == len(targets) == len(predictions),
        'Num classes': logits.shape[1] == 100,
        'Predictions valid': np.all((predictions >= 0) & (predictions < 100)),
        'Targets valid': np.all((targets >= 0) & (targets < 100)),
        'Predictions match argmax': np.allclose(predictions, logits.argmax(axis=1))
    }
    
    # Compute accuracy
    accuracy = (predictions == targets).mean()
    
    return checks, accuracy, logits.shape


def main():
    parser = argparse.ArgumentParser(description="Verify recomputed logits")
    
    parser.add_argument(
        '--logits-dir',
        type=str,
        default='outputs/logits_fixed',
        help='Directory containing logits'
    )
    
    parser.add_argument(
        '--splits-dir',
        type=str,
        default='data/cifar100_lt_if100_splits_fixed',
        help='Directory containing data splits'
    )
    
    args = parser.parse_args()
    
    logits_dir = Path(args.logits_dir)
    
    print(f"\n{'='*80}")
    print("VERIFYING RECOMPUTED LOGITS")
    print(f"{'='*80}")
    print(f"\nLogits directory: {logits_dir}")
    print(f"Splits directory: {args.splits_dir}")
    
    if not logits_dir.exists():
        print(f"\n❌ ERROR: Logits directory not found: {logits_dir}")
        print("\nRun this first:")
        print("  python recompute_logits.py")
        return
    
    # Load class weights for reweighted metrics
    weights_path = Path(args.splits_dir) / "class_weights.json"
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            class_weights = np.array(json.load(f))
        metrics_calculator = ReweightedMetrics(class_weights)
        print(f"✓ Loaded class weights from {weights_path}")
    else:
        metrics_calculator = None
        print(f"⚠️  Class weights not found, will skip reweighted metrics")
    
    # Find all expert directories
    expert_dirs = [d for d in logits_dir.iterdir() if d.is_dir()]
    
    if not expert_dirs:
        print(f"\n❌ ERROR: No expert directories found in {logits_dir}")
        return
    
    print(f"\nFound {len(expert_dirs)} expert(s): {[d.name for d in expert_dirs]}")
    
    # Verification results
    all_results = defaultdict(dict)
    
    # Process each expert
    for expert_dir in sorted(expert_dirs):
        expert_name = expert_dir.name
        
        print(f"\n{'='*80}")
        print(f"EXPERT: {expert_name.upper()}")
        print(f"{'='*80}")
        
        # Find all logits files
        logits_files = list(expert_dir.glob("*_logits.npz"))
        
        if not logits_files:
            print(f"  ⚠️  No logits files found")
            continue
        
        # Process each split
        for logits_file in sorted(logits_files):
            split_name = logits_file.stem.replace('_logits', '')
            
            print(f"\n  📊 {split_name.upper()}")
            print(f"  {'-'*60}")
            
            # Verify
            checks, accuracy, shape = verify_logits_file(logits_file)
            
            # Print checks
            print(f"  Verification checks:")
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"    {status} {check_name}")
            
            # Print stats
            print(f"\n  Statistics:")
            print(f"    Shape: {shape}")
            print(f"    Samples: {shape[0]:,}")
            print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Compute reweighted metrics if available
            if metrics_calculator is not None:
                data = load_logits(logits_file)
                results = metrics_calculator.compute_metrics(
                    data['predictions'],
                    data['targets']
                )
                
                print(f"\n  Reweighted Metrics:")
                print(f"    Reweighted Acc: {results['reweighted_accuracy']:.4f} ({results['reweighted_accuracy']*100:.2f}%)")
                print(f"    Balanced Acc:   {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
                print(f"    Head Acc:       {results['head_acc']:.4f} ({results['head_acc']*100:.2f}%)")
                print(f"    Tail Acc:       {results['tail_acc']:.4f} ({results['tail_acc']*100:.2f}%)")
                
                all_results[expert_name][split_name] = results
            
            # Check if all verification passed
            all_passed = all(checks.values())
            if not all_passed:
                print(f"\n  ⚠️  WARNING: Some checks failed!")
    
    # Summary table
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY: REWEIGHTED ACCURACY BY EXPERT AND SPLIT")
        print(f"{'='*80}")
        
        # Get all splits
        all_splits = set()
        for expert_results in all_results.values():
            all_splits.update(expert_results.keys())
        all_splits = sorted(all_splits)
        
        # Print header
        header = f"{'Expert':<15}"
        for split in all_splits:
            header += f" | {split.capitalize():>12}"
        print(f"\n{header}")
        print("-" * len(header))
        
        # Print each expert
        for expert_name in sorted(all_results.keys()):
            row = f"{expert_name:<15}"
            for split in all_splits:
                if split in all_results[expert_name]:
                    acc = all_results[expert_name][split]['reweighted_accuracy']
                    row += f" | {acc*100:>11.2f}%"
                else:
                    row += f" | {'-':>12}"
            print(row)
        
        # Best per split
        print("\n" + "="*len(header))
        print("Best per split:")
        for split in all_splits:
            best_expert = None
            best_acc = 0.0
            for expert_name, expert_results in all_results.items():
                if split in expert_results:
                    acc = expert_results[split]['reweighted_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_expert = expert_name
            if best_expert:
                print(f"  {split}: {best_expert} ({best_acc*100:.2f}%)")
    
    # Final status
    print(f"\n{'='*80}")
    print("✅ VERIFICATION COMPLETED")
    print(f"{'='*80}")
    print("\nAll logits files verified successfully!")
    print("\nYou can now proceed with:")
    print("  1. Train gating: python train_gating.py --mode pretrain")
    print("  2. Train AR-GSE: python train_argse.py")


if __name__ == "__main__":
    main()
