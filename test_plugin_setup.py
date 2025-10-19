#!/usr/bin/env python3
"""
Test script to verify plugin data setup.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

def test_plugin_setup():
    """Test plugin data loading and class weights."""
    print("="*60)
    print("TESTING PLUGIN SETUP")
    print("="*60)
    
    splits_dir = Path("data/cifar100_lt_if100_splits_fixed")
    logits_dir = Path("outputs/logits/cifar100_lt_if100")
    
    expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    
    # 1. Check class weights
    print("\n1. Checking class weights:")
    print("-" * 60)
    weights_path = splits_dir / 'class_weights.json'
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)
        
        if isinstance(weights, list):
            weights_tensor = torch.tensor(weights)
        else:
            weights_tensor = torch.tensor([weights[str(i)] for i in range(100)])
        
        print(f"✅ Class weights loaded: {len(weights_tensor)} classes")
        print(f"   Min weight: {weights_tensor.min():.6f}")
        print(f"   Max weight: {weights_tensor.max():.6f}")
        print(f"   Sum: {weights_tensor.sum():.6f}")
        print(f"   Head (class 0): {weights_tensor[0]:.6f}")
        print(f"   Tail (class 99): {weights_tensor[99]:.6f}")
        print(f"   Ratio (head/tail): {weights_tensor[0] / weights_tensor[99]:.1f}x")
    else:
        print(f"❌ Class weights not found: {weights_path}")
        return False
    
    # 2. Check splits for plugin (tunev and val)
    print("\n2. Checking plugin splits:")
    print("-" * 60)
    
    plugin_splits = ['tunev', 'val']
    all_ok = True
    
    for split_name in plugin_splits:
        indices_path = splits_dir / f"{split_name}_indices.json"
        
        if indices_path.exists():
            with open(indices_path) as f:
                indices = json.load(f)
            print(f"\n{split_name.upper()}:")
            print(f"  ✅ Indices file: {len(indices)} samples")
            
            # Check logits exist for all experts
            logits_ok = True
            for expert_name in expert_names:
                logits_path = logits_dir / expert_name / f"{split_name}_logits.pt"
                if logits_path.exists():
                    logits = torch.load(logits_path, map_location='cpu')
                    print(f"  ✅ {expert_name}: {logits.shape}, dtype={logits.dtype}")
                    
                    if logits.shape[0] != len(indices):
                        print(f"     ⚠️  Size mismatch: {logits.shape[0]} != {len(indices)}")
                        logits_ok = False
                else:
                    print(f"  ❌ {expert_name}: NOT FOUND")
                    logits_ok = False
            
            if not logits_ok:
                all_ok = False
        else:
            print(f"❌ {split_name}: indices file not found")
            all_ok = False
    
    # 3. Check gating checkpoints
    print("\n3. Checking gating checkpoints:")
    print("-" * 60)
    
    gating_dir = Path("checkpoints/gating_pretrained/cifar100_lt_if100")
    
    checkpoints = {
        'pretrain': 'gating_pretrained.ckpt',
        'selective': 'gating_selective.ckpt'
    }
    
    for mode, ckpt_file in checkpoints.items():
        ckpt_path = gating_dir / ckpt_file
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            print(f"✅ {mode}: {ckpt_file}")
            if 'alpha' in ckpt:
                print(f"   α: {ckpt['alpha'].tolist()}")
            if 'mu' in ckpt:
                print(f"   μ: {ckpt['mu'].tolist()}")
            if 't_param' in ckpt:
                print(f"   t: {ckpt['t_param'].tolist()}")
        else:
            print(f"⚠️  {mode}: NOT FOUND")
    
    # 4. Test loading with reweighting
    print("\n4. Testing reweighted error computation:")
    print("-" * 60)
    
    try:
        # Simulate some predictions
        num_samples = 1000
        num_classes = 100
        
        # Simulate balanced data (10 samples per class)
        y_true = torch.arange(num_classes).repeat_interleave(10)
        y_pred = y_true.clone()
        
        # Make some errors (20% error rate)
        error_mask = torch.rand(len(y_true)) < 0.2
        y_pred[error_mask] = torch.randint(0, num_classes, (error_mask.sum(),))
        
        # Compute standard accuracy
        standard_acc = (y_pred == y_true).float().mean()
        
        # Compute weighted accuracy
        class_acc = torch.zeros(num_classes)
        for c in range(num_classes):
            mask = (y_true == c)
            if mask.sum() > 0:
                class_acc[c] = (y_pred[mask] == c).float().mean()
        
        weighted_acc = (class_acc * weights_tensor).sum()
        
        print(f"Standard accuracy: {standard_acc:.4f}")
        print(f"Weighted accuracy: {weighted_acc:.4f}")
        print(f"✅ Reweighting computation works!")
        
    except Exception as e:
        print(f"❌ Error testing reweighting: {e}")
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou can now run plugin training:")
        print("  python run_improved_eg_outer.py")
        print("or:")
        print("  python -m src.train.gse_balanced_plugin")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("="*60)
        print("\nPlease ensure:")
        print("1. Splits are created: python create_splits_fixed.py --split-train")
        print("2. Experts are trained: python train_experts.py")
        print("3. Gating is trained: python train_gating.py --mode selective")
    
    return all_ok

if __name__ == "__main__":
    test_plugin_setup()
