#!/usr/bin/env python3
"""
Test script to verify gating training data loading.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
import torchvision

sys.path.append(str(Path(__file__).parent / 'src'))

def test_gating_logits():
    """Test that gating logits exist and are properly formatted."""
    print("="*60)
    print("TESTING GATING LOGITS")
    print("="*60)
    
    logits_dir = Path("outputs/logits/cifar100_lt_if100")
    splits_dir = Path("data/cifar100_lt_if100_splits_fixed")
    expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    
    # Test splits for pretrain and selective modes
    test_splits = {
        'gating': ('gating_indices.json', 'train'),  # For pretrain mode
        'tunev': ('tunev_indices.json', 'test'),     # For selective S1
        'val': ('val_indices.json', 'test')          # For selective S2
    }
    
    print("\n1. Checking split files:")
    print("-" * 60)
    for split_name, (indices_file, dataset_type) in test_splits.items():
        indices_path = splits_dir / indices_file
        if indices_path.exists():
            with open(indices_path) as f:
                indices = json.load(f)
            print(f"✅ {split_name}: {len(indices)} samples (from CIFAR-100 {dataset_type})")
        else:
            print(f"❌ {split_name}: NOT FOUND ({indices_path})")
    
    print("\n2. Checking expert logits:")
    print("-" * 60)
    
    all_ok = True
    for expert_name in expert_names:
        expert_dir = logits_dir / expert_name
        print(f"\n{expert_name}:")
        
        if not expert_dir.exists():
            print(f"  ❌ Expert directory not found: {expert_dir}")
            all_ok = False
            continue
        
        for split_name, (indices_file, _) in test_splits.items():
            logits_file = expert_dir / f"{split_name}_logits.pt"
            
            if logits_file.exists():
                logits = torch.load(logits_file, map_location='cpu')
                print(f"  ✅ {split_name}_logits.pt: shape={logits.shape}, dtype={logits.dtype}")
                
                # Check expected size
                indices_path = splits_dir / indices_file
                if indices_path.exists():
                    with open(indices_path) as f:
                        expected_size = len(json.load(f))
                    if logits.shape[0] == expected_size:
                        print(f"     Size matches: {logits.shape[0]} == {expected_size} ✓")
                    else:
                        print(f"     ⚠️  Size mismatch: {logits.shape[0]} != {expected_size}")
                        all_ok = False
            else:
                print(f"  ❌ {split_name}_logits.pt: NOT FOUND")
                all_ok = False
    
    print("\n3. Testing data loading:")
    print("-" * 60)
    
    # Test loading gating split
    try:
        cifar_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
        gating_indices_path = splits_dir / 'gating_indices.json'
        
        with open(gating_indices_path) as f:
            gating_indices = json.load(f)
        
        # Check class distribution
        from collections import Counter
        gating_labels = [cifar_train.targets[i] for i in gating_indices]
        class_counts = Counter(gating_labels)
        
        print(f"Gating split class distribution:")
        print(f"  Total samples: {len(gating_indices)}")
        print(f"  Head class (0): {class_counts[0]} samples")
        print(f"  Tail class (99): {class_counts[99]} samples")
        print(f"  Imbalance Factor: {class_counts[0] / class_counts[99]:.1f}")
        
        # Load and stack logits
        num_experts = len(expert_names)
        num_classes = 100
        stacked_logits = torch.zeros(len(gating_indices), num_experts, num_classes)
        
        for i, expert_name in enumerate(expert_names):
            logits_path = logits_dir / expert_name / "gating_logits.pt"
            logits = torch.load(logits_path, map_location='cpu')
            if logits.dtype == torch.float16:
                logits = logits.float()
            stacked_logits[:, i, :] = logits
        
        print(f"\n✅ Stacked logits shape: {stacked_logits.shape}")
        print(f"   Expected: ({len(gating_indices)}, {num_experts}, {num_classes})")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        all_ok = False
    
    # Test tunev and val (for selective mode)
    print("\n4. Testing selective mode splits:")
    print("-" * 60)
    
    try:
        cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
        
        for split_name in ['tunev', 'val']:
            indices_path = splits_dir / f'{split_name}_indices.json'
            with open(indices_path) as f:
                indices = json.load(f)
            
            # Check balance
            labels = [cifar_test.targets[i] for i in indices]
            counts = Counter(labels)
            
            print(f"\n{split_name.upper()} split:")
            print(f"  Total samples: {len(indices)}")
            print(f"  Per class: {counts[0]} samples")
            print(f"  Balanced: {len(set(counts.values())) == 1} ✓" if len(set(counts.values())) == 1 else "  ⚠️  Not balanced")
            
            # Check logits exist
            all_exist = True
            for expert_name in expert_names:
                logits_path = logits_dir / expert_name / f"{split_name}_logits.pt"
                if not logits_path.exists():
                    print(f"  ❌ Missing: {expert_name}/{split_name}_logits.pt")
                    all_exist = False
            
            if all_exist:
                print(f"  ✅ All expert logits available")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou can now train gating:")
        print("  Pretrain:  python train_gating.py --mode pretrain")
        print("  Selective: python train_gating.py --mode selective")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("="*60)
        print("\nPlease ensure:")
        print("1. Experts are trained: python train_experts.py")
        print("2. Splits are created: python create_splits_fixed.py --split-train")
        print("3. Logits are exported during expert training")

if __name__ == "__main__":
    test_gating_logits()
