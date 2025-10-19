"""
Test script to verify evaluation setup is correct before running AURC evaluation.

Checks:
1. Plugin checkpoint exists and contains required keys
2. All expert logits are available for tunev, val, test splits
3. Class weights file exists for reweighting
4. Splits indices are available
5. Data integrity (no overlap between splits)
"""

import json
from pathlib import Path
import torch
import numpy as np

# Configuration matching eval_gse_plugin.py
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
}

def test_plugin_checkpoint():
    """Test if plugin checkpoint exists and has required keys."""
    print("\n1️⃣ Testing Plugin Checkpoint...")
    ckpt_path = Path(CONFIG['plugin_checkpoint'])
    
    if not ckpt_path.exists():
        print(f"   ❌ Plugin checkpoint not found: {ckpt_path}")
        return False
    
    print(f"   ✅ Checkpoint exists: {ckpt_path}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        required_keys = ['alpha', 'mu', 'class_to_group', 'num_groups']
        optional_keys = ['gating_net_state_dict', 'best_score']
        
        for key in required_keys:
            if key not in checkpoint:
                print(f"   ❌ Missing required key: {key}")
                return False
            print(f"   ✅ Found key: {key}")
        
        for key in optional_keys:
            if key in checkpoint:
                print(f"   ✅ Found optional key: {key}")
        
        # Show dimensions
        print(f"\n   Checkpoint details:")
        print(f"   • α shape: {checkpoint['alpha'].shape}")
        print(f"   • μ shape: {checkpoint['mu'].shape}")
        print(f"   • class_to_group shape: {checkpoint['class_to_group'].shape}")
        print(f"   • num_groups: {checkpoint['num_groups']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        return False

def test_expert_logits():
    """Test if all expert logits are available for evaluation splits."""
    print("\n2️⃣ Testing Expert Logits Availability...")
    logits_root = Path(CONFIG['experts']['logits_dir'])
    
    if not logits_root.exists():
        print(f"   ❌ Logits directory not found: {logits_root}")
        return False
    
    print(f"   ✅ Logits directory exists: {logits_root}")
    
    splits = ['tunev', 'val', 'test']
    all_found = True
    
    for expert_name in CONFIG['experts']['names']:
        expert_dir = logits_root / expert_name
        if not expert_dir.exists():
            print(f"   ❌ Expert directory not found: {expert_dir}")
            all_found = False
            continue
        
        print(f"\n   Expert: {expert_name}")
        for split in splits:
            npz_path = expert_dir / f"{split}_logits.npz"
            pt_path = expert_dir / f"{split}_logits.pt"
            
            if npz_path.exists():
                print(f"   ✅ {split}_logits.npz")
                # Verify can load
                try:
                    data = np.load(npz_path)
                    print(f"      → Shape: {data['logits'].shape}")
                except Exception as e:
                    print(f"      ❌ Error loading: {e}")
                    all_found = False
            elif pt_path.exists():
                print(f"   ✅ {split}_logits.pt")
                try:
                    logits = torch.load(pt_path, map_location='cpu', weights_only=False)
                    print(f"      → Shape: {logits.shape}")
                except Exception as e:
                    print(f"      ❌ Error loading: {e}")
                    all_found = False
            else:
                print(f"   ❌ Missing: {split}_logits (neither .npz nor .pt)")
                all_found = False
    
    return all_found

def test_class_weights():
    """Test if class weights file exists for reweighting."""
    print("\n3️⃣ Testing Class Weights...")
    weights_path = Path(CONFIG['dataset']['splits_dir']) / 'class_weights.json'
    
    if not weights_path.exists():
        print(f"   ⚠️  Class weights not found: {weights_path}")
        print("   → Evaluation will use uniform weighting")
        return True  # Not critical, just a warning
    
    print(f"   ✅ Class weights file exists: {weights_path}")
    
    try:
        with open(weights_path, 'r') as f:
            class_weights = json.load(f)
        
        if isinstance(class_weights, list):
            print(f"   ✅ Loaded {len(class_weights)} class weights (list format)")
            if len(class_weights) == CONFIG['dataset']['num_classes']:
                print(f"   ✅ Correct number of classes: {CONFIG['dataset']['num_classes']}")
            else:
                print(f"   ❌ Wrong number of classes: {len(class_weights)} vs {CONFIG['dataset']['num_classes']}")
                return False
        elif isinstance(class_weights, dict):
            print(f"   ✅ Loaded {len(class_weights)} class weights (dict format)")
        else:
            print(f"   ❌ Unknown format: {type(class_weights)}")
            return False
        
        # Show sample weights
        if isinstance(class_weights, list):
            print(f"\n   Sample weights:")
            print(f"   • Class 0 (head): {class_weights[0]:.6f}")
            print(f"   • Class 49 (mid): {class_weights[49]:.6f}")
            print(f"   • Class 99 (tail): {class_weights[99]:.6f}")
            print(f"   • Ratio (head/tail): {class_weights[0]/class_weights[99]:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading class weights: {e}")
        return False

def test_splits_indices():
    """Test if all split indices files exist."""
    print("\n4️⃣ Testing Split Indices...")
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    if not splits_dir.exists():
        print(f"   ❌ Splits directory not found: {splits_dir}")
        return False
    
    print(f"   ✅ Splits directory exists: {splits_dir}")
    
    required_files = ['tunev_indices.json', 'val_indices.json', 'test_indices.json']
    all_found = True
    indices_data = {}
    
    for filename in required_files:
        filepath = splits_dir / filename
        if not filepath.exists():
            print(f"   ❌ Missing: {filename}")
            all_found = False
        else:
            print(f"   ✅ Found: {filename}")
            try:
                with open(filepath, 'r') as f:
                    indices = json.load(f)
                print(f"      → {len(indices)} samples")
                indices_data[filename] = set(indices)
            except Exception as e:
                print(f"      ❌ Error loading: {e}")
                all_found = False
    
    # Check for overlap
    if len(indices_data) == 3:
        print(f"\n   Checking for overlap between splits...")
        tunev_set = indices_data['tunev_indices.json']
        val_set = indices_data['val_indices.json']
        test_set = indices_data['test_indices.json']
        
        overlap_tunev_val = tunev_set & val_set
        overlap_tunev_test = tunev_set & test_set
        overlap_val_test = val_set & test_set
        
        if overlap_tunev_val:
            print(f"   ❌ Overlap between tunev and val: {len(overlap_tunev_val)} samples")
            all_found = False
        else:
            print(f"   ✅ No overlap between tunev and val")
        
        if overlap_tunev_test:
            print(f"   ❌ Overlap between tunev and test: {len(overlap_tunev_test)} samples")
            all_found = False
        else:
            print(f"   ✅ No overlap between tunev and test")
        
        if overlap_val_test:
            print(f"   ❌ Overlap between val and test: {len(overlap_val_test)} samples")
            all_found = False
        else:
            print(f"   ✅ No overlap between val and test")
        
        # Total coverage
        total = len(tunev_set | val_set | test_set)
        print(f"\n   Total unique samples across all eval splits: {total}")
        print(f"   Expected: 8000 (from CIFAR-100 test set)")
        if total <= 8000:
            print(f"   ✅ Total is within bounds")
        else:
            print(f"   ❌ Total exceeds test set size!")
            all_found = False
    
    return all_found

def test_data_integrity():
    """Verify data can be loaded and has correct dimensions."""
    print("\n5️⃣ Testing Data Integrity...")
    
    try:
        import torchvision
        cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
        print(f"   ✅ CIFAR-100 test set loaded: {len(cifar_test)} samples")
        
        # Load a sample split and verify indices
        splits_dir = Path(CONFIG['dataset']['splits_dir'])
        with open(splits_dir / 'test_indices.json', 'r') as f:
            test_indices = json.load(f)
        
        # Check indices are valid
        max_idx = max(test_indices)
        min_idx = min(test_indices)
        
        if min_idx < 0:
            print(f"   ❌ Invalid negative index: {min_idx}")
            return False
        
        if max_idx >= len(cifar_test):
            print(f"   ❌ Index out of bounds: {max_idx} >= {len(cifar_test)}")
            return False
        
        print(f"   ✅ Indices are valid: [{min_idx}, {max_idx}]")
        
        # Try loading a sample logit file
        logits_root = Path(CONFIG['experts']['logits_dir'])
        expert_name = CONFIG['experts']['names'][0]
        sample_logit_file = logits_root / expert_name / 'test_logits.npz'
        
        if sample_logit_file.exists():
            data = np.load(sample_logit_file)
            logits = data['logits']
            print(f"   ✅ Sample logits loaded: {logits.shape}")
            
            if logits.shape[0] != len(test_indices):
                print(f"   ❌ Logits size mismatch: {logits.shape[0]} vs {len(test_indices)}")
                return False
            
            if logits.shape[1] != CONFIG['dataset']['num_classes']:
                print(f"   ❌ Wrong number of classes: {logits.shape[1]} vs {CONFIG['dataset']['num_classes']}")
                return False
            
            print(f"   ✅ Logits dimensions match expectations")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("EVALUATION SETUP VERIFICATION")
    print("="*70)
    print(f"\nDataset: {CONFIG['dataset']['name']}")
    print(f"Experts: {', '.join(CONFIG['experts']['names'])}")
    print(f"Splits: tunev, val, test (all balanced, from test set)")
    print(f"Reweighting: Using class_weights.json from training distribution")
    
    results = {
        'Plugin Checkpoint': test_plugin_checkpoint(),
        'Expert Logits': test_expert_logits(),
        'Class Weights': test_class_weights(),
        'Split Indices': test_splits_indices(),
        'Data Integrity': test_data_integrity(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:>10} | {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready for AURC evaluation.")
        print("\nRun evaluation with:")
        print("  python -m src.train.eval_gse_plugin")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before evaluation.")
        return 1

if __name__ == '__main__':
    exit(main())
