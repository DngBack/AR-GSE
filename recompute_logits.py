#!/usr/bin/env python3
"""
Recompute logits for val/test/tunev splits using pre-trained expert models.

This script is used when you have new data splits but want to reuse
pre-trained expert models (because training data hasn't changed).

Usage:
    python recompute_logits.py
    python recompute_logits.py --experts-dir checkpoints/experts --output-dir outputs/logits_new
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.balanced_test_splits import CIFAR100LTDataset


def get_model():
    """
    Get the model architecture (should match your expert training).
    Uses the Expert class from src.models.experts.
    """
    try:
        from src.models.experts import Expert
        # Create expert model with CIFAR ResNet-32 backbone
        model = Expert(num_classes=100, backbone_name='cifar_resnet32')
        return model
    except Exception as e:
        print(f"  ⚠️  Error creating Expert model: {e}")
        print(f"  Trying alternative approach...")
        raise


def load_expert_model(checkpoint_path, device):
    """
    Load a pre-trained expert model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        
    Returns:
        Loaded model in eval mode
    """
    print(f"  Loading: {checkpoint_path}")
    
    model = get_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # Checkpoint might be the state dict itself
            model.load_state_dict(checkpoint, strict=False)
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ⚠️  Error loading checkpoint: {e}")
        print(f"  Trying to load with strict=False...")
        
        # Try to match keys by removing/adding prefixes
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Create new state dict with matched keys
        model_dict = model.state_dict()
        matched_dict = {}
        
        for k, v in state_dict.items():
            # Remove 'module.' prefix if exists
            new_k = k.replace('module.', '')
            
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                matched_dict[new_k] = v
            else:
                # Try to find matching key
                for model_k in model_dict.keys():
                    if model_k.endswith(new_k.split('.')[-1]) and model_dict[model_k].shape == v.shape:
                        matched_dict[model_k] = v
                        break
        
        print(f"  Matched {len(matched_dict)}/{len(model_dict)} parameters")
        model.load_state_dict(matched_dict, strict=False)
        print(f"  ✓ Model loaded with partial matching")
    
    model = model.to(device)
    model.eval()
    
    return model


def compute_logits_for_split(model, dataloader, device, desc="Computing logits"):
    """
    Compute logits for a dataset split.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the split
        device: torch device
        desc: Description for progress bar
        
    Returns:
        Tuple of (logits, targets, predictions)
    """
    all_logits = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store results
            all_logits.append(outputs.cpu().numpy())
            all_targets.append(labels.numpy())
            all_predictions.append(outputs.argmax(dim=1).cpu().numpy())
    
    # Concatenate all batches
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    
    return logits, targets, predictions


def save_logits(logits_dict, output_path):
    """
    Save logits to file.
    
    Args:
        logits_dict: Dict with keys 'logits', 'targets', 'predictions'
        output_path: Path to save file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .npz (efficient for numpy arrays)
    np.savez_compressed(
        output_path,
        logits=logits_dict['logits'],
        targets=logits_dict['targets'],
        predictions=logits_dict['predictions']
    )
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute logits for new data splits using pre-trained experts"
    )
    
    parser.add_argument(
        '--experts-dir',
        type=str,
        default='checkpoints/experts',
        help='Directory containing expert checkpoints'
    )
    
    parser.add_argument(
        '--splits-dir',
        type=str,
        default='data/cifar100_lt_if100_splits_fixed',
        help='Directory containing data splits'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/logits_fixed',
        help='Output directory for logits'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--experts',
        nargs='+',
        default=['ce_baseline', 'balsoftmax_baseline', 'logitadjust_baseline', 'decoupling_twostage'],
        help='List of experts to compute logits for'
    )
    
    parser.add_argument(
        '--checkpoint-prefix',
        type=str,
        default='best_',
        choices=['best_', 'final_calibrated_', 'stage1_'],
        help='Checkpoint file prefix (best_, final_calibrated_, stage1_)'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['val', 'test', 'tunev'],
        help='List of splits to compute logits for'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    if args.train_experts:
        print("TRAINING EXPERTS & RECOMPUTING LOGITS")
    else:
        print("RECOMPUTING LOGITS FOR NEW DATA SPLITS")
    print(f"{'='*80}")
    print(f"\nDevice: {device}")
    print(f"Experts directory: {args.experts_dir}")
    print(f"Splits directory: {args.splits_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nExperts: {args.experts}")
    print(f"Checkpoint prefix: {args.checkpoint_prefix}")
    print(f"Splits: {args.splits}")
    
    if args.train_experts:
        print(f"\n🔥 TRAINING MODE ENABLED 🔥")
        print(f"   Epochs: {args.train_epochs}")
        print(f"   Learning rate: {args.train_lr}")
        print(f"   Experts will be trained before computing logits")
    
    # Setup checkpoint paths
    experts_base_dir = Path(args.experts_dir)
    
    # Check if checkpoints are in subdirectory
    cifar100_subdir = experts_base_dir / "cifar100_lt_if100"
    if cifar100_subdir.exists():
        print(f"\n✓ Found cifar100_lt_if100 subdirectory")
        checkpoint_dir = cifar100_subdir
    else:
        checkpoint_dir = experts_base_dir
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Load CIFAR-100 test set
    print(f"\n{'='*80}")
    print("STEP 1: Loading CIFAR-100 test dataset")
    print(f"{'='*80}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    cifar_test = torchvision.datasets.CIFAR100(
        root='data', train=False, download=False, transform=None
    )
    print(f"✓ Loaded CIFAR-100 test: {len(cifar_test)} samples")
    
    # Load data splits
    print(f"\n{'='*80}")
    print("STEP 2: Loading data splits")
    print(f"{'='*80}")
    
    dataloaders = {}
    for split_name in args.splits:
        # Load indices
        indices_path = Path(args.splits_dir) / f"{split_name}_indices.json"
        print(f"\nLoading {split_name} split from {indices_path}")
        
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        
        # Create dataset
        dataset = CIFAR100LTDataset(cifar_test, indices, transform)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        dataloaders[split_name] = dataloader
        print(f"  ✓ {split_name}: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Process each expert
    print(f"\n{'='*80}")
    print("STEP 3: Computing logits for each expert")
    print(f"{'='*80}")
    
    for expert_name in args.experts:
        print(f"\n{'='*60}")
        print(f"EXPERT: {expert_name.upper()}")
        print(f"{'='*60}")
        
        # Try different checkpoint naming patterns
        checkpoint_patterns = [
            checkpoint_dir / f"{args.checkpoint_prefix}{expert_name}.pth",
            checkpoint_dir / f"{expert_name}.pth",
            checkpoint_dir / f"expert_{expert_name}.pth",
        ]
        
        checkpoint_path = None
        for pattern in checkpoint_patterns:
            if pattern.exists():
                checkpoint_path = pattern
                break
        
        if checkpoint_path is None:
            print(f"  ⚠️  Checkpoint not found. Tried:")
            for pattern in checkpoint_patterns:
                print(f"     - {pattern}")
            print(f"  Skipping {expert_name}")
            continue
        
        model = load_expert_model(checkpoint_path, device)
        print(f"  ✓ Model loaded successfully")
        
        # Compute logits for each split
        for split_name, dataloader in dataloaders.items():
            print(f"\n  Computing logits for {split_name}...")
            
            logits, targets, predictions = compute_logits_for_split(
                model,
                dataloader,
                device,
                desc=f"  {expert_name}/{split_name}"
            )
            
            # Compute accuracy
            accuracy = (predictions == targets).mean()
            print(f"  ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  ✓ Shape: logits={logits.shape}, targets={targets.shape}")
            
            # Save logits
            output_path = Path(args.output_dir) / expert_name / f"{split_name}_logits.npz"
            save_logits(
                {
                    'logits': logits,
                    'targets': targets,
                    'predictions': predictions
                },
                output_path
            )
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ LOGITS COMPUTATION COMPLETED")
    print(f"{'='*80}")
    
    print(f"\n📁 Output structure:")
    output_path = Path(args.output_dir)
    for expert_name in args.experts:
        expert_dir = output_path / expert_name
        if expert_dir.exists():
            print(f"\n  {expert_name}/")
            for npz_file in sorted(expert_dir.glob("*.npz")):
                # Load and check
                data = np.load(npz_file)
                print(f"    ├── {npz_file.name}")
                print(f"    │   ├── logits: {data['logits'].shape}")
                print(f"    │   ├── targets: {data['targets'].shape}")
                print(f"    │   └── predictions: {data['predictions'].shape}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("\n1️⃣  Verify logits are correct:")
    print("   python verify_logits.py")
    print("\n2️⃣  Train gating network with new data:")
    print("   python train_gating.py --mode pretrain")
    print("\n3️⃣  Continue with AR-GSE training:")
    print("   python train_argse.py")


def verify_logits_quick(output_dir):
    """Quick verification of saved logits."""
    output_path = Path(output_dir)
    
    print(f"\n{'='*60}")
    print("Quick Verification")
    print(f"{'='*60}")
    
    for expert_dir in output_path.iterdir():
        if expert_dir.is_dir():
            print(f"\n{expert_dir.name}:")
            for npz_file in expert_dir.glob("*.npz"):
                data = np.load(npz_file)
                logits = data['logits']
                targets = data['targets']
                preds = data['predictions']
                
                acc = (preds == targets).mean()
                print(f"  {npz_file.stem}: {len(targets)} samples, acc={acc:.4f}")


if __name__ == "__main__":
    main()
