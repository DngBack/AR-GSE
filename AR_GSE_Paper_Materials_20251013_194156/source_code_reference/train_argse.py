#!/usr/bin/env python3
"""
AR-GSE Training Script
Trains the complete AR-GSE ensemble model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AR-GSE ensemble model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without training'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint if available'
    )
    
    return parser.parse_args()

def setup_argse_environment(args):
    """Setup AR-GSE training environment."""
    try:
        import torch
        
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        print("🚀 AR-GSE Ensemble Training Pipeline")
        print(f"Device: {device}")
        
        if args.verbose:
            print(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name()}")
        
        return device
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        sys.exit(1)

def check_prerequisites():
    """Check if all prerequisites are met for AR-GSE training."""
    print("🔍 Checking prerequisites...")
    
    # Check if expert models exist
    expert_dir = Path("checkpoints/experts/cifar100_lt_if100")
    if not expert_dir.exists():
        raise Exception("Expert models directory not found. Please train experts first.")
    
    expert_models = ['ce_model.pth', 'logitadjust_model.pth', 'balsoftmax_model.pth']
    missing_experts = []
    
    for model in expert_models:
        if not (expert_dir / model).exists():
            missing_experts.append(model)
    
    if missing_experts:
        raise Exception(f"Missing expert models: {missing_experts}")
    
    # Check if gating models exist
    gating_dir = Path("checkpoints/gating_pretrained/cifar100_lt_if100")
    if not gating_dir.exists():
        raise Exception("Gating models directory not found. Please train gating models first.")
    
    # Check data splits
    splits_dir = Path("data/cifar100_lt_if100_splits")
    if not splits_dir.exists():
        raise Exception("Data splits not found. Please prepare data splits first.")
    
    required_splits = ['train_indices.json', 'val_lt_indices.json', 'test_lt_indices.json', 'tuneV_indices.json']
    missing_splits = []
    
    for split in required_splits:
        if not (splits_dir / split).exists():
            missing_splits.append(split)
    
    if missing_splits:
        raise Exception(f"Missing data splits: {missing_splits}")
    
    print("✅ All prerequisites met")

def run_argse_training(args):
    """Run the AR-GSE training."""
    if args.dry_run:
        print("🔍 [DRY RUN] Would run AR-GSE ensemble training")
        return
    
    try:
        print("🎯 Running AR-GSE ensemble training")
        
        # Execute the training script
        import subprocess
        
        command = ["python", "run_improved_eg_outer.py"]
        
        if args.verbose:
            print(f"Executing: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=False,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("✅ Successfully completed AR-GSE training")
        else:
            raise Exception(f"Training failed with exit code: {result.returncode}")
            
    except Exception as e:
        raise Exception(f"Failed to run AR-GSE training: {str(e)}")

def main():
    """Main function for AR-GSE training."""
    args = parse_arguments()
    
    print("=" * 60)
    print("AR-GSE ENSEMBLE TRAINING")
    print("=" * 60)
    
    # Setup environment
    setup_argse_environment(args)
    
    try:
        # Check prerequisites
        check_prerequisites()
        
        print(f"\n{'='*40}")
        print("🎯 AR-GSE Ensemble Training")
        print(f"{'='*40}")
        
        print("\n📋 Training Process:")
        print("  - Load pretrained expert models")
        print("  - Load pretrained gating models")
        print("  - Train ensemble with gating mechanism")
        print("  - Apply rejection learning on long-tail data")
        
        # Run the training
        run_argse_training(args)
        
        print(f"\n{'='*60}")
        print("🏁 AR-GSE TRAINING SUMMARY")
        print(f"{'='*60}")
        print("✅ AR-GSE ensemble training completed successfully!")
        
        print("\n➡️  Next step: Evaluate AR-GSE model")
        print("   Command: python evaluate_argse.py")
        
        # Show output locations
        print("\n📁 Output locations:")
        print("   Checkpoints: checkpoints/argse_worst_eg_improved_v3_3/cifar100_lt_if100/")
        print("   Results: results_worst_eg_improved/cifar100_lt_if100/")
            
    except Exception as e:
        print(f"\n❌ AR-GSE training failed: {e}")
        
        print("\nTroubleshooting tips:")
        print("1. Ensure all expert models are trained")
        print("2. Ensure gating models are trained (both pretrain and selective)")
        print("3. Ensure data splits are prepared")
        print("4. Check available GPU memory if using CUDA")
        
        if args.verbose:
            import traceback
            print("\nFull error traceback:")
            traceback.print_exc()
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()