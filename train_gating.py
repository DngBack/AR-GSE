#!/usr/bin/env python3
"""
Gating Training Script for AR-GSE
Trains gating models for the AR-GSE ensemble with different modes.
"""

import sys
import argparse
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train gating models for AR-GSE ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pretrain', 'selective'],
        required=True,
        help='Gating training mode: "pretrain" for warm up, "selective" for expert selection'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
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
    
    return parser.parse_args()

def setup_gating_environment(args):
    """Setup gating training environment."""
    try:
        import torch
        
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        print("🚀 AR-GSE Gating Training Pipeline")
        print(f"Device: {device}")
        print(f"Mode: {args.mode}")
        
        if args.verbose:
            print(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name()}")
        
        return device
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        sys.exit(1)

def run_gating_training(args):
    """Run the gating training with specified mode."""
    if args.dry_run:
        print(f"🔍 [DRY RUN] Would run gating training in {args.mode} mode")
        return
    
    try:
        # Import the training module based on mode
        if args.mode == 'pretrain':
            print("🎯 Running gating model warm-up (pretrain mode)")
            command = "python -m src.train.train_gating_only --mode pretrain"
        elif args.mode == 'selective':
            print("🎯 Running gating model for expert selection (selective mode)")
            command = "python -m src.train.train_gating_only --mode selective"
        
        if args.verbose:
            print(f"Executing: {command}")
        
        # Execute the training
        import subprocess
        result = subprocess.run(
            command.split(),
            capture_output=False,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully completed gating training ({args.mode} mode)")
        else:
            raise Exception(f"Training failed with exit code: {result.returncode}")
            
    except Exception as e:
        raise Exception(f"Failed to run gating training: {str(e)}")

def main():
    """Main function for gating training."""
    args = parse_arguments()
    
    print("=" * 60)
    print("AR-GSE GATING TRAINING")
    print("=" * 60)
    
    # Setup environment
    setup_gating_environment(args)
    
    try:
        print(f"\n{'='*40}")
        print(f"🎯 Gating Training Mode: {args.mode.upper()}")
        print(f"{'='*40}")
        
        if args.mode == 'pretrain':
            print("\n📋 Pretrain Mode:")
            print("  - Warm up gating model")
            print("  - Prepare for expert selection")
        elif args.mode == 'selective':
            print("\n📋 Selective Mode:")
            print("  - Train gating for expert selection")
            print("  - Requires pretrained gating model")
        
        # Run the training
        run_gating_training(args)
        
        print(f"\n{'='*60}")
        print("🏁 GATING TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Gating training ({args.mode}) completed successfully!")
        
        if args.mode == 'pretrain':
            print("\n➡️  Next step: Run selective gating training")
            print("   Command: python train_gating.py --mode selective")
        elif args.mode == 'selective':
            print("\n➡️  Next step: Train AR-GSE ensemble")
            print("   Command: python run_improved_eg_outer.py")
            
    except Exception as e:
        print(f"\n❌ Gating training failed: {e}")
        
        print("\nTroubleshooting tips:")
        print("1. Ensure expert models are trained first")
        print("2. Check that data splits are prepared")
        print("3. Verify CUDA/GPU availability if using GPU")
        
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