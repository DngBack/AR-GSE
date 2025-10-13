#!/usr/bin/env python3
"""
AR-GSE Evaluation Script
Evaluates the trained AR-GSE ensemble model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AR-GSE ensemble model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific checkpoint to evaluate'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['test', 'val', 'tunev'],
        default='test',
        help='Dataset split to evaluate on'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for evaluation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed evaluation results'
    )
    
    return parser.parse_args()

def setup_evaluation_environment(args):
    """Setup evaluation environment."""
    try:
        import torch
        
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        print("🚀 AR-GSE Evaluation Pipeline")
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset}")
        
        if args.verbose:
            print(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name()}")
        
        return device
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        sys.exit(1)

def check_evaluation_prerequisites():
    """Check if evaluation prerequisites are met."""
    print("🔍 Checking evaluation prerequisites...")
    
    # Check if trained model exists
    model_dir = Path("checkpoints/argse_worst_eg_improved_v3_3/cifar100_lt_if100")
    if not model_dir.exists():
        raise Exception("AR-GSE model directory not found. Please train AR-GSE first.")
    
    # Look for model files
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        raise Exception("No trained AR-GSE models found. Please train AR-GSE first.")
    
    # Check data splits
    splits_dir = Path("data/cifar100_lt_if100_splits")
    if not splits_dir.exists():
        raise Exception("Data splits not found. Please prepare data splits first.")
    
    print("✅ Evaluation prerequisites met")
    return model_files

def run_evaluation(args):
    """Run the AR-GSE evaluation."""
    try:
        print("🎯 Running AR-GSE evaluation")
        
        # Execute the evaluation script
        import subprocess
        
        command = ["python", "-m", "src.train.eval_agse_plugin"]
        
        if args.verbose:
            print(f"Executing: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=False,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("✅ Successfully completed AR-GSE evaluation")
            return True
        else:
            raise Exception(f"Evaluation failed with exit code: {result.returncode}")
            
    except Exception as e:
        raise Exception(f"Failed to run AR-GSE evaluation: {str(e)}")

def main():
    """Main function for AR-GSE evaluation."""
    args = parse_arguments()
    
    print("=" * 60)
    print("AR-GSE EVALUATION")
    print("=" * 60)
    
    # Setup environment
    setup_evaluation_environment(args)
    
    try:
        # Check prerequisites
        model_files = check_evaluation_prerequisites()
        
        print(f"\n{'='*40}")
        print("🎯 AR-GSE Model Evaluation")
        print(f"{'='*40}")
        
        if args.verbose:
            print(f"\nFound {len(model_files)} model files:")
            for model_file in model_files:
                print(f"  - {model_file.name}")
        
        print("\n📋 Evaluation Process:")
        print("  - Load trained AR-GSE model")
        print(f"  - Evaluate on {args.dataset} dataset")
        print("  - Generate metrics and results")
        print("  - Compute rejection learning performance")
        
        # Run the evaluation
        run_evaluation(args)
        
        print(f"\n{'='*60}")
        print("🏁 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print("✅ AR-GSE evaluation completed successfully!")
        
        # Show results locations
        print("\n📊 Results saved to:")
        print("   - results_worst_eg_improved/cifar100_lt_if100/metrics.json")
        print("   - results_worst_eg_improved/cifar100_lt_if100/rc_curve.csv")
        print("   - results_worst_eg_improved/cifar100_lt_if100/aurc_detailed_results.csv")
        
        print("\n➡️  Next steps:")
        print("   - Review evaluation metrics")
        print("   - Run comprehensive inference: python comprehensive_inference.py")
        print("   - Run demo inference: python demo_inference.py")
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        
        print("\nTroubleshooting tips:")
        print("1. Ensure AR-GSE model is trained successfully")
        print("2. Check that data splits are prepared")
        print("3. Verify model checkpoint paths")
        print("4. Check available memory if using GPU")
        
        if args.verbose:
            import traceback
            print("\nFull error traceback:")
            traceback.print_exc()
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()