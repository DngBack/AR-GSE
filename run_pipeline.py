#!/usr/bin/env python3
"""
Complete AR-GSE Training Pipeline
Runs the complete training pipeline from data preparation to evaluation.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete AR-GSE training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--start-from',
        type=str,
        choices=['data', 'experts', 'gating', 'argse', 'eval'],
        default='data',
        help='Starting step of the pipeline'
    )
    
    parser.add_argument(
        '--stop-at',
        type=str,
        choices=['data', 'experts', 'gating', 'argse', 'eval'],
        default='eval',
        help='Stopping step of the pipeline'
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
        help='Show pipeline without execution'
    )
    
    return parser.parse_args()

def run_command(command, description, args):
    """Run a command with error handling."""
    print(f"\n{'='*60}")
    print(f"🎯 {description}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print(f"🔍 [DRY RUN] Would execute: {' '.join(command)}")
        return True
    
    try:
        if args.verbose:
            print(f"Executing: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    # Define pipeline steps
    steps = {
        'data': {
            'command': ['python', 'create_splits.py'],
            'description': 'Data Preparation (CIFAR-100-LT Splits)'
        },
        'experts': {
            'command': ['python', 'train_experts.py'] + (['--verbose'] if args.verbose else []),
            'description': 'Expert Models Training'
        },
        'gating': {
            'command': [
                ['python', 'train_gating.py', '--mode', 'pretrain'] + (['--verbose'] if args.verbose else []),
                ['python', 'train_gating.py', '--mode', 'selective'] + (['--verbose'] if args.verbose else [])
            ],
            'description': 'Gating Models Training'
        },
        'argse': {
            'command': ['python', 'train_argse.py'] + (['--verbose'] if args.verbose else []),
            'description': 'AR-GSE Ensemble Training'
        },
        'eval': {
            'command': ['python', 'evaluate_argse.py'] + (['--verbose'] if args.verbose else []),
            'description': 'AR-GSE Evaluation'
        }
    }
    
    # Determine step order
    step_order = ['data', 'experts', 'gating', 'argse', 'eval']
    start_idx = step_order.index(args.start_from)
    stop_idx = step_order.index(args.stop_at)
    
    if start_idx > stop_idx:
        print("❌ Error: Start step comes after stop step")
        sys.exit(1)
    
    steps_to_run = step_order[start_idx:stop_idx + 1]
    
    print("🚀 AR-GSE COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    print(f"Pipeline steps: {' → '.join(steps_to_run)}")
    print(f"Device: {args.device}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No actual execution")
    
    # Execute pipeline steps
    for step in steps_to_run:
        step_info = steps[step]
        
        # Handle special case for gating (two commands)
        if step == 'gating':
            for i, cmd in enumerate(step_info['command']):
                mode = 'pretrain' if i == 0 else 'selective'
                desc = f"{step_info['description']} ({mode})"
                success = run_command(cmd, desc, args)
                if not success and not args.dry_run:
                    print(f"\n❌ Pipeline failed at step: {step} ({mode})")
                    sys.exit(1)
        else:
            success = run_command(step_info['command'], step_info['description'], args)
            if not success and not args.dry_run:
                print(f"\n❌ Pipeline failed at step: {step}")
                sys.exit(1)
    
    # Pipeline completion
    print(f"\n{'='*60}")
    print("🏁 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    if not args.dry_run:
        print("\n📊 Results available in:")
        print("   - checkpoints/: Trained models")
        print("   - results_worst_eg_improved/: Evaluation results")
        print("   - data/cifar100_lt_if100_splits/: Dataset splits")
        
        print("\n🎯 Next steps:")
        print("   - python demo_inference.py (Quick demonstration)")
        print("   - python comprehensive_inference.py (Detailed analysis)")
    else:
        print(f"\n🔍 Dry run completed. Pipeline would execute {len(steps_to_run)} steps.")

if __name__ == "__main__":
    main()