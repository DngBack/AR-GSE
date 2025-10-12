#!/usr/bin/env python3
"""
Advanced script to create CIFAR-100-LT dataset splits with customizable parameters.
Supports command-line arguments for flexible configuration.
"""

import sys
import argparse
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.enhanced_datasets import create_full_cifar100_lt_splits

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create CIFAR-100-LT dataset splits with customizable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--imb-factor', 
        type=float, 
        default=100.0,
        help='Imbalance factor for long-tail distribution'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default="data/cifar100_lt_if100_splits",
        help='Output directory for split files'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ratio of validation set from combined val+test'
    )
    
    parser.add_argument(
        '--tunev-ratio',
        type=float,
        default=0.15,
        help='Ratio of tuneV set from test set'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command-line arguments."""
    errors = []
    
    if args.imb_factor <= 1.0:
        errors.append("Imbalance factor must be greater than 1.0")
    
    if not (0.0 < args.val_ratio < 1.0):
        errors.append("Validation ratio must be between 0.0 and 1.0")
    
    if not (0.0 < args.tunev_ratio < 1.0):
        errors.append("TuneV ratio must be between 0.0 and 1.0")
    
    if args.seed < 0:
        errors.append("Random seed must be non-negative")
    
    if errors:
        print("ERROR: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

def main():
    """Main function to create CIFAR-100-LT splits."""
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)
    
    if args.verbose:
        print("Configuration:")
        print(f"  Imbalance Factor: {args.imb_factor}")
        print(f"  Output Directory: {args.output_dir}")
        print(f"  Validation Ratio: {args.val_ratio}")
        print(f"  TuneV Ratio: {args.tunev_ratio}")
        print(f"  Random Seed: {args.seed}")
        print()
    
    print("Starting CIFAR-100-LT dataset splits creation...")
    print("-" * 50)
    
    try:
        # Create the splits with specified parameters
        datasets, splits = create_full_cifar100_lt_splits(
            imb_factor=args.imb_factor,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            tunev_ratio=args.tunev_ratio,
            seed=args.seed
        )
        
        print("\n" + "=" * 50)
        print("SUCCESS: Dataset splits created successfully!")
        print("=" * 50)
        
        # Print summary information
        print("\nDataset Summary:")
        total_samples = sum(len(dataset) for dataset in datasets.values())
        for name, dataset in datasets.items():
            percentage = (len(dataset) / total_samples) * 100
            print(f"  {name.upper():<8}: {len(dataset):>6,} samples ({percentage:5.1f}%)")
        
        print(f"\nTotal samples: {total_samples:,}")
        
        # Print file information
        output_dir = Path(args.output_dir)
        print(f"\nOutput files saved to: {args.output_dir}/")
        if output_dir.exists():
            print("Files created:")
            for file_path in sorted(output_dir.glob("*.json")):
                file_size = file_path.stat().st_size
                print(f"  - {file_path.name:<25} ({file_size:,} bytes)")
        
        # Calculate and display imbalance statistics
        if args.verbose:
            print("\nImbalance Statistics:")
            train_dataset = datasets['train']
            print(f"  Training set size: {len(train_dataset):,}")
            print(f"  Configured IF: {args.imb_factor}")
            print(f"  Validation ratio: {args.val_ratio}")
            print(f"  TuneV ratio: {args.tunev_ratio}")
        
    except ImportError as e:
        print("\nERROR: Failed to import required modules!")
        print(f"Error message: {str(e)}")
        print("\nPlease ensure:")
        print("1. The 'src' directory exists with required modules")
        print("2. All dependencies are installed (torch, torchvision, etc.)")
        sys.exit(1)
        
    except Exception as e:
        print("\nERROR: Failed to create dataset splits!")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("1. CIFAR-100 data can be downloaded/accessed")
        print("2. Sufficient disk space is available")
        print("3. Output directory is writable")
        print("4. Parameters are valid")
        sys.exit(1)

if __name__ == "__main__":
    main()