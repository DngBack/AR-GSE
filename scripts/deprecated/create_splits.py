#!/usr/bin/env python3
"""
Script to create CIFAR-100-LT dataset splits.
This script creates the train, validation, test, and tuneV splits for CIFAR-100-LT dataset
with imbalance factor of 100.
"""

import sys
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.enhanced_datasets import create_full_cifar100_lt_splits

def main():
    """Main function to create CIFAR-100-LT splits."""
    print("Starting CIFAR-100-LT dataset splits creation...")
    print("-" * 50)
    
    try:
        # Create the splits with default parameters
        datasets, splits = create_full_cifar100_lt_splits(
            imb_factor=100,                           # Imbalance factor
            output_dir="data/cifar100_lt_if100_splits",  # Output directory
            val_ratio=0.2,                            # Validation ratio
            tunev_ratio=0.15,                         # TuneV ratio
            seed=42                                   # Random seed for reproducibility
        )
        
        print("\n" + "=" * 50)
        print("SUCCESS: Dataset splits created successfully!")
        print("=" * 50)
        
        # Print summary information
        print("\nDataset Summary:")
        for name, dataset in datasets.items():
            print(f"  {name.upper():<8}: {len(dataset):>6,} samples")
        
        print("\nOutput files saved to: data/cifar100_lt_if100_splits/")
        print("Files created:")
        output_dir = Path("data/cifar100_lt_if100_splits")
        if output_dir.exists():
            for file_path in sorted(output_dir.glob("*.json")):
                print(f"  - {file_path.name}")
        
    except Exception as e:
        print("\nERROR: Failed to create dataset splits!")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("1. CIFAR-100 data is available")
        print("2. Required dependencies are installed")
        print("3. Sufficient disk space is available")
        sys.exit(1)

if __name__ == "__main__":
    main()