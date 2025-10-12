#!/usr/bin/env python3
"""
Script to clean up CIFAR-100-LT splits files.
Removes existing split files to start fresh.
"""

import argparse
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up CIFAR-100-LT splits files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--splits-dir',
        type=str,
        default="data/cifar100_lt_if100_splits",
        help='Directory containing split files to clean'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    return parser.parse_args()

def main():
    """Main function to clean splits."""
    args = parse_arguments()
    
    splits_dir = Path(args.splits_dir)
    
    if not splits_dir.exists():
        print(f"Directory {args.splits_dir} does not exist. Nothing to clean.")
        return
    
    # Find all JSON files in the directory
    json_files = list(splits_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {args.splits_dir}. Nothing to clean.")
        return
    
    # Show files to be deleted
    print(f"Found {len(json_files)} split files in {args.splits_dir}:")
    for file_path in json_files:
        file_size = file_path.stat().st_size
        print(f"  - {file_path.name} ({file_size:,} bytes)")
    
    # Confirmation
    if not args.force:
        response = input("\nDo you want to delete these files? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    # Delete files
    deleted_count = 0
    for file_path in json_files:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"Deleted: {file_path.name}")
        except Exception as e:
            print(f"Error deleting {file_path.name}: {e}")
    
    print(f"\nSuccessfully deleted {deleted_count}/{len(json_files)} files.")
    
    # Check if directory is empty and offer to remove it
    remaining_files = list(splits_dir.glob("*"))
    if not remaining_files:
        if not args.force:
            response = input(f"Directory {args.splits_dir} is now empty. Delete it? [y/N]: ")
            if response.lower() in ['y', 'yes']:
                try:
                    splits_dir.rmdir()
                    print(f"Deleted empty directory: {args.splits_dir}")
                except Exception as e:
                    print(f"Error deleting directory: {e}")
        else:
            try:
                splits_dir.rmdir()
                print(f"Deleted empty directory: {args.splits_dir}")
            except Exception as e:
                print(f"Error deleting directory: {e}")

if __name__ == "__main__":
    main()