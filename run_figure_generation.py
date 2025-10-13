#!/usr/bin/env python3
"""
Quick script to generate paper figures for AR-GSE.
Run this to create all necessary plots and tables for the paper.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages for figure generation."""
    required_packages = [
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0', 
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0'
    ]
    
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}, continuing...")

def run_figure_generation():
    """Run the figure generation script."""
    print("\n🎨 Generating AR-GSE paper figures...")
    
    try:
        # Run the plot generation script
        subprocess.check_call([sys.executable, 'generate_paper_plots.py'])
        print("\n✅ Figure generation completed successfully!")
        
        # Check output
        output_dir = Path("paper_figures")
        if output_dir.exists():
            files = list(output_dir.iterdir())
            print(f"\n📁 Generated {len(files)} files in {output_dir}")
            
            # Show key files
            key_files = [
                'figure_hero_rc_curves.pdf',
                'figure_c1_coverage_gap.pdf', 
                'figure_c2_optimization.pdf',
                'figure_c3_expert_gating.pdf'
            ]
            
            print("\n🎯 Key figures for paper:")
            for key_file in key_files:
                if (output_dir / key_file).exists():
                    print(f"  ✅ {key_file}")
                else:
                    print(f"  ❌ {key_file} (missing)")
                    
        else:
            print("❌ Output directory not found!")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating figures: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🚀 AR-GSE Paper Figure Generation")
    print("=" * 40)
    
    # Install requirements
    install_requirements()
    
    # Generate figures
    success = run_figure_generation()
    
    if success:
        print("\n🎉 All done! Your paper figures are ready.")
        print("\n📝 Next steps:")
        print("1. Check the 'paper_figures' directory")
        print("2. Use PDF files for your paper")
        print("3. Use PNG files for presentations")
        print("4. Read README_figures.md for detailed descriptions")
    else:
        print("\n❌ Something went wrong. Please check the error messages above.")

if __name__ == "__main__":
    main()