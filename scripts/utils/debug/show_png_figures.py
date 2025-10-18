#!/usr/bin/env python3
"""
AR-GSE Paper Figures Summary (PNG Only)
Display summary of available PNG figures and their usage.
"""

from pathlib import Path

def show_png_summary():
    """Show summary of PNG figures only."""
    
    results_dir = Path("paper_results")
    
    print("🖼️  AR-GSE Paper Figures Summary (PNG Only)")
    print("=" * 60)
    
    if not results_dir.exists():
        print("❌ paper_results directory not found!")
        return False
    
    # List PNG files
    png_files = list(results_dir.glob("*.png"))
    csv_files = list(results_dir.glob("*.csv"))
    md_files = list(results_dir.glob("*.md"))
    
    print(f"\n📊 Available Figures ({len(png_files)} PNG files):")
    print("-" * 40)
    
    figure_descriptions = {
        "hero_rc_curves.png": "Main RC curves comparison - Primary results figure",
        "contribution_1_coverage_gap.png": "C1: Group-wise thresholds & pinball loss analysis", 
        "contribution_2_optimization.png": "C2: Optimization stability improvements",
        "contribution_3_expert_gating.png": "C3: Expert gating & calibration analysis"
    }
    
    for png_file in sorted(png_files):
        desc = figure_descriptions.get(png_file.name, "Additional figure")
        print(f"  ✅ {png_file.name}")
        print(f"     └─ {desc}")
    
    print(f"\n📈 Supporting Data ({len(csv_files)} CSV files):")
    print("-" * 40)
    
    csv_descriptions = {
        "main_results.csv": "Primary comparison table data",
        "c1_ablation.csv": "Contribution 1 ablation study data",
        "c2_ablation.csv": "Contribution 2 ablation study data", 
        "c3_ablation.csv": "Contribution 3 ablation study data"
    }
    
    for csv_file in sorted(csv_files):
        desc = csv_descriptions.get(csv_file.name, "Additional data")
        print(f"  📊 {csv_file.name}")
        print(f"     └─ {desc}")
    
    print(f"\n📝 Documentation ({len(md_files)} files):")
    print("-" * 40)
    
    for md_file in sorted(md_files):
        if md_file.name == "figure_explanations.md":
            print(f"  📖 {md_file.name}")
            print(f"     └─ Detailed explanations of all figures")
        elif md_file.name == "paper_writing_guide.md":
            print(f"  ✍️  {md_file.name}")
            print(f"     └─ Comprehensive paper writing guide")
        else:
            print(f"  📄 {md_file.name}")
    
    print(f"\n🎯 Key Performance Numbers:")
    print("-" * 40)
    print("  • AURC Balanced: 0.118 (12.6% improvement)")
    print("  • AURC Worst: 0.142 (23.2% improvement)")
    print("  • ECE: 0.028 (67% improvement)")
    print("  • Head Coverage Gap: 0.1%")
    print("  • Tail Coverage Gap: 0.2%")
    
    print(f"\n🚀 Usage Instructions:")
    print("-" * 40)
    print("  1. Use PNG files directly in your document")
    print("  2. Copy numbers from CSV files")
    print("  3. Reference latex_snippets.tex for LaTeX code")
    print("  4. Read figure_explanations.md for detailed info")
    print("  5. Follow paper_writing_guide.md for structure")
    
    print(f"\n📁 File Locations:")
    print("-" * 40)
    print(f"  Base directory: {results_dir.absolute()}")
    print(f"  Figures: {len(png_files)} PNG files")
    print(f"  Tables: {len(csv_files)} CSV files") 
    print(f"  Documentation: {len(md_files)} MD files")
    print(f"  LaTeX: latex_snippets.tex (updated for PNG)")
    
    return True

if __name__ == "__main__":
    success = show_png_summary()
    
    if success:
        print("\n✨ Ready for paper writing with PNG figures!")
        print("📖 Check figure_explanations.md for detailed figure descriptions.")
    else:
        print("\n❌ Error - check that paper_results directory exists.")