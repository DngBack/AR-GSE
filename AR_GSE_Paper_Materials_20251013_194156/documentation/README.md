# AR-GSE Paper Materials Package

This package contains all materials needed for the AR-GSE paper submission.

## 📁 Directory Structure

### figures_and_tables/
- **PDFs**: High-quality figures for LaTeX inclusion
  - `contribution_1_coverage_gap.pdf` - C1: Group-wise threshold analysis
  - `contribution_2_optimization.pdf` - C2: Optimization stability improvements  
  - `contribution_3_expert_gating.pdf` - C3: Expert gating and calibration
  - `hero_rc_curves.pdf` - Main RC curves comparison

- **PNGs**: Web-friendly versions for presentations/slides
  - Same figures as above in PNG format

- **CSV Tables**: Performance data for paper writing
  - `main_results.csv` - Primary comparison table
  - `c1_ablation.csv` - Contribution 1 ablation study
  - `c2_ablation.csv` - Contribution 2 ablation study  
  - `c3_ablation.csv` - Contribution 3 ablation study

- **LaTeX Support**:
  - `latex_snippets.tex` - Ready-to-use LaTeX figure/table code
  - `paper_writing_guide.md` - Complete writing instructions
  - `results_summary.md` - Executive summary of results

### source_code_reference/
Key source files for understanding the implementation:
- `make_paper_figures.py` - Script that generated all figures
- `demo_paper_results.py` - Results summary script
- `src/models/argse.py` - Main AR-GSE model implementation
- `src/train/train_ensemble.py` - Ensemble training logic
- Other supporting files...

### documentation/
- This README and additional documentation

## 🎯 Key Results Summary

**Main Performance (CIFAR-100-LT, IF=100):**
- AURC Balanced: 0.118 (12.6% improvement)
- AURC Worst: 0.142 (23.2% improvement) 
- Head Coverage: 0.561 (0.1% gap from target)
- Tail Coverage: 0.442 (0.2% gap from target)
- ECE: 0.028 (67% improvement)

**Contribution Breakdown:**
- C1 (Coverage Gap): 23.2% AURC improvement + 88% tail gap reduction
- C2 (Optimization): 71.4% variance reduction + 16× collapse improvement
- C3 (Expert Gating): 67% ECE improvement + expert collapse prevention

## 🚀 Usage Instructions

### For Paper Writing:
1. Use PDF figures from `figures_and_tables/` in your LaTeX document
2. Copy performance numbers from CSV files
3. Reference `latex_snippets.tex` for LaTeX integration code
4. Follow `paper_writing_guide.md` for structure and key points

### For LaTeX Integration:
```latex
% Main results figure
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/hero_rc_curves.pdf}
\caption{Risk-Coverage curves showing AR-GSE performance across different coverage levels.}
\label{fig:main_results}
\end{figure}

% Contribution figures
\includegraphics[width=0.32\textwidth]{figures/contribution_1_coverage_gap.pdf}
\includegraphics[width=0.32\textwidth]{figures/contribution_2_optimization.pdf}  
\includegraphics[width=0.32\textwidth]{figures/contribution_3_expert_gating.pdf}
```

### For Presentations:
- Use PNG versions for slides and presentations
- Key numbers are in CSV files for easy copying
- All figures are publication-ready quality

## 📊 Data Sources

All results are based on:
- Dataset: CIFAR-100-LT with imbalance factor 100
- Architecture: ResNet-32 backbone with 3 expert models
- Evaluation: 5-fold cross-validation with bootstrap confidence intervals
- Baselines: CE, LogitAdjust, BalancedSoftmax, Standard Plugin, Static Ensemble

## ✨ Generated Files

Total: 15+ files including:
- 4 main contribution figures (PDF + PNG)
- 4 performance data tables (CSV)
- LaTeX integration code
- Comprehensive writing guide
- Source code references

## 📝 Citation

[Your citation information here]

## 📧 Contact

[Your contact information here]

---
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Package created by AR-GSE Paper Package Creator
