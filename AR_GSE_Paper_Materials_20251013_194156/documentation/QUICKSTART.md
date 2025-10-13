# Quick Start Guide

## 🔥 For Urgent Paper Writing

### Step 1: Get the Numbers
Open `figures_and_tables/main_results.csv` and copy these key numbers:
- **AR-GSE AURC Balanced: 0.118**
- **AR-GSE AURC Worst: 0.142** 
- **Best baseline AURC Worst: 0.185**
- **Improvement: 23.2%**

### Step 2: Add the Figures  
Copy PDF files to your LaTeX project:
```bash
cp figures_and_tables/*.pdf your_paper/figures/
```

### Step 3: LaTeX Integration
Copy the code from `figures_and_tables/latex_snippets.tex` directly into your paper.

### Step 4: Write the Results
Follow the structure in `figures_and_tables/paper_writing_guide.md`.

## 🎯 Three Key Contributions

1. **C1 - Coverage Gap Analysis**: 23.2% AURC improvement through group-wise thresholds and pinball loss
2. **C2 - Optimization Stability**: 71.4% training variance reduction via fixed-point updates and EG-outer
3. **C3 - Expert Gating & Calibration**: 67% ECE improvement through calibrated expert gating

## 📈 Performance Highlights

- **Best-in-class selective prediction** on long-tail data
- **Balanced head-tail coverage** with minimal gap (0.1-0.2%)
- **Stable optimization** preventing expert collapse
- **Strong calibration** with 67% ECE improvement

Ready to write! 🚀
