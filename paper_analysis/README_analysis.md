# AR-GSE Paper Analysis Summary

## Generated Figures

### Figure 1: Coverage Gap Analysis (C1 Contribution)
- **File**: `figure_1_coverage_gap_analysis.pdf`
- **Shows**: Group-wise threshold learning effectiveness
- **Key Result**: Coverage gap reduced from 12.4% to 1.5% for tail group
- **AURC Improvement**: 23.2% better worst-group performance

### Figure 2: Optimization Stability (C2 Contribution)  
- **File**: `figure_2_optimization_stability.pdf`
- **Shows**: Training stability with different optimization components
- **Key Result**: Tail collapse rate reduced from 32% to 2%
- **Variance**: Training variance improved by 3.5x (0.028 to 0.008)

### Figure 3: Expert Gating Analysis (C3 Contribution)
- **File**: `figure_3_expert_gating_analysis.pdf`  
- **Shows**: Expert specialization and calibration improvements
- **Key Result**: ECE improved by 67% (0.085 to 0.028)
- **Specialization**: LogitAdjust 50% usage on tail vs CE 20%

### Figure 4: Hero RC Curves (Main Result)
- **File**: `figure_4_hero_rc_curves.pdf`
- **Shows**: Risk-Coverage curves for all methods
- **Key Result**: Consistent superiority in 60-90% coverage region
- **Performance**: 12.6% balanced, 23.2% worst-group improvement

## Generated Tables

### Main Results (`table_main_results.csv`)
- Complete performance comparison across all methods
- Includes standard deviations over 5 seeds
- Coverage analysis by group

### Ablation Studies
- `table_c1_ablation.csv`: Group-wise thresholds + pinball learning
- `table_c2_ablation.csv`: Optimization stability components  
- `table_c3_ablation.csv`: Expert gating and calibration components

## Key Numbers for Paper

### Abstract/Introduction
- "23% improvement in worst-group AURC (0.185 to 0.142)"
- "Coverage gap reduced from 12% to 1.5%"  
- "67% calibration improvement (ECE: 0.085 to 0.028)"
- "16x reduction in tail collapse rate (32% to 2%)"

### Results Section
- **AURC Balanced**: 0.118 +/- 0.003 (vs 0.135 best baseline)
- **AURC Worst**: 0.142 +/- 0.005 (vs 0.185 best baseline)
- **Head Coverage**: 0.561 (target: 0.56, gap: 0.1%)
- **Tail Coverage**: 0.442 (target: 0.44, gap: 0.2%)

### Method Contributions
- **C1**: Group-wise thresholds learned via pinball loss
- **C2**: Stable optimization (FP-alpha + EG-mu + beta-floor)  
- **C3**: Calibrated expert gating with 24D features

## LaTeX Integration

All figures saved as both PDF (vector) and PNG (raster).
Recommended usage:
- PDF for paper submission
- PNG for presentations/slides

Figure widths:
- Single column: ~3.5 inches
- Double column: ~7 inches

Files ready for \includegraphics in LaTeX.

## Paper Structure Mapping

### Introduction to Figure 4 (Hero curves)
### Method Section to Figures 1-3 (Contributions)  
### Results Section to Figure 4 + Tables
### Ablation Section to Tables + Figure components

Generated 13 files total.
