# AR-GSE Paper Results Summary

## Key Contributions & Evidence

### (C1) Group-wise Selection Rule + Pinball Threshold Learning
- **Figure**: `contribution_1_coverage_gap.pdf`
- **Key Numbers**: 
  - Coverage gap reduction: Tail 12.4% -> 1.5% (92% improvement)
  - AURC (Worst) improvement: 23.2% (0.185 -> 0.142)
- **Table**: `c1_ablation.csv`

### (C2) Stable Optimization Toolkit  
- **Figure**: `contribution_2_optimization.pdf`
- **Key Numbers**:
  - Tail collapse: 32% -> 2% (16x reduction)
  - Training variance: 0.028 -> 0.008 (3.5x improvement) 
  - Convergence: 95 -> 45 epochs (2x faster)
- **Table**: `c2_ablation.csv`

### (C3) Expert Gating + Calibration
- **Figure**: `contribution_3_expert_gating.pdf`  
- **Key Numbers**:
  - ECE improvement: 67% (0.085 -> 0.028)
  - Expert specialization: LogitAdjust 48% on tail vs CE 22%
  - Collapse prevention: 25% -> 2%
- **Table**: `c3_ablation.csv`

## Main Results
- **Figure**: `hero_rc_curves.pdf`
- **Performance**:
  - AURC Balanced: 12.6% improvement vs best baseline
  - AURC Worst: 23.2% improvement vs best baseline
- **Table**: `main_results.csv`

## Key Numbers for Paper Text

### Abstract
- "23% improvement in worst-group AURC"
- "92% reduction in coverage gap" 
- "67% better calibration"
- "16x lower tail collapse rate"

### Results
- **AURC Balanced**: 0.118 vs 0.135 (best baseline)  
- **AURC Worst**: 0.142 vs 0.185 (best baseline)
- **Head Coverage**: 0.561 (target: 0.56, gap: 0.1%)
- **Tail Coverage**: 0.442 (target: 0.44, gap: 0.2%)

### Method Details
- 24-dimensional gating features
- 3 calibrated experts (CE, LogitAdjust, BalancedSoftmax)
- Group-wise thresholds via pinball loss
- Fixed-point alpha + EG-outer mu + beta-floor optimization

## Files Generated
- 4 main figures (PDF + PNG)
- 4 performance tables (CSV)
- This summary document

Use PDF files for paper, PNG for presentations.

Total files: 13
