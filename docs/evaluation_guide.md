# AR-GSE Evaluation - Quick Reference

## Overview

The evaluation phase computes AURC (Area Under Risk-Coverage) curves to measure the quality of selective prediction using the trained AR-GSE model.

## Methodology

### Key Principle: **Balanced Data + Reweighted Metrics**
- **Train on balanced splits** (tunev, val, test - all from test set)
- **Reweight metrics** using class_weights.json to reflect long-tail performance
- **No data duplication** - all splits are naturally balanced

### Evaluation Pipeline
1. **Load plugin checkpoint**: α*, μ*, class_to_group, gating network
2. **Compute confidence scores**: GSE margins from mixture posteriors
3. **Optimize thresholds**: On validation (tunev + val) for each rejection cost c
4. **Evaluate on test**: Measure coverage and risk with reweighted metrics
5. **Compute AURC**: Integrate risk over coverage

## Data Splits

### All from Test Set (Balanced)
```
Test set (10,000 samples) split into:
├── tunev: 1,000 samples (10 per class) - for threshold tuning
├── val:   1,000 samples (10 per class) - for validation
└── test:  8,000 samples (80 per class) - for evaluation

Validation = tunev + val (2,000 samples)
Evaluation = test (8,000 samples)
```

### Why This Works
- **Balanced splits** = clean data, no artificial duplication
- **Reweighted metrics** = simulate long-tail performance
- **Separate train/test** = no data leakage (experts trained on train set)

## Required Files

### 1. Plugin Checkpoint
```
checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt
```
Contains: α*, μ*, class_to_group, gating_net_state_dict, num_groups

### 2. Expert Logits
```
outputs/logits/cifar100_lt_if100/{expert_name}/
├── tunev_logits.npz  (or .pt)
├── val_logits.npz
└── test_logits.npz
```

### 3. Split Indices
```
data/cifar100_lt_if100_splits_fixed/
├── tunev_indices.json
├── val_indices.json
└── test_indices.json
```

### 4. Class Weights (for reweighting)
```
data/cifar100_lt_if100_splits_fixed/class_weights.json
```

## Commands

### Test Setup
```bash
# Verify all files are ready
python test_eval_setup.py
```

### Run Evaluation
```bash
# Full AURC evaluation with reweighting
python -m src.train.eval_gse_plugin
```

### Configuration

Edit `CONFIG` in `src/train/eval_gse_plugin.py`:

```python
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'aurc_eval': {
        'mode': 'detailed',  # 'fast', 'detailed', or 'full'
        'metrics': ['balanced', 'worst'],  # Group-aware metrics
        'interpolate_smooth_curves': True,
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
}
```

### Evaluation Modes

**Fast Mode** (9 points):
- Quick evaluation, strategic cost values
- Good for debugging and quick checks
- Runtime: ~1-2 minutes

**Detailed Mode** (21 points) - **Recommended**:
- Better visualization, smoother curves
- Good balance between speed and quality
- Runtime: ~3-5 minutes

**Full Mode** (41 points):
- Publication-quality smooth curves
- Most accurate AURC computation
- Runtime: ~5-10 minutes

## Output Files

### 1. aurc_detailed_results.csv
```csv
metric,cost,coverage,risk
balanced,0.0,1.0,0.234
balanced,0.05,0.95,0.189
...
worst,0.0,1.0,0.456
worst,0.05,0.92,0.398
...
```

### 2. aurc_summary.json
```json
{
    "balanced": 0.1234,
    "balanced_02_10": 0.0987,
    "worst": 0.2345,
    "worst_02_10": 0.1876
}
```

### 3. aurc_curves.png
Three subplots:
1. **Full Range (0-1)**: Complete risk-coverage curves
2. **Focused Range (0.2-1)**: Zoomed view for practical coverage
3. **AURC Comparison**: Bar chart comparing metrics

## Metrics Explained

### Balanced Error
- **Definition**: Average of group-wise errors
- **Formula**: `(error_head + error_tail) / 2`
- **Reweighting**: Each group weighted by class distribution
- **Use**: Overall fairness across groups

### Worst-Group Error
- **Definition**: Maximum error across groups
- **Formula**: `max(error_head, error_tail)`
- **Reweighting**: Applied within each group
- **Use**: Ensure no group left behind

### AURC Interpretation
- **Lower is better** (less area under curve = better performance)
- **Full range (0-1)**: Complete evaluation
- **Practical range (0.2-1)**: Excludes extreme low coverage
- **Comparison**: Balanced vs Worst shows group disparity

## How Reweighting Works

### Standard Accuracy (uniform)
```python
accuracy = correct_predictions / total_predictions
```

### Reweighted Accuracy (long-tail)
```python
# Per-class accuracy
for class_c in classes:
    class_acc[c] = correct[c] / total[c]

# Weight by training distribution
weighted_acc = sum(class_acc[c] * class_weight[c])
```

### Example
Training distribution (IF=100):
- **Head class**: 500 samples → weight = 0.046
- **Tail class**: 5 samples → weight = 0.0005
- **Ratio**: 100x difference

On balanced test data (10 per class):
- Without reweighting: each class contributes 1%
- With reweighting: head contributes 4.6%, tail contributes 0.05%
- Reflects actual long-tail performance

## Troubleshooting

### Missing Plugin Checkpoint
```bash
# Train plugin first
python run_improved_eg_outer.py
```

### Missing Expert Logits
```bash
# Experts should export logits during training
python train_experts.py
```

### Wrong Paths
Check CONFIG paths match your directory structure:
- `splits_dir`: data/cifar100_lt_if100_splits_fixed
- `logits_dir`: outputs/logits/cifar100_lt_if100/
- `plugin_checkpoint`: checkpoints/argse_worst_eg_improved/.../gse_balanced_plugin.ckpt

### No Class Weights
- **Effect**: Evaluation uses uniform weighting
- **Fix**: Not critical, but reweighting provides better long-tail metrics
- **Generate**: Created during split creation (create_splits_fixed.py)

### Verification Failed
```bash
python test_eval_setup.py
```
This will show exactly what's missing.

## Complete Workflow

```bash
# 1. Create splits with class weights
python create_splits_fixed.py --split-train --visualize

# 2. Train experts (exports logits)
python train_experts.py

# 3. Train gating (pretrain + selective)
python train_gating.py --mode pretrain
python train_gating.py --mode selective

# 4. Train plugin
python run_improved_eg_outer.py

# 5. Verify evaluation setup
python test_eval_setup.py

# 6. Run AURC evaluation
python -m src.train.eval_gse_plugin

# 7. Check results
ls -lh results_worst_eg_improved/cifar100_lt_if100/
```

## Understanding Results

### Good Performance
- **Low AURC** (< 0.15 for balanced, < 0.25 for worst)
- **Small gap** between balanced and worst (~10-20%)
- **High coverage** at low risk (curves near origin)
- **Smooth curves** (no erratic behavior)

### Poor Performance
- **High AURC** (> 0.3)
- **Large gap** between balanced and worst (>50%)
- **Low coverage** needed for acceptable risk
- **Unstable curves** (oscillations, plateaus)

### What to Look For
1. **Curve shape**: Steep drop at high coverage = good selective prediction
2. **Group disparity**: Gap between balanced/worst = fairness issue
3. **Coverage range**: Can achieve <10% error at what coverage?
4. **Tail performance**: Worst metric reflects tail class quality

## Key Insights

✅ **Balanced data**: Clean, no duplication issues
✅ **Reweighted metrics**: Reflects long-tail reality
✅ **Proper validation**: Optimize on tunev+val, evaluate on test
✅ **Group-aware**: Balanced and worst metrics ensure fairness
✅ **No leakage**: Train/test separation maintained throughout

The evaluation properly measures how AR-GSE handles long-tail distribution while training on balanced data with reweighted metrics.
