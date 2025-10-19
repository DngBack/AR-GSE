# AR-GSE Plugin Training - Quick Reference

## Overview

The plugin training phase optimizes the AR-GSE parameters (α, μ, thresholds) on balanced validation data with reweighting to reflect long-tail performance.

## Data Requirements

### Input Splits (from test set - balanced)
- **S1 (tunev)**: 1,000 samples (10 per class) - for parameter tuning
- **S2 (val)**: 1,000 samples (10 per class) - for validation

### Required Files
```
data/cifar100_lt_if100_splits_fixed/
├── tunev_indices.json         ← S1 split
├── val_indices.json           ← S2 split
└── class_weights.json         ← For reweighting

outputs/logits/cifar100_lt_if100/{expert}/
├── tunev_logits.pt           ← Expert predictions on S1
└── val_logits.pt             ← Expert predictions on S2

checkpoints/gating_pretrained/cifar100_lt_if100/
├── gating_pretrained.ckpt    ← Pre-trained gating (optional)
└── gating_selective.ckpt     ← Selective gating (recommended)
```

## Key Features

### 1. Reweighted Metrics ✅
- **Problem**: S1 and S2 are balanced, but we want to optimize for long-tail performance
- **Solution**: Use class weights from training distribution to reweight accuracy
- **Effect**: Metrics reflect how model performs on actual long-tail test data

### 2. Balanced Data, Long-tail Performance
- Train/optimize on balanced splits (clean, no duplication)
- Evaluate with reweighting (simulates long-tail distribution)
- Best of both worlds: clean data + realistic metrics

### 3. Multiple Objectives
- **Balanced**: Minimize average group error (reweighted)
- **Worst**: Minimize worst-group error (reweighted)
- **Hybrid**: Combine both objectives

## Training Commands

### Test Setup
```bash
# Verify all data is ready
python test_plugin_setup.py
```

### Basic Plugin Training
```bash
# Balanced objective
python -m src.train.gse_balanced_plugin

# Worst-group objective (uses EG-outer)
python run_improved_eg_outer.py
```

### Advanced Options
Edit `CONFIG` in `src/train/gse_balanced_plugin.py`:

```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
    },
    'experts': {
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'plugin_params': {
        'objective': 'worst',  # 'balanced', 'worst', or 'hybrid'
        'M': 12,               # Plugin iterations
        'alpha_steps': 5,      # Alpha fixed-point steps
        'cov_target': 0.58,    # Target coverage
        'use_eg_outer': False, # Use EG-outer for worst-group (set True for worst objective)
    }
}
```

## How Reweighting Works

### Standard Accuracy (on balanced data)
```python
# Each class contributes equally
accuracy = correct_predictions / total_predictions
```

### Reweighted Accuracy (simulates long-tail)
```python
# Each class weighted by its frequency in training set
for class_c in classes:
    class_acc[c] = correct[c] / total[c]
    
# Weight by training distribution
weighted_acc = sum(class_acc[c] * class_weight[c])
```

### Example
- **Head class (500 samples in train)**: weight = 0.046
- **Tail class (5 samples in train)**: weight = 0.0005
- **Ratio**: 100x difference

Getting tail classes right matters less in absolute count, but when reweighted by training distribution, the metrics properly reflect model performance on long-tail data.

## Output

### Checkpoint File
```
checkpoints/argse_balanced_plugin/cifar100_lt_if100/gse_balanced_plugin.ckpt
```

Contains:
- `alpha`: Per-group acceptance parameters [K]
- `mu`: Per-group threshold adjustments [K]
- `t_group` or `threshold`: Acceptance thresholds
- `class_to_group`: Class grouping [C]
- `class_weights`: For evaluation (optional)
- `best_score`: Best reweighted error on S2
- `gating_net_state_dict`: Gating network weights

### Metrics Reported
- **Reweighted accuracy**: Simulates long-tail performance
- **Standard accuracy**: On balanced val set
- **Group-wise errors**: Head vs tail performance
- **Coverage**: Fraction of accepted samples

## Workflow Integration

```bash
# Complete pipeline
python create_splits_fixed.py --split-train --visualize  # 1. Create splits
python train_experts.py                                   # 2. Train experts
python train_gating.py --mode pretrain                    # 3. Pretrain gating
python train_gating.py --mode selective                   # 4. Selective gating
python test_plugin_setup.py                               # 5. Verify setup
python run_improved_eg_outer.py                           # 6. Plugin training
python evaluate_argse.py                                  # 7. Final evaluation
```

## Troubleshooting

### Missing logits
```bash
# Re-train experts to export logits
python train_experts.py
```

### Wrong paths
```bash
# Check CONFIG paths in gse_balanced_plugin.py
'splits_dir': './data/cifar100_lt_if100_splits_fixed'
'logits_dir': './outputs/logits/cifar100_lt_if100/'
```

### No gating checkpoint
```bash
# Train gating first
python train_gating.py --mode selective
```

### Verify data integrity
```bash
python test_plugin_setup.py
```

## Notes

1. **Reweighting is automatic**: Class weights loaded from `class_weights.json`
2. **Balanced splits**: S1/S2 are balanced (no duplication), cleaner training
3. **Long-tail metrics**: Reweighting ensures metrics reflect actual performance
4. **Group-wise optimization**: Optimizes head/tail performance separately
5. **No data leakage**: S1/S2 are from test set, separate from training

## Performance Tips

- Use `selective` gating checkpoint for better initialization
- Increase `M` (iterations) for better convergence
- Adjust `alpha_steps` for finer-grained optimization
- Use `worst` objective for better tail performance
- Monitor both reweighted and standard metrics
