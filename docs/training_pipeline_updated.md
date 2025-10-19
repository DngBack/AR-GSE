# AR-GSE Training Pipeline - Updated Workflow

## Overview
This document describes the complete training pipeline with the new data split strategy.

## Data Split Strategy

### Training Set (from CIFAR-100 train, 50,000 samples)
- **Original Train**: 10,847 samples (long-tail, IF=100)
  - Split into:
    - **Expert Split**: 9,719 samples (90%, long-tail, IF≈112.5) - for training experts
    - **Gating Split**: 1,128 samples (10%, long-tail, IF≈50.0) - for pretrain gating

### Test Set (from CIFAR-100 test, 10,000 samples)
All balanced splits:
- **Test**: 8,000 samples (80 per class) - final evaluation
- **Val**: 1,000 samples (10 per class) - validation during expert training
- **TuneV**: 1,000 samples (10 per class) - for selective gating training (S1)

## Training Pipeline

### Step 1: Create Data Splits
```bash
python create_splits_fixed.py --split-train --visualize
```

**Creates:**
- `data/cifar100_lt_if100_splits_fixed/`
  - `train_indices.json` (10,847 samples)
  - `expert_indices.json` (9,719 samples) ← for expert training
  - `gating_indices.json` (1,128 samples) ← for gating pretrain
  - `test_indices.json` (8,000 samples)
  - `val_indices.json` (1,000 samples) ← for validation
  - `tunev_indices.json` (1,000 samples) ← for selective training
  - `class_weights.json` ← for reweighted metrics

**Visualizations:**
- `outputs/visualizations/splits_distribution_complete.png`
- `outputs/visualizations/expert_vs_gating_comparison.png`

---

### Step 2: Train Expert Models
```bash
# Train all experts (default: uses expert split)
python train_experts.py

# Or train specific expert
python train_experts.py --expert ce
python train_experts.py --expert logitadjust
python train_experts.py --expert balsoftmax

# Use full train instead of expert split (optional)
python train_experts.py --use-full-train
```

**Training Details:**
- **Train on**: Expert split (9,719 samples, long-tail)
- **Validate on**: Val split (1,000 samples, balanced) with reweighting
- **Validation metrics**:
  - Standard accuracy (on balanced val)
  - Reweighted accuracy (simulates long-tail performance)
  - Group-wise accuracy (head/tail)

**Saves:**
- Checkpoints: `checkpoints/experts/cifar100_lt_if100/`
  - `best_ce_baseline.pth`
  - `best_logitadjust_baseline.pth`
  - `best_balsoftmax_baseline.pth`
  - `final_calibrated_*.pth` (with temperature scaling)

**Exports Logits:**
- `outputs/logits/cifar100_lt_if100/{expert_name}/`
  - `train_logits.pt` (10,847)
  - `expert_logits.pt` (9,719) ← for analysis
  - `gating_logits.pt` (1,128) ← for gating pretrain
  - `val_logits.pt` (1,000) ← for validation
  - `test_logits.pt` (8,000) ← for final eval
  - `tunev_logits.pt` (1,000) ← for selective training

---

### Step 3: Test Expert Data (Optional)
```bash
python test_expert_dataloader.py
```

Verifies:
- Expert split loading
- Class distributions
- Imbalance factors

---

### Step 4: Test Gating Data (Optional)
```bash
python test_gating_data.py
```

Verifies:
- All logits files exist
- Correct shapes and sizes
- Data loading for both pretrain and selective modes

---

### Step 5: Train Gating Network - Pretrain Mode
```bash
python train_gating.py --mode pretrain
```

**Training Details:**
- **Train on**: Gating split (1,128 samples, long-tail, from train set)
- **Purpose**: Warm up gating network to learn expert mixing
- **Loss**: Mixture cross-entropy with diversity regularization
- **Epochs**: 25 (configurable)

**Uses:**
- Logits: `outputs/logits/cifar100_lt_if100/{expert}/gating_logits.pt`
- Splits: `data/cifar100_lt_if100_splits_fixed/gating_indices.json`

**Saves:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_pretrain.ckpt`

---

### Step 6: Train Gating Network - Selective Mode
```bash
python train_gating.py --mode selective
```

**Training Details:**
- **Train on (S1)**: TuneV split (1,000 samples, balanced, from test set)
- **Validate on (S2)**: Val split (1,000 samples, balanced, from test set)
- **Purpose**: Learn selective prediction with coverage targets
- **Method**: Alternating optimization (Stage A + Stage B cycles)
  - Stage A: Warm-up with mixture CE
  - Stage B: Alternating gating/α updates + μ grid search

**Uses:**
- Logits: 
  - `outputs/logits/cifar100_lt_if100/{expert}/tunev_logits.pt` (S1)
  - `outputs/logits/cifar100_lt_if100/{expert}/val_logits.pt` (S2)
- Splits:
  - `data/cifar100_lt_if100_splits_fixed/tunev_indices.json`
  - `data/cifar100_lt_if100_splits_fixed/val_indices.json`

**Saves:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt`
- `checkpoints/gating_pretrained/cifar100_lt_if100/selective_training_logs.json`

**Learns:**
- Gating weights (expert selection)
- α parameters (group-wise acceptance)
- μ parameters (threshold adjustment)
- Per-group thresholds (via Pinball loss)

---

### Step 7: Evaluate AR-GSE
```bash
python evaluate_argse.py
```

**Evaluation:**
- Uses **Test split** (8,000 samples, balanced)
- Computes metrics:
  - Overall accuracy
  - Group-wise accuracy (head/tail)
  - Coverage
  - Selective accuracy (on accepted samples)
  - AURC (Area Under Risk-Coverage curve)

**Uses:**
- Expert logits: `outputs/logits/cifar100_lt_if100/{expert}/test_logits.pt`
- Gating checkpoint: `checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt`
- Class weights: `data/cifar100_lt_if100_splits_fixed/class_weights.json`

---

## Key Differences from Old Pipeline

### Old Pipeline
- ❌ Duplicated test samples for imbalanced validation
- ❌ Used full train for both experts and gating
- ❌ No clear separation between expert/gating training data
- ❌ Uniform validation without reweighting

### New Pipeline ✅
- ✅ Balanced test/val/tunev (no duplication)
- ✅ Expert split (90%) for expert training
- ✅ Gating split (10%) for gating pretrain
- ✅ TuneV split (balanced) for selective training
- ✅ Reweighted validation metrics
- ✅ Clear data separation prevents overfitting

---

## File Structure

```
AR-GSE/
├── data/
│   └── cifar100_lt_if100_splits_fixed/
│       ├── train_indices.json (10,847)
│       ├── expert_indices.json (9,719) ← expert training
│       ├── gating_indices.json (1,128) ← gating pretrain
│       ├── test_indices.json (8,000)
│       ├── val_indices.json (1,000) ← validation
│       ├── tunev_indices.json (1,000) ← selective S1
│       ├── expert_class_counts.json
│       ├── gating_class_counts.json
│       └── class_weights.json ← for reweighting
│
├── outputs/
│   ├── logits/cifar100_lt_if100/
│   │   ├── ce_baseline/
│   │   │   ├── expert_logits.pt
│   │   │   ├── gating_logits.pt ← pretrain
│   │   │   ├── val_logits.pt
│   │   │   ├── test_logits.pt
│   │   │   └── tunev_logits.pt ← selective
│   │   ├── logitadjust_baseline/...
│   │   └── balsoftmax_baseline/...
│   │
│   └── visualizations/
│       ├── splits_distribution_complete.png
│       └── expert_vs_gating_comparison.png
│
└── checkpoints/
    ├── experts/cifar100_lt_if100/
    │   ├── best_ce_baseline.pth
    │   ├── best_logitadjust_baseline.pth
    │   └── best_balsoftmax_baseline.pth
    │
    └── gating_pretrained/cifar100_lt_if100/
        ├── gating_pretrain.ckpt
        ├── gating_selective.ckpt
        └── selective_training_logs.json
```

---

## Quick Start

```bash
# 1. Create splits
python create_splits_fixed.py --split-train --visualize

# 2. Train experts (uses expert split, validates with reweighting)
python train_experts.py

# 3. Test data loading (optional)
python test_gating_data.py

# 4. Train gating - pretrain (uses gating split)
python train_gating.py --mode pretrain

# 5. Train gating - selective (uses tunev + val)
python train_gating.py --mode selective

# 6. Evaluate (uses test split)
python evaluate_argse.py
```

---

## Troubleshooting

### Missing logits files
```bash
# Re-train experts to export logits
python train_experts.py
```

### Wrong split sizes
```bash
# Re-create splits
python create_splits_fixed.py --split-train --visualize
```

### Check data integrity
```bash
# Test expert data
python test_expert_dataloader.py

# Test gating data
python test_gating_data.py
```

---

## Notes

1. **Reweighting**: Val metrics are reweighted using `class_weights.json` to reflect long-tail performance on balanced data.

2. **No Data Leakage**: 
   - Expert/Gating splits are disjoint (from train set)
   - Test/Val/TuneV are disjoint (from test set)
   - No overlap between any splits

3. **Imbalance Preservation**: 
   - Expert split maintains long-tail distribution (~IF=112.5)
   - Gating split maintains long-tail distribution (~IF=50.0)
   - Both preserve relative class frequencies

4. **Balanced Evaluation**:
   - Val/Test/TuneV are perfectly balanced
   - Reweighting at metric computation time
   - More reliable evaluation than duplicated imbalanced sets
