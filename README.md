# 🚀 AR-GSE Training & Evaluation - Fixed Workflow

## Overview

This guide covers the **complete training and evaluation workflow** for AR-GSE with **proper long-tail evaluation** using:
- ✅ Balanced test splits (8:1:1 ratio, no duplication)
- ✅ Reweighted metrics (accounts for training class imbalance)
- ✅ No data leakage
- ✅ Reusable expert checkpoints

---

## 📋 Prerequisites

```bash
# Activate environment
conda activate argse

# Navigate to project
cd /home/duong.xuan.bach/AR-GSE
```

---

## 🔄 Complete Training Pipeline

### Step 1: Create Fixed Data Splits (✅ DONE)

```bash
python3 create_splits_fixed.py
```

**Output:**
```
data/cifar100_lt_if100_splits_fixed/
├── train_indices.json (10,847 samples - long-tail)
├── test_indices.json (8,000 samples - balanced)
├── val_indices.json (1,000 samples - balanced)
├── tunev_indices.json (1,000 samples - balanced)
├── class_weights.json (for reweighting)
├── train_class_counts.json
└── reweight_info.json
```

**Key Features:**
- Train: Long-tail CIFAR-100-LT (IF=100)
- Test/Val/TuneV: Split 8:1:1 from balanced CIFAR-100 test set
- No overlap, no duplication
- Class weights computed from train distribution

---

### Step 2: Train Expert Models (Optional - if needed)

**Skip this if you already have expert checkpoints!**

#### Option A: Train manually (original method)

```bash
# Train each expert
python3 train_experts.py --expert ce_baseline --epochs 200
python3 train_experts.py --expert balsoftmax_baseline --epochs 200
python3 train_experts.py --expert logitadjust_baseline --epochs 200
```

**Time:** ~4 hours total

#### Option B: Train via recompute_logits.py (NEW!)

```bash
python3 recompute_logits.py \
    --train-experts \
    --train-epochs 200 \
    --train-lr 0.1 \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline
```

**Time:** ~4-5 hours (trains + computes logits automatically)

**What it does:**
- Trains each expert from scratch
- Uses appropriate loss per expert (CE, BalSoftmax, LogitAdjust)
- Saves checkpoints automatically
- **Bonus:** Computes logits immediately after training

**Output:**
```
checkpoints/experts/cifar100_lt_if100/
├── best_ce_baseline.pth
├── best_balsoftmax_baseline.pth
├── best_logitadjust_baseline.pth
└── ... (optimizer states, etc.)

outputs/logits_fixed/
├── ce_baseline/
├── balsoftmax_baseline/
└── logitadjust_baseline/
```

---

### Step 3: Recompute Logits with New Splits

**This step is REQUIRED if you're using new data splits AND didn't use Option B above!**

```bash
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline \
    --checkpoint-prefix best_
```

**What it does:**
- Loads pre-trained expert weights
- Computes logits for NEW splits (tunev, val, test)
- Saves in `.npz` format

**Output:**
```
outputs/logits_fixed/
├── ce_baseline/
│   ├── tunev_logits.npz (1,000 samples)
│   ├── val_logits.npz (1,000 samples)
│   └── test_logits.npz (8,000 samples)
├── balsoftmax_baseline/
│   └── ... (same structure)
└── logitadjust_baseline/
    └── ... (same structure)
```

**Verify:**
```bash
python3 verify_logits.py
```

---

### Step 4: Train Gating Network - Pretrain Mode

```bash
python3 train_gating.py --mode pretrain
```

**What it does:**
- Warm up gating network
- Learn to mix expert predictions
- Uses `tunev` split for training

**Key Configuration:**
- Logits from: `outputs/logits_fixed/`
- Splits from: `data/cifar100_lt_if100_splits_fixed/`
- Batch size: 2
- Epochs: 25
- Learning rate: 8e-4

**Output:**
```
checkpoints/gating_pretrained/cifar100_lt_if100/
└── gating_pretrained.ckpt
```

**Expected Loss:**
- Epoch 1: ~4.6 (random guess for 100 classes)
- Final: ~3.5-3.8 (converged)

---

### Step 5: Train Gating Network - Selective Mode

```bash
python3 train_gating.py --mode selective
```

**What it does:**
- Learn optimal expert selection thresholds
- Optimize for worst-group performance
- Uses `tunev` + `val` splits

**Key Features:**
- Stage A: Warm-up (5 epochs)
- Stage B: Alternating optimization (6 cycles × 3 epochs)
- Learns α (scale), μ (shift), t (thresholds) per group

**Output:**
```
checkpoints/gating_pretrained/cifar100_lt_if100/
├── gating_selective.ckpt
└── selective_training_logs.json
```

**Expected Metrics:**
- Coverage: 0.2-0.5 (selective rejection)
- Worst-group error: Improving over cycles

---

### Step 6: Train Full AR-GSE Ensemble

```bash
python3 run_improved_eg_outer.py
```

**What it does:**
- Loads pretrained gating
- Runs EG-Outer optimization for α, μ
- Balances head/tail performance

**Key Configuration:**
- EG outer iterations: 30
- EG step size: 0.2
- Inner iterations: 10
- Anti-collapse mechanisms enabled

**Output:**
```
checkpoints/argse_worst_eg_improved/cifar100_lt_if100/
└── gse_balanced_plugin.ckpt
```

---

### Step 7: Evaluate with Reweighted Metrics

```bash
python -m src.train.eval_gse_plugin
```

**What it does:**
- Loads trained AR-GSE model
- Evaluates on `test` split (8,000 balanced samples)
- Computes AURC with **reweighted metrics**
- Accounts for training class imbalance

**Key Features:**
- Threshold optimization on: `tunev` + `val` (2,000 samples)
- Final evaluation on: `test` (8,000 samples)
- Reweighting: Tail classes get higher weight
- Metrics: standard, balanced, worst-group

**Output:**
```
results_worst_eg_improved/cifar100_lt_if100/
├── aurc_detailed_results.csv
├── aurc_summary.json
└── aurc_curves.png
```

**Example Results:**
```
============================================================
FINAL AURC RESULTS (REWEIGHTED FOR LONG-TAIL)
============================================================

📊 AURC (Full Range 0-1):
   •     STANDARD AURC: 0.123456
   •     BALANCED AURC: 0.145678
   •        WORST AURC: 0.167890

📊 AURC (Practical Range 0.2-1):
   •     STANDARD AURC: 0.098765
   •     BALANCED AURC: 0.112345
   •        WORST AURC: 0.134567

✅ Metrics reweighted by train class distribution
📝 Lower AURC is better
```

---

## 🔧 Troubleshooting

### Issue: "Missing logits for expert"

**Solution:** Run `recompute_logits.py` first:
```bash
python3 recompute_logits.py --experts ce_baseline balsoftmax_baseline logitadjust_baseline
```

### Issue: "Size mismatch (1000 vs 1446)"

**Cause:** Using old logits with new data splits

**Solution:** 
1. Ensure `train_gating.py` uses default paths:
   - `--logits-dir ./outputs/logits_fixed/`
   - `--splits-dir ./data/cifar100_lt_if100_splits_fixed`
2. Recompute logits if needed

### Issue: "Gating loss not decreasing (~9.9)"

**Status:** ✅ FIXED

**What was wrong:** 
- Double log in loss computation
- `nll_loss` expects log probabilities, but we passed log(log(probs))

**Fixed in:** `src/train/train_gating_only.py` line 818

### Issue: "FileNotFoundError: tuneV_indices.json"

**Cause:** Case sensitivity - file is `tunev_indices.json` (lowercase)

**Status:** ✅ FIXED - All scripts now use lowercase

---

## 📊 Data Split Summary

| Split | Source | Size | Distribution | Purpose |
|-------|--------|------|--------------|---------|
| **train** | CIFAR-100-LT train | 10,847 | Long-tail (IF=100) | Train experts |
| **tunev** | CIFAR-100 test | 1,000 | Balanced | Gating training |
| **val** | CIFAR-100 test | 1,000 | Balanced | Threshold optimization |
| **test** | CIFAR-100 test | 8,000 | Balanced | Final evaluation |

**Key Properties:**
- No overlap between splits
- No duplication
- Test/val/tunev are balanced (10 samples per class)
- Reweighting applied at metrics computation time

---

## 🎯 Key Differences from Original

| Aspect | Original | Fixed |
|--------|----------|-------|
| **Test Data** | Long-tail (duplicated) | Balanced (8:1:1 split) |
| **Data Leakage** | Yes (duplication) | No (proper splits) |
| **Metrics** | Uniform weights | Reweighted by train freq |
| **Tunev Source** | Train set (long-tail) | Test set (balanced) |
| **Logits Format** | `.pt` only | `.npz` (with fallback to `.pt`) |
| **File Names** | Mixed case (tuneV) | Lowercase (tunev) |

---

## 📂 Directory Structure

```
AR-GSE/
├── data/
│   ├── cifar100_lt_if100_splits_fixed/    # NEW: Fixed splits
│   │   ├── train_indices.json
│   │   ├── test_indices.json
│   │   ├── val_indices.json
│   │   ├── tunev_indices.json
│   │   └── class_weights.json
│   └── cifar100_lt_if100_splits/          # OLD: Original splits
│
├── outputs/
│   ├── logits_fixed/                      # NEW: Recomputed logits
│   │   ├── ce_baseline/
│   │   ├── balsoftmax_baseline/
│   │   └── logitadjust_baseline/
│   └── logits/                            # OLD: Original logits
│
├── checkpoints/
│   ├── experts/cifar100_lt_if100/         # Pre-trained experts (reusable)
│   ├── gating_pretrained/cifar100_lt_if100/
│   └── argse_worst_eg_improved/
│
└── results_worst_eg_improved/cifar100_lt_if100/
    ├── aurc_detailed_results.csv
    ├── aurc_summary.json
    └── aurc_curves.png
```

---

## 🚀 Quick Start (Full Pipeline)

```bash
# 1. Setup
conda activate argse
cd /home/duong.xuan.bach/AR-GSE

# 2. Create splits (if not done)
python3 create_splits_fixed.py

# 3. Recompute logits from existing expert weights
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline \
    --checkpoint-prefix best_

# 4. Verify logits
python3 verify_logits.py

# 5. Train gating (pretrain)
python3 train_gating.py --mode pretrain

# 6. Train gating (selective)
python3 train_gating.py --mode selective

# 7. Train full AR-GSE
python3 run_improved_eg_outer.py

# 8. Evaluate with reweighted metrics
python -m src.train.eval_gse_plugin
```

**Total time:** ~2-3 hours (without retraining experts)



## 🎉 What's Fixed?

1. ✅ **No data leakage** - Proper 8:1:1 split without duplication
2. ✅ **Balanced test set** - All classes equally represented
3. ✅ **Reweighted metrics** - Accounts for training imbalance
4. ✅ **Reusable experts** - No need to retrain from scratch
5. ✅ **Consistent naming** - All lowercase file names
6. ✅ **Format support** - Both .npz and .pt logits
7. ✅ **Gating loss fix** - Proper cross-entropy computation
8. ✅ **Comprehensive docs** - Clear workflow and troubleshooting

**Result:** Proper long-tail evaluation that fairly assesses performance across all classes! 🚀
