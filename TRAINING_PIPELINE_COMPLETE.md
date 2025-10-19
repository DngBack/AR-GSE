# Complete AR-GSE Training & Evaluation Pipeline - FIXED VERSION

## 🎯 Overview

This guide covers the COMPLETE pipeline with ALL fixes applied:
1. ✅ Data splits (9:1 expert/gating)
2. ✅ Reweighted metrics (class_weights.json)
3. ✅ Expert training (expert split + reweighted val)
4. ✅ Gating training (pretrain + selective modes)
5. ✅ Plugin training (per-group thresholds)
6. ✅ **AURC evaluation (per-group thresholds - FIXED!)**

## 📋 Full Pipeline Commands

### Step 1: Create Data Splits (if not done)
```bash
python create_splits_fixed.py --split-train --visualize
```

**Output:**
- `data/cifar100_lt_if100_splits_fixed/`
  - expert_indices.json (9,719 samples)
  - gating_indices.json (1,128 samples)
  - tunev_indices.json (1,000 samples)
  - val_indices.json (1,000 samples)
  - test_indices.json (8,000 samples)
  - class_weights.json (for reweighting)

### Step 2: Train Experts
```bash
# Train all 3 experts
python train_experts.py

# Or train individually
python train_experts.py --expert ce_baseline
python train_experts.py --expert logitadjust_baseline
python train_experts.py --expert balsoftmax_baseline
```

**Key features:**
- Uses expert split (9,719 samples)
- Validates on balanced val split (1,000 samples) WITH reweighting
- Exports logits for gating/plugin training

**Output:**
- `checkpoints/experts/cifar100_lt_if100/{expert}/best_model.ckpt`
- `outputs/logits/cifar100_lt_if100/{expert}/expert_logits.pt`
- `outputs/logits/cifar100_lt_if100/{expert}/tunev_logits.pt`
- `outputs/logits/cifar100_lt_if100/{expert}/val_logits.pt`
- `outputs/logits/cifar100_lt_if100/{expert}/test_logits.pt`

### Step 3: Train Gating Network

#### 3a. Pretrain Mode (on gating split)
```bash
python train_gating.py --mode pretrain
```

**Features:**
- Uses gating split (1,128 samples) - separate from expert training
- Pretrains gating to recognize expert expertise
- No reweighting (training set already long-tail)

**Output:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_pretrained.ckpt`

#### 3b. Selective Mode (on tunev + val)
```bash
python train_gating.py --mode selective
```

**Features:**
- Uses tunev + val (2,000 balanced samples)
- Fine-tunes for selective prediction
- WITH reweighting (balanced → long-tail metrics)

**Output:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt`

### Step 4: Plugin Training (Optimize α, μ, thresholds)

#### 4a. Worst-Group Objective (Recommended for Long-Tail)
```bash
python run_improved_eg_outer.py
```

**Features:**
- Uses tunev (S1) + val (S2) for optimization
- Optimizes per-group thresholds (t_head, t_tail)
- Minimizes worst-group error
- WITH reweighting

**Algorithm:**
- EG-outer iterations: 30
- Inner plugin iterations: 10
- Per-group threshold optimization
- Anti-collapse β with momentum

**Output:**
- `checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt`

Contains:
```python
{
    'alpha': [1.0, 1.0],              # May not change if fixed
    'mu': [-0.39, 0.39],              # Optimized
    't_group': [-0.427, -0.034],      # Per-group thresholds
    'per_group_threshold': True,       # Flag
    'gating_net_state_dict': {...},   # Gating weights
    'best_score': 0.2105,              # Training error
}
```

#### 4b. Balanced Objective (Alternative)
```bash
python -m src.train.gse_balanced_plugin
```

Edit `CONFIG` to set:
```python
'objective': 'balanced',
'use_eg_outer': False,
```

### Step 5: AURC Evaluation ✨ **FIXED VERSION**

```bash
python -m src.train.eval_gse_plugin
```

**What's FIXED:**
- ✅ Uses per-group thresholds from checkpoint
- ✅ Consistent with plugin training mechanism
- ✅ Proper coverage range [0, 1]
- ✅ Accurate AURC values

**Algorithm:**
```python
1. Load checkpoint with t_group = [-0.427, -0.034]
2. For each scale factor s in [0, 0.2, ..., 5.0]:
   - Compute scaled thresholds: t_k = t_group[k] * s
   - For each sample i:
     - Get group: g_i = class_to_group[y_i]
     - Accept if: raw_margin_i > t_{g_i}
   - Compute coverage and risk
3. Integrate to get AURC
```

**Expected Output:**
```
✅ Loaded optimal parameters:
   α* = [1.0, 1.0]
   μ* = [-0.39, 0.39]
   t_group* = [-0.427, -0.034]
   → Using PER-GROUP thresholds (consistent with plugin training)

🔍 DEBUGGING: Per-group threshold behavior
   Base thresholds: [-0.427, -0.034]
   scale=0.0 → t_group=[0.0, 0.0] → coverage=0.023
   scale=1.0 → t_group=[-0.427, -0.034] → coverage=0.45
   scale=2.0 → t_group=[-0.854, -0.068] → coverage=0.85

BALANCED AURC (0-1):     0.20-0.25  ← Much better!
WORST AURC (0-1):        0.30-0.40  ← Much better!
```

**Output Files:**
- `results_worst_eg_improved/cifar100_lt_if100/aurc_curves.png`
- `results_worst_eg_improved/cifar100_lt_if100/aurc_detailed_results.csv`
- `results_worst_eg_improved/cifar100_lt_if100/aurc_summary.json`

## 🔍 Verification Checklist

### After Expert Training:
```bash
python test_expert_dataloader.py
```
- [ ] Expert uses 9,719 samples
- [ ] Validation uses 1,000 samples (balanced)
- [ ] Reweighted accuracy shown

### After Gating Training:
```bash
python test_gating_data.py
```
- [ ] Pretrain uses 1,128 gating samples
- [ ] Selective uses 2,000 tunev+val samples
- [ ] Logits loaded correctly

### After Plugin Training:
```bash
python verify_plugin_checkpoint.py
```
- [ ] α, μ values reasonable
- [ ] t_group contains per-group thresholds
- [ ] per_group_threshold = True
- [ ] best_score matches expected performance

### After AURC Evaluation:
- [ ] Coverage reaches ~1.0 (or close)
- [ ] AURC values reasonable (< 0.5)
- [ ] Per-group mode indicated in output
- [ ] RC curves smooth and monotonic

## 📊 Expected Performance

### Before Fixes (OLD - WRONG):
```
Plugin training: error = 0.21 (per-group thresholds)
AURC eval:       AURC = 0.35/0.54 (global threshold)
→ MISMATCH! Parameters evaluated incorrectly
```

### After Fixes (NEW - CORRECT):
```
Plugin training: error = 0.21 (per-group thresholds)
AURC eval:       AURC = 0.20-0.25/0.30-0.40 (per-group thresholds)
→ CONSISTENT! Matches training performance
```

## 🐛 Common Issues & Solutions

### Issue 1: Coverage doesn't reach 1.0
**Cause:** Very uncertain samples with margins < minimum threshold
**Solution:** Normal behavior, coverage ~0.95-0.99 is acceptable

### Issue 2: AURC still high after fix
**Check:**
1. Is per-group mode activated? Look for "Using PER-GROUP thresholds"
2. Are t_group values loaded? Check debug output
3. Is checkpoint from correct training run?

### Issue 3: Per-group thresholds not found
**Cause:** Checkpoint from old training without per-group support
**Solution:** Re-run plugin training with `run_improved_eg_outer.py`

### Issue 4: Dimension mismatch in gating
**Cause:** Gating checkpoint has different feature dimensions
**Solution:** Re-train gating OR let evaluation use random weights (suboptimal)

## 📝 Key Takeaways

1. **Consistency is critical**: Training and evaluation must use same mechanism
2. **Per-group thresholds**: More flexible than global, better for long-tail
3. **Reweighting**: Essential for proper long-tail evaluation on balanced data
4. **Data separation**: Expert/gating splits from train, tunev/val/test from test set
5. **Verification**: Always run test scripts to catch issues early

## 🔗 Related Documentation

- `TRAINING_QUICK_REF.md` - Quick command reference
- `docs/threshold_mismatch_analysis.md` - Detailed problem analysis
- `docs/aurc_bug_fix.md` - Previous bug fix documentation
- `docs/reweighting_explained.md` - Reweighting methodology
- `EVALUATION_SUMMARY.md` - High-level summary

## 🎯 Next Steps

After successful evaluation:
1. Generate paper figures and tables
2. Compare with baselines
3. Ablation studies (different α, μ, objectives)
4. Write up results

Good luck! 🚀
