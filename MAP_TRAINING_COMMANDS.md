# MAP Training Pipeline - Commands Reference

## 🚀 Quick Start (Full Pipeline)

```bash
chmod +x run_map_pipeline.sh
./run_map_pipeline.sh
```

---

## 📋 Manual Step-by-Step Commands

### Step 1: Train Experts (90% of train)
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 train_experts.py
```

**Output:**
- `checkpoints/experts/cifar100_lt_if100/ce_baseline/best_model.pth`
- `checkpoints/experts/cifar100_lt_if100/logitadjust_baseline/best_model.pth`
- `checkpoints/experts/cifar100_lt_if100/balsoftmax_baseline/best_model.pth`

**Time:** ~2-3 hours (3 experts × 200 epochs)

---

### Step 2: Compute Expert Logits
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 recompute_logits.py
```

**Output:**
- `outputs/logits/cifar100_lt_if100/{expert}/{split}_logits.pt`

**Splits:** expert, gating, val, tunev, test

**Time:** ~5-10 minutes

---

### Step 3: Train Gating Network (10% of train)
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing dense \
    --epochs 100 \
    --lr 0.001 \
    --batch-size 128
```

**Output:**
- `checkpoints/gating_map/cifar100_lt_if100/best_gating.pth`
- Logs: Train NLL, Val NLL, Balanced Acc

**Time:** ~30 minutes (100 epochs)

---

### Step 4: Train MAP Plugin (Grid Search)
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_simple.py --objective balanced
```

**Arguments:**
- `--objective balanced`: Optimize mean of group errors
- `--objective worst`: Optimize worst-group error
- `--eg_outer`: Use EG-outer for worst-group (update β weights)
- `--no_reweight`: Disable test set reweighting

**Grid Search (defined in train_map_simple.py):**
- Threshold θ: 17 values from 0.1 to 0.9
- Gamma γ: [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
- Total: 17 × 7 = 119 combinations

**Output:**
- `checkpoints/map_simple/cifar100_lt_if100/best_selector.pth`
- `results/map_simple/cifar100_lt_if100/optimization_log.json`
- `results/map_simple/cifar100_lt_if100/rc_curve.csv`

**Time:** ~10-20 minutes

---

### Step 5: Evaluate MAP Plugin
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/eval_map_plugin.py --visualize
```

**Arguments:**
- `--visualize`: Generate RC curves and other plots
- `--no_reweight`: Disable test set reweighting

**Output:**
- `results/map_plugin/cifar100_lt_if100/metrics.json`
- `results/map_plugin/cifar100_lt_if100/rc_curve.csv`
- `results/map_plugin/cifar100_lt_if100/plots/rc_curves.png`

**Time:** ~5 minutes

---

## 🔧 Advanced Options

### Train MAP with Worst-Group Objective
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_simple.py --objective worst --eg_outer
```

### Customize Grid Search Ranges

Edit `train_map_simple.py` line 51-52:
```python
'threshold_grid': list(np.linspace(0.1, 0.9, 17)),  # Change range/density
'gamma_grid': [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],  # Add more values
```

### Train with Different Gating Epochs
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing dense \
    --epochs 200 \
    --lr 0.0005 \
    --batch-size 64
```

---

## 📊 Expected Results

### After Training Experts (Step 1)
```
CE Baseline:      ~38-40% test accuracy
LogitAdjust:      ~46-48% test accuracy
BalSoftmax:       ~46-48% test accuracy
```

### After Training Gating (Step 3)
```
Train NLL:        ~0.005-0.007
Val NLL:          ~2.01-2.02 (correct due to balanced val set)
Val Balanced Acc: ~49-50%
Effective Experts: ~2.98-3.00 (dense routing)
```

### After MAP Training (Step 4)
```
Best config found (θ, γ)
Val metrics:
  - Balanced error: <50%
  - Coverage: ~60-80%
  - Head error: <30%
  - Tail error: <70%
```

### After Evaluation (Step 5)
```
Test metrics (reweighted to match train distribution):
  - Selective error: <45%
  - Coverage: ~65-75%
  - AURC: <0.25
  - Group-wise fairness metrics
```

---

## 🐛 Troubleshooting

### Issue: "Logits not found"
**Solution:** Run Step 2 first: `python3 recompute_logits.py`

### Issue: "Gating checkpoint not found"
**Solution:** Run Step 3 first: `python3 src/train/train_gating_map.py --routing dense --epochs 100`

### Issue: "Train loss is negative"
**Solution:** Already fixed! Entropy regularizer now returns positive values

### Issue: "Val NLL is very high (~2.02)"
**Solution:** This is correct! Val set is balanced, experts perform poorly on it

### Issue: Grid search takes too long
**Solution:** Reduce grid density in `train_map_simple.py`:
```python
'threshold_grid': list(np.linspace(0.1, 0.9, 9)),  # 17 → 9
'gamma_grid': [0.0, 0.5, 1.0, 2.0],  # 7 → 4
```

---

## 📁 Directory Structure

```
AR-GSE/
├── checkpoints/
│   ├── experts/cifar100_lt_if100/        # Step 1 output
│   ├── gating_map/cifar100_lt_if100/     # Step 3 output
│   └── map_simple/cifar100_lt_if100/     # Step 4 output
├── outputs/
│   └── logits/cifar100_lt_if100/         # Step 2 output
├── results/
│   ├── map_simple/cifar100_lt_if100/     # Step 4 results
│   └── map_plugin/cifar100_lt_if100/     # Step 5 results
└── data/
    └── cifar100_lt_if100_splits_fixed/   # Data splits (expert/gating)
```

---

## 🎯 Next Steps After Pipeline

1. **Analyze Results:**
   ```bash
   cat results/map_simple/cifar100_lt_if100/optimization_log.json
   ```

2. **Compare with Baselines:**
   - Train worst-group version: `python3 train_map_simple.py --objective worst --eg_outer`
   - Compare balanced vs worst-group objectives

3. **Generate Paper Plots:**
   ```bash
   python3 generate_final_plots.py
   ```

4. **Run Full Evaluation Suite:**
   ```bash
   python3 evaluate_argse.py --save-results --verbose
   ```
