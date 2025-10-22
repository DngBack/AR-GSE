# Quick Start: Gating Network Training

## 🎯 Mục tiêu
Huấn luyện Gating Network để học trọng số mixture tối ưu từ các experts đã train.

## 📋 Prerequisites

✅ Đã train experts (ce, logitadjust, balsoftmax)  
✅ Logits đã được export tại `outputs/logits/cifar100_lt_if100/`  
✅ Splits đã được tạo tại `data/cifar100_lt_if100_splits_fixed/`

## 🚀 Training Commands

### 1. Basic Training (Dense Routing)
```bash
cd /home/duong.xuan.bach/AR-GSE
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py --routing dense
```

**Expected output:**
- Training ~100 epochs
- Validation every 5 epochs
- Best model saved based on balanced accuracy

### 2. Sparse Routing (Top-K)
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing top_k \
    --top_k 2 \
    --lambda_lb 1e-2
```

**Benefits của Top-K:**
- Faster inference (chỉ dùng K=2 experts)
- Load-balancing loss tránh collapse
- Noise injection for exploration

### 3. Custom Hyperparameters
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py \
    --routing dense \
    --epochs 150 \
    --batch_size 64 \
    --lr 5e-4 \
    --lambda_lb 1e-2 \
    --lambda_h 0.02
```

## 📊 Monitor Training

### Key Metrics to Watch:

**Loss components:**
```
NLL: 2-4 (mixture negative log-likelihood)
Load-balancing: 0.01-0.02 (if using top-k)
Entropy: -0.8 to -1.2 (negative for maximization)
```

**Gating quality:**
```
Mixture Acc: Should > max(individual expert accs)
Effective Experts: Close to E (e.g., 2.8/3.0)
Gating Entropy: 0.8-1.0 (diversity)
Load Std: < 0.1 (balance)
```

**Group-wise performance:**
```
Head Acc: 0.70-0.80
Tail Acc: 0.50-0.65
Balanced Acc: (Head + Tail) / 2
```

## 🔍 Check Results

### Training History
```bash
cat results/gating_map/cifar100_lt_if100/training_history.json
```

### Load Best Model
```python
import torch
from src.models.gating_network_map import GatingNetwork

# Initialize
gating = GatingNetwork(num_experts=3, num_classes=100)

# Load checkpoint
checkpoint = torch.load('checkpoints/gating_map/cifar100_lt_if100/best_gating.pth')
gating.load_state_dict(checkpoint['model_state_dict'])

# Inference
posteriors = ...  # [B, E, C] từ experts
weights, _ = gating(posteriors)
mixture = gating.get_mixture_posterior(posteriors, weights)
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Always set PYTHONPATH:
```bash
export PYTHONPATH=/home/duong.xuan.bach/AR-GSE:$PYTHONPATH
```

### Issue: Logits not found
**Check:**
```bash
ls outputs/logits/cifar100_lt_if100/ce_baseline/
# Should see: gating_logits.pt, val_logits.pt, etc.
```

**Fix:** Re-run expert training:
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_expert.py
```

### Issue: Gating collapse (effective_experts < 1.5)
**Solutions:**
1. Increase entropy regularization: `--lambda_h 0.05`
2. Increase load-balancing: `--lambda_lb 5e-2`
3. Decrease learning rate: `--lr 5e-4`

### Issue: Low mixture accuracy
**Solutions:**
1. Train longer: `--epochs 200`
2. Simplify architecture: modify CONFIG in train_gating_map.py
   ```python
   'hidden_dims': [128],  # Instead of [256, 128]
   ```
3. Check expert calibration (run TemperatureScaling again)

## 📈 Expected Timeline

**Dense routing:**
- Training time: ~5-10 minutes (100 epochs, GPU)
- Best model typically around epoch 60-80

**Top-K routing:**
- Training time: ~5-10 minutes
- May need more epochs (~150) for stability

## 🔄 Integration với MAP Pipeline

Sau khi train gating:

1. **Extract mixture posteriors:**
```python
# For S1/S2 optimization
mixture_posteriors_s1 = []
mixture_posteriors_s2 = []

for split in ['tunev', 'val']:
    logits = load_expert_logits(split)
    posteriors = torch.softmax(logits, dim=-1)
    weights, _ = gating(posteriors)
    mixture = gating.get_mixture_posterior(posteriors, weights)
    # Save for MAP plugin training
```

2. **Compute U(x) for MAP margin:**
```python
from src.models.gating_network_map import compute_uncertainty_for_map

U = compute_uncertainty_for_map(
    posteriors, weights, mixture,
    coeffs={'a': 1.0, 'b': 1.0, 'd': 1.0}
)
# Use in: m_MAP = m_L2R - gamma * U
```

3. **Run MAP plugin optimization:**
```python
# Fixed-point update α on S1
# Model selection (μ, γ, ν) on S2
# (See MAP pipeline documentation)
```

## ✅ Validation Checklist

Trước khi chuyển sang MAP training:

- [ ] All tests passed (`test_gating_map.py`)
- [ ] Gating trained successfully
- [ ] Mixture acc > best individual expert
- [ ] Effective experts ≥ 2.0 (for E=3)
- [ ] Balanced acc reasonable (head & tail gap < 20%)
- [ ] Best model checkpoint saved

## 📚 Next Steps

1. ✅ **Gating trained** (you are here)
2. ⏭️ **S1: Fixed-point α optimization**
3. ⏭️ **S2: Grid search (μ, γ, ν)**
4. ⏭️ **EG-outer** (optional, for R_max)
5. ⏭️ **CRC** (optional, for finite-sample guarantee)
6. ⏭️ **Final evaluation on test set**

## 🎓 Understanding Output

### Terminal Output Example:
```
Epoch  50/100:
  Train Loss: 3.2451 (NLL=3.2398), LR=0.000850
  Mixture Acc: 0.4523, Effective Experts: 2.87
  Val Loss: 3.3012
  Val Acc: Overall=0.4520, Head=0.7234, Tail=0.5678, Balanced=0.6456
  💾 Saved best model (balanced_acc=0.6456)
```

**Interpretation:**
- NLL ~3.2: reasonable (log(100) ≈ 4.6 là random)
- Mixture acc 0.45: tốt cho CIFAR-100 (100 classes)
- Effective experts 2.87/3: excellent diversity
- Balanced acc 0.65: good balance head/tail

### Checkpoint Contents:
```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': {...},  # Gating weights
    'optimizer_state_dict': {...},
    'val_metrics': {
        'mixture_acc': 0.4520,
        'balanced_acc': 0.6456,
        'head_acc': 0.7234,
        'tail_acc': 0.5678,
        ...
    },
    'config': {...}  # Full training config
}
```

---

## 🚨 Common Mistakes

❌ **Forget PYTHONPATH**
```bash
python3 src/train/train_gating_map.py  # Will fail
```
✅ **Correct:**
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/train_gating_map.py
```

❌ **Wrong expert names in config**
```python
'experts': {
    'names': ['expert1', 'expert2']  # Must match folder names!
}
```

❌ **Use uncalibrated logits**
→ Mixture NLL sẽ không có ý nghĩa (not proper probabilities)

✅ **Always use calibrated expert logits** (từ TemperatureScaling)

---

Happy training! 🎉

Nếu gặp vấn đề, check:
1. Test suite: `PYTHONPATH=$PWD:$PYTHONPATH python3 src/train/test_gating_map.py`
2. Documentation: `docs/gating_network_implementation.md`
3. Expert logits có tồn tại và shape đúng không
