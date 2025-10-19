# AR-GSE Training - Quick Reference

## 📊 Data Splits Overview

| Split | Source | Size | Distribution | Purpose |
|-------|--------|------|--------------|---------|
| **Expert** | CIFAR-100 train | 9,719 | Long-tail (IF≈112) | Train experts |
| **Gating** | CIFAR-100 train | 1,128 | Long-tail (IF≈50) | Pretrain gating |
| **Val** | CIFAR-100 test | 1,000 | Balanced (10/class) | Validation |
| **TuneV** | CIFAR-100 test | 1,000 | Balanced (10/class) | Selective training (S1) |
| **Test** | CIFAR-100 test | 8,000 | Balanced (80/class) | Final evaluation |

## 🚀 Training Commands

### 1. Create Splits
```bash
python create_splits_fixed.py --split-train --visualize
```

### 2. Train Experts
```bash
# Train all experts
python train_experts.py

# Or individually
python train_experts.py --expert ce
python train_experts.py --expert logitadjust
python train_experts.py --expert balsoftmax
```

### 3. Train Gating
```bash
# Pretrain mode (uses gating split)
python train_gating.py --mode pretrain

# Selective mode (uses tunev + val)
python train_gating.py --mode selective
```

### 4. Evaluate
```bash
python evaluate_argse.py
```

## 🧪 Testing

```bash
# Test expert dataloader
python test_expert_dataloader.py

# Test gating data
python test_gating_data.py
```

## 📁 Key Files

- **Splits**: `data/cifar100_lt_if100_splits_fixed/*.json`
- **Weights**: `data/cifar100_lt_if100_splits_fixed/class_weights.json`
- **Expert models**: `checkpoints/experts/cifar100_lt_if100/`
- **Expert logits**: `outputs/logits/cifar100_lt_if100/{expert}/`
- **Gating models**: `checkpoints/gating_pretrained/cifar100_lt_if100/`

## 🔑 Key Changes

✅ **Expert training**: Uses 90% of train (expert split) + reweighted validation  
✅ **Gating pretrain**: Uses 10% of train (gating split)  
✅ **Gating selective**: Uses balanced tunev (S1) + val (S2)  
✅ **No duplication**: All test-based splits are balanced  
✅ **Reweighting**: Validation metrics reflect long-tail performance  

## 📖 Documentation

- **Full pipeline**: [`docs/training_pipeline_updated.md`](training_pipeline_updated.md)
- **Dataset info**: [`docs/dataset.md`](dataset.md)
- **Expert training**: [`docs/experts_training.md`](experts_training.md)
- **Gating training**: [`docs/gating_training.md`](gating_training.md)
