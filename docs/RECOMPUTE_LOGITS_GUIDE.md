# 📘 Hướng Dẫn Recompute Logits

## 🎯 Mục Đích

Khi bạn có **expert weights đã train** nhưng **data splits mới**, thay vì train lại experts (mất ~4 giờ), chỉ cần **recompute logits** (mất ~15 phút).

---

## ⚙️ Chuẩn Bị

### 1. Kiểm Tra Expert Checkpoints

```bash
ls checkpoints/experts/cifar100_lt_if100/
```

**Kết quả mong đợi:**
```
best_ce_baseline.pth
best_balsoftmax_baseline.pth
best_logitadjust_baseline.pth
best_decoupling_twostage.pth (optional)
```

### 2. Kiểm Tra Data Splits Mới

```bash
ls data/cifar100_lt_if100_splits_fixed/
```

**Kết quả mong đợi:**
```
train_indices.json
test_indices.json (8,000 samples)
val_indices.json (1,000 samples)
tunev_indices.json (1,000 samples)
class_weights.json
train_class_counts.json
```

---

##  🚀 Cách Chạy

### Option 1: Tất Cả Experts (Recommended)

```bash
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline \
    --checkpoint-prefix best_
```

### Option 2: Custom Settings

```bash
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline \
    --checkpoint-prefix best_ \
    --splits val test \
    --batch-size 64 \
    --output-dir outputs/logits_new
```

### Option 3: Final Calibrated Checkpoints

```bash
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline \
    --checkpoint-prefix final_calibrated_
```

---

## 📊 Output Structure

```
outputs/logits_fixed/
├── ce_baseline/
│   ├── val_logits.npz       # 1,000 samples
│   ├── test_logits.npz      # 8,000 samples
│   └── tunev_logits.npz     # 1,000 samples
├── balsoftmax_baseline/
│   ├── val_logits.npz
│   ├── test_logits.npz
│   └── tunev_logits.npz
└── logitadjust_baseline/
    ├── val_logits.npz
    ├── test_logits.npz
    └── tunev_logits.npz
```

### Format của `.npz` file

```python
import numpy as np

data = np.load('outputs/logits_fixed/ce_baseline/val_logits.npz')

# Available arrays:
data['logits']       # Shape: (N, 100) - raw logits
data['targets']      # Shape: (N,) - true labels  
data['predictions']  # Shape: (N,) - predicted labels (argmax of logits)
```

---

## ✅ Verify Logits

Sau khi recompute, verify để đảm bảo đúng:

```bash
python3 verify_logits.py
```

**Check list:**
- ✓ Shape consistency
- ✓ Predictions match argmax
- ✓ Accuracy reasonable (>60%)
- ✓ No NaN or Inf values
- ✓ Reweighted metrics computed correctly

---

## 🔧 Troubleshooting

### Lỗi 1: Checkpoint not found

**Nguyên nhân:** Tên expert không đúng hoặc checkpoint không tồn tại

**Giải pháp:**
```bash
# List available checkpoints
ls checkpoints/experts/cifar100_lt_if100/

# Use exact names
python3 recompute_logits.py --experts ce_baseline balsoftmax_baseline
```

### Lỗi 2: Model architecture mismatch

**Error:**
```
RuntimeError: Error(s) in loading state_dict for Expert:
    Missing key(s): ...
    Unexpected key(s): ...
```

**Nguyên nhân:** Model class không khớp với checkpoint

**Giải pháp:** Script đã được update để tự động load `Expert` class từ `src.models.experts`

### Lỗi 3: Size mismatch

**Error:**
```
size mismatch for fc.weight: copying a param with shape torch.Size([100, 64]) 
from checkpoint, the shape in current model is torch.Size([100, 512]).
```

**Nguyên nhân:** Backbone khác nhau (ResNet-32 vs ResNet-18)

**Giải pháp:** Đảm bảo dùng đúng backbone. Script sử dụng `cifar_resnet32` theo mặc định.

### Lỗi 4: CUDA out of memory

**Giải pháp:**
```bash
python3 recompute_logits.py --batch-size 64
# hoặc
python3 recompute_logits.py --batch-size 32
```

### Lỗi 5: Wrong logits shape

**Nguyên nhân:** Data splits mới khác size với cũ

**Kiểm tra:**
```bash
# Check splits size
python3 -c "
import json
with open('data/cifar100_lt_if100_splits_fixed/val_indices.json') as f:
    print(f'Val: {len(json.load(f))} samples')
with open('data/cifar100_lt_if100_splits_fixed/test_indices.json') as f:
    print(f'Test: {len(json.load(f))} samples')
"
```

---

## 🔍 Advanced: Inspect Checkpoint

Nếu gặp lỗi, inspect checkpoint để hiểu cấu trúc:

```python
import torch

checkpoint = torch.load(
    'checkpoints/experts/cifar100_lt_if100/best_ce_baseline.pth',
    map_location='cpu'
)

print("Keys:", checkpoint.keys())

# Get state dict
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

# Print first few parameters
for i, (k, v) in enumerate(list(state_dict.items())[:10]):
    print(f"{k}: {v.shape}")
```

---

## 📝 Workflow Hoàn Chỉnh

```bash
# 1. Tạo data splits mới
python3 create_splits_fixed.py

# 2. Recompute logits
python3 recompute_logits.py \
    --experts ce_baseline balsoftmax_baseline logitadjust_baseline \
    --checkpoint-prefix best_

# 3. Verify
python3 verify_logits.py

# 4. Train gating (nếu logits OK)
python3 train_gating.py --mode pretrain

# 5. Continue training
python3 train_gating.py --mode selective
python3 train_argse.py
```

---

## 💡 Tips

1. **Use best vs final_calibrated:**
   - `best_*`: Best validation accuracy during training
   - `final_calibrated_*`: After temperature scaling
   - Thường dùng `best_` cho logits

2. **Batch size:**
   - Default 128 works for most GPUs
   - Giảm xuống 64/32 nếu out of memory
   - Tăng lên 256 nếu GPU lớn → faster

3. **Parallel processing:**
   Script chạy tuần tự từng expert. Nếu muốn nhanh hơn, chạy song song:
   ```bash
   python3 recompute_logits.py --experts ce_baseline &
   python3 recompute_logits.py --experts balsoftmax_baseline &
   python3 recompute_logits.py --experts logitadjust_baseline &
   wait
   ```

4. **Reuse logits:**
   Logits chỉ cần compute 1 lần. Sau đó có thể dùng lại cho nhiều experiments.

---

## ⏱️ Performance

| Expert | Samples | Time (GPU) | Memory |
|--------|---------|------------|---------|
| CE | 10,000 | ~2 min | ~2 GB |
| BalSoftmax | 10,000 | ~2 min | ~2 GB |
| LogitAdjust | 10,000 | ~2 min | ~2 GB |
| **Total** | **30,000** | **~6 min** | **~2 GB** |

Vs train từ đầu: **~4 hours** → Tiết kiệm **97.5%** thời gian! 🚀

---

## ✅ Success Criteria

Sau khi recompute, check:

1. **Files exist:**
   ```bash
   ls outputs/logits_fixed/*/val_logits.npz
   ls outputs/logits_fixed/*/test_logits.npz
   ls outputs/logits_fixed/*/tunev_logits.npz
   ```

2. **Accuracy reasonable:**
   ```
   CE: ~70-75%
   BalSoftmax: ~72-77%
   LogitAdjust: ~71-76%
   ```

3. **Shape correct:**
   ```
   Val: (1000, 100)
   Test: (8000, 100)
   TuneV: (1000, 100)
   ```

4. **No errors in verify:**
   ```bash
   python3 verify_logits.py  # Should show ✓ for all checks
   ```

---

Bây giờ bạn có thể proceed với gating training! 🎯
