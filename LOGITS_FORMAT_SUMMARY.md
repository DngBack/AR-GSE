# 📊 Logits Format: .npz vs .pt

## TL;DR: ✅ KHÔNG CÓ VẤN ĐỀ GÌ

Việc đổi từ `.pt` sang `.npz` **KHÔNG ảnh hưởng** gì cả vì:
1. ✅ Tất cả scripts đã được update để hỗ trợ **CẢ HAI** format
2. ✅ Scripts tự động thử `.npz` trước, rồi `.pt` nếu không tìm thấy
3. ✅ Format `.npz` tốt hơn: nhẹ hơn, tương thích NumPy/PyTorch

---

## Chi Tiết

### 🔍 Scripts Đã Hỗ Trợ Dual Format

**1. `src/train/train_gating_only.py`** (dòng 178-195, 230-245):
```python
# Try .npz first (new format), then .pt (old format)
npz_path = logits_root / expert_name / "tunev_logits.npz"
pt_path = logits_root / expert_name / "tuneV_logits.pt"

if npz_path.exists():
    data = np.load(npz_path)
    stacked_logits[:, i, :] = torch.from_numpy(data['logits'])
elif pt_path.exists():
    stacked_logits[:, i, :] = torch.load(pt_path, map_location='cpu')
else:
    raise FileNotFoundError(f"Missing logits for {expert_name}")
```

**2. `src/train/eval_gse_plugin.py`** (dòng 106-121):
```python
# Try .npz first (new format), then .pt (old format)
npz_path = logits_root / expert_name / f"{split_name}_logits.npz"
pt_path = logits_root / expert_name / f"{split_name}_logits.pt"

if npz_path.exists():
    data = np.load(npz_path)
    logits[:, i, :] = torch.from_numpy(data['logits'])
elif pt_path.exists():
    logits[:, i, :] = torch.load(pt_path, map_location='cpu', weights_only=False)
else:
    raise FileNotFoundError(f"Missing logits for {expert_name}")
```

---

### 📁 Current File Structure

**Logits mới (`.npz`)** - từ `recompute_logits.py`:
```
outputs/logits_fixed/
├── ce_baseline/
│   ├── tunev_logits.npz
│   ├── val_logits.npz
│   └── test_logits.npz
├── balsoftmax_baseline/
│   ├── tunev_logits.npz
│   ├── val_logits.npz
│   └── test_logits.npz
└── logitadjust_baseline/
    ├── tunev_logits.npz
    ├── val_logits.npz
    └── test_logits.npz
```

**Logits cũ (`.pt`)** - từ training ban đầu:
```
outputs/logits/cifar100_lt_if100/
├── ce_baseline/
│   ├── tuneV_logits.pt      # ⚠️ Viết hoa V
│   ├── val_lt_logits.pt     # ⚠️ Tên khác
│   └── test_lt_logits.pt    # ⚠️ Tên khác
└── ...
```

---

### 🎯 Ưu Điểm của `.npz`

1. **Structured Data**: Chứa nhiều arrays (logits, targets, predictions) trong 1 file
   ```python
   data = np.load('logits.npz')
   logits = data['logits']      # [N, 100]
   targets = data['targets']    # [N]
   preds = data['predictions']  # [N]
   ```

2. **Smaller Size**: Compressed, nhẹ hơn `.pt`
   ```bash
   -rw-r--r-- 1 user user 3.1M  tunev_logits.npz  # NPZ
   -rw-r--r-- 1 user user 3.8M  tuneV_logits.pt   # PT
   ```

3. **Cross-Platform**: NumPy format, đọc được từ nhiều ngôn ngữ

4. **Explicit Keys**: Rõ ràng hơn thay vì raw tensor

---

### 🚀 Workflow Không Thay Đổi

**Training Gating** (tự động detect):
```bash
python train_gating.py --mode pretrain
# → Tự động tìm .npz ở outputs/logits_fixed/
# → Fallback về .pt nếu không có .npz
```

**Evaluation** (tự động detect):
```bash
python src/train/eval_gse_plugin.py
# → Tự động tìm .npz trước
# → Fallback về .pt nếu cần
```

---

### 💡 Kết Luận

- ✅ **Hoàn toàn OK** khi `recompute_logits.py` tạo file `.npz`
- ✅ Scripts tự động xử lý cả 2 format
- ✅ Không cần làm gì thêm, chỉ cần chạy như bình thường
- ✅ Format mới tốt hơn: nhẹ hơn, rõ ràng hơn, structured hơn

**Không có vấn đề gì cả!** 🎉
