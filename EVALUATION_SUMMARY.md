# 🎯 Tóm tắt vấn đề AURC Evaluation

## Bạn đã làm đúng hầu hết, nhưng có MỘT vấn đề quan trọng:

### ✅ Những gì đã FIX (ĐÚNG):
1. **Bug #1 - Margin calculation**: ✅ Fixed
   - Trước: Dùng `compute_margin()` với c=0
   - Sau: Dùng `compute_raw_margin()` (không subtract c)
   - **Impact**: Threshold behavior đã đúng (threshold = -c)

2. **Reweighting**: ✅ Implemented correctly  
   - Class weights loaded và applied
   - Metrics reflect long-tail performance

3. **Data splits**: ✅ Correct
   - tunev + val cho validation
   - test cho evaluation

### ⚠️ Vấn đề còn lại (QUAN TRỌNG):

**Plugin training dùng PER-GROUP thresholds, nhưng AURC eval dùng GLOBAL threshold!**

```python
# Plugin Training (trong gse_balanced_plugin.py):
t_head = -0.427  # Head group threshold
t_tail = -0.034  # Tail group threshold
accept = (raw_margin > t_group[sample_group])  # Khác nhau theo group!

# AURC Evaluation (trong eval_gse_plugin.py):
threshold = -c  # SAME for all samples
accept = (raw_margin >= threshold)  # Giống nhau cho mọi sample!
```

**Checkpoint cho thấy:**
```
α = [1.0, 1.0]              ← Không optimize
μ = [-0.39, 0.39]           ← Đã optimize  
t_group = [-0.427, -0.034]  ← Per-group thresholds
Best training error = 0.2105 ← Rất tốt!

Nhưng AURC = 0.354 / 0.540  ← Cao hơn nhiều! Why?
```

**Lý do:** Parameters được optimize cho per-group mechanism, nhưng evaluate theo global mechanism!

## 🔧 Cần làm gì tiếp theo?

### Option 1: Fix AURC evaluation (RECOMMENDED)
Sửa `eval_gse_plugin.py` để dùng per-group thresholds:
```python
# Thay vì
threshold = -c
accepted = (raw_margins >= threshold)

# Nên dùng
t_group = ckpt['t_group']  # Load từ checkpoint
y_groups = class_to_group[labels]
thresholds_per_sample = t_group[y_groups]
accepted = (raw_margins > thresholds_per_sample)
```

**Dự đoán:** AURC sẽ giảm xuống ~0.20-0.30 (gần với training error 0.2105)

### Option 2: Re-train plugin với global threshold
Sửa plugin training để optimize global threshold thay vì per-group.

**Trade-off:** Mất flexibility nhưng đơn giản hơn.

## 📊 Kết quả hiện tại có sử dụng được không?

**Có**, nhưng cần lưu ý:
- AURC values (0.35/0.54) là **upper bound** (pessimistic)
- Thực tế plugin tốt hơn (training error = 0.21)
- Nếu cần con số chính xác cho paper, phải fix evaluation

## 🎯 Khuyến nghị

**Ngắn hạn (nếu cần kết quả nhanh):**
- Dùng kết quả hiện tại nhưng note rằng "evaluated with global threshold"
- Report training error (0.2105) as reference

**Dài hạn (nếu cần chính xác):**
- Implement per-group AURC evaluation
- Re-run và so sánh

Bạn muốn tôi implement Option 1 (fix AURC eval) không?
