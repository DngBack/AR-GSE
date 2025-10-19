# AR-GSE Architecture Mismatch - Critical Issue

## 🚨 TÓM TẮT VẤN ĐỀ

Plugin training và AURC evaluation đang dùng **HAI CƠ CHẾ THRESHOLD KHÁC NHAU**:

- **Training**: Per-group thresholds (t_head=-0.427, t_tail=-0.034)  
- **Evaluation**: Global threshold (threshold=-c for all samples)

→ **Parameters được optimize cho một objective nhưng evaluate theo objective khác!**

## 📊 Phân tích chi tiết

### Checkpoint Analysis
```
α = [1.0, 1.0]              ← Không được optimize
μ = [-0.39, 0.39]           ← Đã optimize
t_group = [-0.427, -0.034]  ← Per-group thresholds
Best validation error = 0.2105
```

### Current AURC Results
```
BALANCED AURC = 0.354612    ← Cao bất thường
WORST AURC = 0.540348       ← Rất cao
Coverage max = 0.952        ← Không đạt 1.0
```

### Why So Bad?

**Plugin was trained with:**
```python
# Different threshold per group
if sample_from_head: accept if raw_margin > -0.427
if sample_from_tail: accept if raw_margin > -0.034
```

**But AURC evaluates with:**
```python
# Same threshold for all
accept if raw_margin >= -c  (same c for everyone)
```

**Result:** Parameters (α,μ) optimal for per-group are suboptimal for global!

## ✅ GIẢI PHÁP

Sửa `eval_gse_plugin.py` để dùng per-group thresholds, nhất quán với training.

### Implementation Plan

**File cần sửa:** `src/train/eval_gse_plugin.py`

**Changes needed:**

1. Load `t_group` từ checkpoint
2. Implement per-group threshold evaluation
3. Update AURC computation để dùng per-group mechanism

**Pseudocode:**
```python
def aurc_per_group_thresholds(...):
    # Load per-group thresholds từ checkpoint
    t_group_opt = ckpt['t_group']  # [-0.427, -0.034]
    
    # Sweep scale factor thay vì sweep c
    for scale in [0.0, 0.1, ..., 1.0, 1.5, 2.0]:
        # Scale thresholds uniformly
        t_group = [t * scale for t in t_group_opt]
        
        # Apply per-group thresholds
        raw_margins = compute_raw_margin(...)
        y_groups = class_to_group[y]
        thresholds = t_group[y_groups]
        accepted = raw_margins > thresholds
        
        # Compute coverage and risk
        coverage = accepted.mean()
        risk = compute_group_risk(...)
        
        rc_points.append((scale, coverage, risk))
```

## 🎯 Expected Results After Fix

**Predicted AURC (với per-group thresholds):**
```
BALANCED AURC = 0.20-0.25   ← Giảm ~40%
WORST AURC = 0.30-0.40      ← Giảm ~30%  
Coverage max = 1.000        ← Full range
```

**Lý do dự đoán:**
- Plugin training đạt error 0.2105 với per-group thresholds
- Khi evaluate đúng mechanism, AURC phải gần với training error
- Current AURC 0.35-0.54 cao bất thường vì evaluate sai mechanism

## 📝 Next Steps

1. Implement per-group AURC evaluation
2. Re-run evaluation với correct mechanism
3. Compare results: global vs per-group
4. Document correct protocol
5. Consider re-training plugin if needed

## 🔗 Related Files

- `src/train/eval_gse_plugin.py` - Needs fix
- `src/train/gse_balanced_plugin.py` - Reference implementation
- `checkpoints/.../gse_balanced_plugin.ckpt` - Contains t_group
- `docs/aurc_bug_fix.md` - Previous bug fix documentation
