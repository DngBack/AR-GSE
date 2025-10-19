# AURC Evaluation Bug Fix - AR-GSE

## 🐛 Bug Discovered

### Symptoms
- Coverage max chỉ ~0.996 (không đạt 1.0)
- AURC cao bất thường (0.38-0.50)
- Threshold behavior kỳ lạ:
  - c=0.0 → threshold=1.37 (lớn hơn max margin!)
  - c=0.1 → threshold=1.37 (giống nhau?)
  - Margins max chỉ 0.38 mà threshold lại 1.37

### Root Cause Analysis

**Sai lầm nghiêm trọng trong code cũ:**

```python
# File: eval_gse_plugin.py (BUG VERSION)

# Bước 1: Tính margin VỚI c=0 (nhưng vẫn trừ c trong hàm!)
from src.train.gse_balanced_plugin import compute_margin

margins = compute_margin(eta, alpha, mu, c=0.0, class_to_group)
# ↑ Hàm này LUÔN LUÔN trừ c: margin = raw - c
# Vậy margins = raw - 0.0 = raw (chưa có vấn đề...)

# Bước 2: Tìm optimal threshold (SAI!)
def find_optimal_threshold_for_cost(..., cost_c):
    # Tạo candidate thresholds từ data
    unique_scores = torch.unique(confidence_scores)
    thresholds = [min-1, ...unique_scores..., max+1]
    
    # Tìm threshold tốt nhất
    for threshold in thresholds:
        accepted = confidence_scores >= threshold
        # Optimize Chow's objective
        ...
    return best_threshold
```

**Vấn đề:**
1. AR-GSE decision rule: accept if `margin + c >= 0` hay `raw_margin >= -c`
2. Code đang tìm threshold từ data (như selective prediction thông thường)
3. Nhưng AR-GSE có **threshold deterministic**: `threshold = -c`
4. Không cần optimize - chỉ cần áp dụng trực tiếp!

### Why This Failed

**AR-GSE formulation:**
```
raw_margin = score - threshold_per_sample
           = max_y (α_g(y) * η_y) - Σ_y [(1/α_g(y) - μ_g(y)) * η_y]

Decision: accept if raw_margin + c >= 0
       ↔ accept if raw_margin >= -c
```

**Chow's rule (general selective prediction):**
```
confidence = any_score(x)
Decision: accept if confidence >= threshold
where threshold is optimized to minimize: coverage * risk + (1-coverage) * c
```

**Key difference:**
- AR-GSE: threshold = -c (deterministic, no optimization needed)
- Chow's rule: threshold = optimize(data) (depends on data distribution)

Code cũ đang dùng Chow's optimization cho AR-GSE → **SAI LOGIC!**

## ✅ Fix Applied

### Changes

#### 1. Import RAW margin function
```python
# OLD
from src.train.gse_balanced_plugin import compute_margin

# NEW  
from src.train.gse_balanced_plugin import compute_raw_margin
```

#### 2. Compute RAW margins (không trừ c)
```python
# OLD - Tính margin với c=0 (confusing!)
margins = compute_margin(eta, alpha, mu, c=0.0, class_to_group)

# NEW - Tính RAW margin (rõ ràng hơn)
raw_margins = compute_raw_margin(eta, alpha, mu, class_to_group)
```

#### 3. Simplified threshold finding
```python
# OLD - Optimize threshold từ data (40+ lines code, WRONG!)
def find_optimal_threshold_for_cost(..., cost_c):
    unique_scores = torch.unique(confidence_scores)
    thresholds = ...
    for threshold in thresholds:
        ...optimize...
    return best_threshold

# NEW - Direct computation (deterministic!)
def find_optimal_threshold_for_cost(..., cost_c):
    """
    For AR-GSE with raw margins, the threshold is simply -c.
    
    The AR-GSE decision rule is:
        accept if: raw_margin >= -c
    """
    return -cost_c
```

### Mathematical Justification

**AR-GSE objective** (from paper):
```
min_{α,μ,c} E[Error on accepted samples]
subject to: E[Coverage] = target_coverage
```

**Lagrangian:**
```
L = E[Coverage * Error] + λ * (E[Coverage] - target)
  = E[Coverage * Error] + λ * E[Coverage] - λ * target
```

**Optimal decision:**
```
Accept if: Error_probability < λ
```

For AR-GSE, the margin formulation gives:
```
margin = score - threshold_per_sample
Accept if: margin + c >= 0
where c = λ (Lagrange multiplier / rejection cost)
```

Therefore: **threshold = -c** (deterministic)

## 📊 Expected Results After Fix

### Before Fix (BUG):
```
Coverage range: 0.000 to 0.996
BALANCED AURC (0-1): 0.381203
WORST AURC (0-1): 0.507188

Threshold behavior:
c=0.00 → threshold=1.3713 → coverage=0.0000
c=0.10 → threshold=1.3713 → coverage=0.0000
c=0.50 → threshold=-0.6408 → coverage=0.4809
```

### After Fix (EXPECTED):
```
Coverage range: 0.000 to 1.000
BALANCED AURC (0-1): ~0.15-0.25 (much lower!)
WORST AURC (0-1): ~0.25-0.35

Threshold behavior:
c=0.00 → threshold=0.0000 → coverage=~0.70
c=0.10 → threshold=-0.1000 → coverage=~0.75
c=0.50 → threshold=-0.5000 → coverage=~0.85
c=0.99 → threshold=-0.9900 → coverage=~0.95
```

**Why lower AURC is expected:**
- Raw margins typically in range [-1, 1]
- threshold = -c sweeps from 0 to -1 as c goes 0→1
- This gives smooth coverage increase 0→1
- Previous bug had weird thresholds (1.37!) causing coverage stuck at 0

## 🎯 Verification Checklist

After running fixed code:

- [ ] Coverage reaches exactly 1.0 (accept all) when c=1.0
- [ ] Coverage is ~0 when c=0.0 (reject most)
- [ ] Threshold = -c for all cost values
- [ ] Raw margin distribution makes sense (e.g., mean ~0, range [-2, 2])
- [ ] AURC values are lower (better) than before
- [ ] Risk-coverage curves are smooth and monotonic

## 📝 Lessons Learned

1. **Know your algorithm**: AR-GSE ≠ General selective prediction
   - AR-GSE has deterministic threshold
   - Don't blindly copy code from other methods

2. **Check formulations**: 
   - margin vs raw_margin
   - When c is subtracted, when it's not
   - Decision rule: margin+c>=0 vs margin>=threshold

3. **Sanity checks**:
   - Threshold should be in reasonable range
   - Coverage should span [0, 1]
   - AURC should be < 0.5 (better than random)

4. **Code clarity**:
   - Use `raw_margin` name for margins without c
   - Use `margin` for margins after subtracting c
   - Add comments explaining the math

## 🔗 Related Files

- `src/train/eval_gse_plugin.py` - Main evaluation script (FIXED)
- `src/train/gse_balanced_plugin.py` - Plugin training (has compute_raw_margin)
- `docs/reweighting_explained.md` - Explains reweighting methodology
- `demo_reweighting.py` - Demo showing reweighting effects

## 📚 References

- Chow's rule: "On optimum recognition error and reject tradeoff" (1970)
- AR-GSE paper: Algorithm 1 (margin formulation)
- Selective prediction: Geifman & El-Yaniv (2017)
