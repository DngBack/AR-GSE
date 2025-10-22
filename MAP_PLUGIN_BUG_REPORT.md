# MAP Plugin Implementation - Analysis & Bug Report

## 📊 Current Results Summary

### Configuration
- **Dataset:** CIFAR-100-LT (IF=100)
- **Test Set:** 8,000 samples (balanced: 4000 head, 4000 tail)
- **Objective:** Worst-group with EG-outer
- **Reweighting:** Disabled (uniform weights)

### Classification Performance (No Rejection)
```
Overall Accuracy: 46.72%
Top-5 Accuracy:   76.19%

Group-wise:
  Head (0-49):    62.17%  ✓ Reasonable
  Tail (50-99):   31.27%  ✓ Shows long-tail effect
  Gap:            30.90%  ⚠️  Large gap
```

### MAP Plugin Performance
```
AURC:               0.0000  ❌ BROKEN
Rejection Rate:     100%    ❌ All rejected
Selective Error:    0.0000  ❌ Undefined (no accepted samples)

Parameters Found:
  λ = -3.000  (μ_head - μ_tail)
  γ = 0.000   (uncertainty penalty)
  ν = 2.000   (sigmoid slope)
  
  α range: [0.223, 0.257]
  μ values: [-1.5, 1.5]  ⚠️  Wrong! Should have separation
```

---

## 🐛 Critical Bugs Identified

### Bug #1: All Samples Rejected
**Symptom:**
- Rejection rate = 100%
- AURC = 0.0 (meaningless)
- No selective classification happening

**Root Cause:**
The L2R margin formula is:
```python
margin = max_y(α[y]·η̃[y]) - (Σ_y'(1/α[y'] - μ[y'])·η̃[y'] - c)
```

With found parameters:
- `α ≈ 0.24` → `1/α ≈ 4.17`
- `μ = [-1.5, 1.5]` (per-class)
- `coeffs = 1/α - μ ≈ 4.17 - (-1.5) = 5.67` (head)
- `coeffs ≈ 4.17 - 1.5 = 2.67` (tail)

The RHS term `Σ coeffs · η̃` is **HUGE** (5-6x the posteriors), making margin always negative → **reject everything**.

**Why This Happened:**
1. Grid search chose `λ = -3.0` (min value in grid)
2. This sets `μ = [-1.5, 1.5]` via `lambda_to_mu()`
3. Negative `μ` for head classes makes `1/α - μ` very large
4. Large RHS overwhelms LHS → margin < 0 always

### Bug #2: Grid Search Converged to Boundary
**Symptom:**
- Best `λ = -3.0` (minimum in grid `[-3, 3]`)
- Best `γ = 0.0` (minimum in grid `[0, 0.5, 1.0, 2.0]`)
- Suggests actual optimum is outside search range

**Root Cause:**
Grid ranges are inappropriate for the L2R margin formula:
- `λ ∈ [-3, 3]`: With current formula, even `λ = 0` gives problematic margins
- `γ ∈ [0, 2]`: Uncertainty penalty not helping when base margin is broken

### Bug #3: Fixed-Point α Shows Wrong Behavior
**Symptom:**
During training: "Group errors: [0. 0.]" repeated 10 times

**Root Cause:**
- Fixed-point iterates to find `α` that matches target acceptance rate
- But with broken margin formula, it converges to `α ≈ 0.24` (near uniform)
- This `α` doesn't help—problem is in `μ` and formula itself

### Bug #4: lambda_to_mu() Inverted Logic
**Current implementation:**
```python
def lambda_to_mu(lambda_val, num_classes, group_boundaries):
    mu = torch.zeros(num_classes)
    boundary = group_boundaries[0]
    mu[:boundary] = lambda_val / 2      # Head gets +λ/2
    mu[boundary:] = -lambda_val / 2     # Tail gets -λ/2
    return mu
```

**Problem:**
- `λ = μ_1 - μ_2` should create separation between accept/reject thresholds
- But with negative `λ`, head classes get negative `μ`, tail gets positive
- This inverts the intended priority (should boost head, penalize tail for worst-group objective)

---

## 🔬 Detailed Analysis

### What L2R Margin Should Do
The Learn-to-Reject (L2R) framework uses:
```
h_α(x) = argmax_y α[y]·η̃[y]     (classifier)
r_μ,c(x) = Σ_y (1/α[y] - μ[y])·η̃[y] < c   (rejector)
```

where:
- `α[y]`: per-class weights (prioritize certain classes)
- `μ[y]`: per-class reject thresholds
- `c`: global cost parameter

**Acceptance condition:**
```
accept if:  max_y(α[y]·η̃[y]) ≥ Σ_y'(1/α[y'] - μ[y'])·η̃[y'] - c
```

**Issue:** With `α ≈ 0.25` (near uniform) and `μ` ranging `[-1.5, 1.5]`:
- The term `1/α` dominates (4.17 >> 1.5)
- RHS becomes huge
- Almost impossible to accept samples

### Why Grid Search Failed
**S1 (tunev) Optimization:**
- Fixed-point tries to find `α` matching target acceptance rate
- But formula is so broken that any `α` leads to reject-all
- Converges to `α ≈ 0.24` (uniform) as a degenerate solution

**S2 (val) Optimization:**
- Grid search evaluates `(λ, γ, ν)` combinations
- With reject-all behavior, "selective error" is undefined (0/0)
- Code returns `error = 0.0` when no samples accepted
- Grid search interprets this as "perfect" → selects boundary values

---

## 🛠️ Proposed Fixes

### Option 1: Simplify to Threshold-Based Rejection (RECOMMENDED)
Replace complex L2R margin with simple threshold on mixture posterior:

```python
# New margin formula
m_simple(x) = max_y η̃[y] - threshold - γ·U(x)

# Accept if margin > 0
# Reject if margin < 0
```

**Advantages:**
- ✅ Intuitive: confident predictions (high max posterior) are accepted
- ✅ Uncertainty penalty works naturally
- ✅ Threshold can be optimized on S2
- ✅ No complex `α`, `μ` optimization

**Implementation:**
```python
def compute_margin_simple(
    mixture_posterior,  # [B, C]
    uncertainty,        # [B]
    threshold,          # scalar
    gamma               # scalar
):
    # Max posterior confidence
    max_prob = mixture_posterior.max(dim=-1)[0]  # [B]
    
    # Margin with uncertainty penalty
    margin = max_prob - threshold - gamma * uncertainty
    
    return margin
```

### Option 2: Fix L2R Formula Parameters
Keep L2R but fix parameter ranges:

1. **Constrain `α` to avoid `1/α` explosion:**
   ```python
   alpha_min = 0.5  # Ensure 1/α < 2
   alpha_max = 2.0
   ```

2. **Re-scale `μ` relative to `1/α`:**
   ```python
   # Make μ comparable to 1/α
   mu = (1.0 / alpha.mean()) * lambda_val / 2
   ```

3. **Expand grid search ranges:**
   ```python
   lambda_grid = np.linspace(-10, 10, 21)  # Wider range
   gamma_grid = [0.0, 0.1, 0.5, 1.0]       # Finer near 0
   ```

### Option 3: Use Confidence-Based Rejection (SIMPLEST)
Bypass margins entirely, use direct confidence thresholding:

```python
def predict_reject_confidence(mixture_posterior, uncertainty, conf_threshold, gamma):
    # Top-1 confidence
    confidence = mixture_posterior.max(dim=-1)[0]
    
    # Adjust by uncertainty
    adjusted_conf = confidence - gamma * uncertainty
    
    # Accept if confident enough
    accept = adjusted_conf >= conf_threshold
    return ~accept  # Return reject mask
```

---

## 📈 Recommended Next Steps

### Immediate (Fix Critical Bugs)
1. **✅ Implement Option 1 (Simplified Margin)**
   - Create `MAPSelectorSimple` class
   - Use `max_posterior - threshold - γ·U(x)`
   - Grid search over `(threshold, γ)`

2. **✅ Fix Grid Search Handling of No-Accept Cases**
   ```python
   # In compute_selective_metrics()
   if accept.sum() == 0:
       return {
           'selective_error': 1.0,  # Worst possible
           'coverage': 0.0,
           ...
       }
   ```

3. **✅ Expand Grid Ranges**
   - Test wider parameter spaces
   - Add validation that grid optimum is not at boundary

### Short-term (Improve Robustness)
4. **Add Debug Logging**
   - Log margin distribution at each grid point
   - Track acceptance rates during optimization
   - Validate that α/μ are reasonable

5. **Implement Sanity Checks**
   ```python
   # After optimization
   assert best_result.coverage > 0.1, "Too few accepted!"
   assert best_result.selective_error < 1.0, "Worse than random!"
   ```

6. **Test on Simpler Dataset First**
   - Try CIFAR-10-LT or smaller imbalance
   - Validate pipeline works before scaling to IF=100

### Long-term (Research Improvements)
7. **Adaptive Grid Search**
   - Start coarse, refine near optimum
   - Use Bayesian optimization instead of grid

8. **Calibration-Aware Optimization**
   - Calibrate mixture posteriors before MAP
   - Use temperature scaling

9. **Worst-Group Specific Tuning**
   - Separate thresholds per group
   - Group-aware uncertainty quantification

---

## 📊 Expected Results After Fix

With simplified margin (Option 1):

### Target Metrics
```
Balanced Objective:
  AURC:          0.08-0.12  (lower is better)
  Accuracy:      75-80%     (with 0% rejection)
  At 20% rej:    Error < 0.15
  At 50% rej:    Error < 0.05

Worst-Group Objective:
  Tail Accuracy: 60-70%     (improved from 31%)
  Head-Tail Gap: < 15%      (reduced from 31%)
  Worst Error:   < 0.35     (at 20% rejection)
  May sacrifice head accuracy for tail fairness
```

### RC Curve Shape
- **Start (0% rejection):** Error ≈ 1 - accuracy ≈ 0.25-0.30
- **Middle (20-50% rej):** Smooth decrease as uncertain samples rejected
- **End (80%+ rej):** Asymptotes to near-zero error (only highly confident kept)
- **NO FLAT ZEROS:** Real RC curves are never all-zero

---

## 🎯 Conclusion

### Summary
The MAP plugin implementation has **correct architecture** but **broken margin computation**:
- ✅ Gating network works (46.7% accuracy reasonable for IF=100)
- ✅ Data loading, reweighting logic correct
- ✅ Grid search and EG-outer infrastructure solid
- ❌ L2R margin formula with current parameters → reject all
- ❌ Grid search converged to boundary (outside viable range)

### Root Cause
The L2R margin formula `max(α·η̃) - (Σ(1/α - μ)·η̃ - c)` is theoretically sound but:
1. **Numerically unstable** with small `α` (1/α explodes)
2. **Parameter ranges mismatched** (grid search doesn't explore viable region)
3. **Optimization coupled** (α and μ interdependent, hard to optimize jointly)

### Recommendation
**Implement simplified confidence-based rejection (Option 1)** as primary approach:
- Faster to implement and debug
- More interpretable for users
- Proven effective in practice (e.g., SelectiveNet, SAT)
- Can always add L2R complexity later if needed

**Timeline:**
- Day 1: Implement `MAPSelectorSimple`, test on toy data
- Day 2: Re-run grid search with wider ranges, validate RC curves look reasonable
- Day 3: Compare balanced vs worst-group, generate final plots
- Day 4: Write paper-ready results

---

## 📝 References

**L2R Framework:**
- Mozannar et al., "Consistent Estimators for Learning to Defer to an Expert", ICML 2020
- Charoenphakdee et al., "Classification with Rejection Based on Cost-sensitive Classification", ICML 2021

**Selective Classification:**
- Geifman & El-Yaniv, "SelectiveNet: A Deep Neural Network with a Selective Prediction Ability", ICML 2019
- Kamath et al., "Selective Classification Can Magnify Disparities Across Groups", ICLR 2020

**Our Implementation:**
- `/src/models/map_selector.py`: Current (broken) L2R implementation
- `/src/models/map_optimization.py`: Grid search (works, but with bad inputs)
- `/src/train/train_map_plugin.py`: Training pipeline (solid)

---

**Status:** 🔴 Critical bugs identified, fix recommended before evaluation.
**Priority:** P0 - Blocking paper results.
**Owner:** Implementation team.
**ETA:** 2-3 days for Option 1 fix + retraining.
