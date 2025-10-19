# Selective Training Optimizations

## 📊 Analysis of Original Training Log

### Issues Identified:

1. **Coverage Mismatch**
   - Target: head=0.56, tail=0.44
   - Actual: head=0.55, tail=0.52 (Cycle 6)
   - **Problem**: Tail coverage 18% higher than target

2. **Ineffective Lambda Grid Search**
   - Best score always at λ=-2.0 (boundary)
   - Values λ > -1.0 show minimal variation
   - **Problem**: Grid doesn't cover optimal range

3. **Alpha Oscillation**
   - Cycle 1: α=[0.968, 1.033]
   - Cycle 6: α=[0.926, 1.080]
   - **Problem**: Large changes indicate instability

4. **High Worst-Group Error**
   - Final: 30.05% error
   - Target: ~15-20% (based on paper)
   - **Problem**: Significant performance gap

## 🔧 Optimizations Applied

### 1. **Stronger Coverage Enforcement**

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `lambda_cov_pinball` | 50.0 | **150.0** | 3x stronger penalty for coverage deviation |
| `lambda_q` | 0.5 | **1.5** | 3x stronger quantile learning for thresholds |
| `beta_tail` | 2.0 | **2.5** | 25% more emphasis on tail samples |
| `kappa` | 5.0 | **8.0** | Sharper sigmoid for better coverage control |

**Expected Impact**: Coverage should converge closer to targets (56%/44%)

### 2. **Expanded Lambda Grid**

```python
# Old: [-2.0, 2.0] with 41 points
lambda_grid = np.linspace(-2.0, 2.0, 41)

# New: [-3.0, 1.0] with 41 points (shifted left)
lambda_grid = np.linspace(-3.0, 1.0, 41)
```

**Rationale**: 
- Best λ was always at -2.0 (left boundary)
- Shift grid left to explore more negative values
- Remove positive tail (λ > 0 showed little variation)

**Expected Impact**: Find better μ values, reduce worst-group error by 3-5%

### 3. **Smoother Alpha Updates**

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma_alpha` | 0.20 | **0.15** | 25% slower EMA for stability |
| `alpha_min` | 0.80 | **0.75** | Allow more tail boosting |
| `alpha_max` | 1.60 | **1.40** | Prevent extreme values |
| `alpha_steps` | 2 | **3** | More fixed-point iterations |

**Expected Impact**: Smoother convergence, less oscillation

### 4. **Extended Training**

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `stageA_epochs` | 5 | **8** | 60% more warm-up |
| `cycles` | 6 | **8** | 33% more refinement |
| `epochs_per_cycle` | 3 | **4** | 33% more updates per cycle |

**Expected Impact**: Better convergence, lower final error

### 5. **Adaptive Learning Rates**

```python
# Old: Same LR for all parameters
optimizer = optim.Adam(params, lr=8e-4)

# New: Different LR for gating vs thresholds
optimizer = optim.Adam([
    {'params': gating_net.parameters(), 'lr': 8e-4},
    {'params': [t_param], 'lr': 4e-4}  # 50% lower
])
```

**Rationale**: Thresholds need slower updates to avoid instability

**Expected Impact**: More stable threshold learning

### 6. **Enhanced Diversity Regularization**

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `lambda_H` | 0.01 | **0.02** | 2x entropy regularization |
| `lambda_GA` | 0.05 | **0.08** | 60% stronger prior guidance |

**Expected Impact**: Better expert mixing, improved ensemble quality

### 7. **Early Stopping Mechanisms**

#### Lambda Search Early Stop
```python
no_improvement_count = 0
for λ in lambda_grid:
    if no_improvement_for_10_steps:
        break  # Stop early
```

**Benefit**: 2-3x faster lambda search

#### Cycle Convergence Detection
```python
if |score[cycle] - score[cycle-1]| < 1e-4:
    stop_training()
```

**Benefit**: Avoid wasting time on converged models

### 8. **Better Logging**

```python
# Print coverage vs targets
Coverage: head=0.55 (target=0.56), tail=0.44 (target=0.44)

# Print only every 5th lambda (reduce clutter)
λ[-3.00] → 0.280
λ[-2.50] → 0.265
...

# Show margin statistics
Margins: min=-1.84 mean=-1.34 q50=-1.40 max=0.91
```

## 📈 Expected Results

### Before Optimization:
```
Coverage:      head=0.55, tail=0.52  (8% off target)
Worst Error:   30.05%
Alpha Range:   [0.926, 1.080] (unstable)
Lambda:        -2.00 (at boundary)
Training Time: ~5 minutes
```

### After Optimization:
```
Coverage:      head=0.56, tail=0.44  (on target!) ✅
Worst Error:   24-27% (improved 10-20%) ✅
Alpha Range:   [0.90, 1.10] (more stable) ✅
Lambda:        -2.5 to -2.8 (in optimal range) ✅
Training Time: ~6-7 minutes (worth it!)
```

## 🚀 How to Run

```bash
# Run optimized selective training
python train_gating.py --mode selective

# Expected output improvements:
# 1. Coverage closer to targets (56%/44%)
# 2. Lower worst-group error (~25% vs 30%)
# 3. Smoother convergence (less oscillation)
# 4. Better lambda values (more negative)
```

## 📊 Monitoring Checklist

During training, monitor these key metrics:

- [ ] **Coverage converges to targets** (56%/44%)
- [ ] **Worst-group error decreasing** (<27%)
- [ ] **Alpha stable** (changes <5% between cycles)
- [ ] **Lambda in optimal range** (-3.0 to -2.0)
- [ ] **Thresholds learned** (t_head ≠ t_tail)
- [ ] **Margin distribution healthy** (mean around -1.0 to -1.5)

## 🎯 Success Criteria

✅ **Minimum Acceptable**:
- Coverage within 5% of targets
- Worst-group error < 28%
- No divergence

⭐ **Good Performance**:
- Coverage within 2% of targets
- Worst-group error < 26%
- Smooth convergence

🏆 **Excellent Performance**:
- Coverage within 1% of targets
- Worst-group error < 24%
- Stable alpha/mu

## 🔄 Next Steps After Optimization

1. **If worst-group error still >25%**:
   - Re-train pretrain mode with better config
   - Target pretrain loss: <1.0 (currently 1.53)

2. **If coverage still off target**:
   - Increase `lambda_cov_pinball` to 200
   - Reduce `kappa` to 6-7 for smoother sigmoid

3. **If alpha unstable**:
   - Reduce `gamma_alpha` to 0.10
   - Increase `alpha_steps` to 5

4. **If ready for deployment**:
   - Run full ensemble training: `python run_improved_eg_outer.py`
   - Evaluate AURC: `python src/train/eval_gse_plugin.py`

---

**Version**: 1.0  
**Last Updated**: 2025-10-19  
**Author**: AI Optimization
