# MAP Plugin Quick Reference Guide

## 🎯 Overview

MAP (Mixture-Aware Plug-in) cho phép selective classification với rejection option, tối ưu hóa risk trên balanced/worst-group với sample reweighting.

---

## 📁 File Structure

```
src/
├── models/
│   ├── map_selector.py           # MAP classifier/rejector
│   └── map_optimization.py       # Grid search, EG-outer, RC curves
├── train/
│   ├── train_map_plugin.py       # Training script ⭐
│   └── eval_map_plugin.py        # Evaluation script ⭐
```

---

## 🚀 Quick Start

### **1. Train MAP Plugin (Balanced Objective)**

```bash
cd /home/duong.xuan.bach/AR-GSE

# Balanced objective (default)
python src/train/train_map_plugin.py --objective balanced

# Output:
# - checkpoints/map_plugin/cifar100_lt_if100/map_parameters.json
# - results/map_plugin/cifar100_lt_if100/rc_curve.json
```

**Expected Output:**
```
✅ Optimization completed!
   Best λ: 1.234
   Best γ: 0.500
   Best ν: 5.000

✅ RC curve computed
   AURC: 0.1234
```

---

### **2. Train MAP Plugin (Worst-Group Objective)**

```bash
# Worst-group với EG-outer
PYTHONPATH=$PWD:$PYTHONPATH python src/train/train_map_plugin.py --objective worst --eg_outer
```

**Chi tiết:**
- `--objective worst`: Tối ưu worst-group error
- `--eg_outer`: Sử dụng multiplicative weights (β) cho minimax
- S1: Fixed-point α với β-weighted acceptance
- S2: Grid search với β-weighted error

---

### **3. Evaluate on Test Set**

```bash
# With reweighting (default)
PYTHONPATH=$PWD:$PYTHONPATH python src/train/eval_map_plugin.py --split test --visualize

# Without reweighting
PYTHONPATH=$PWD:$PYTHONPATH python src/train/eval_map_plugin.py --split test --no_reweight
```

**Output Files:**
- `results/map_plugin/cifar100_lt_if100/evaluation_test.json`
- `results/map_plugin/cifar100_lt_if100/rc_curve_test.png`
- `results/map_plugin/cifar100_lt_if100/rc_curve_groups_test.png`

**Expected Output:**
```
📊 SUMMARY
Accuracy: 0.7654
AURC: 0.1234
Head Acc: 0.8123
Tail Acc: 0.6543
```

---

## 🔧 Configuration

### Training Config (`train_map_plugin.py`)

```python
CONFIG = {
    'map': {
        # Grid ranges
        'lambda_grid': [-3.0, -2.0, ..., 3.0],  # 13 points
        'gamma_grid': [0.0, 0.5, 1.0, 2.0],     # 4 points
        'nu_grid': [2.0, 5.0, 10.0],            # 3 points
        
        # Fixed-point
        'fp_iterations': 10,
        'fp_ema': 0.7,
        
        # EG-outer
        'eg_iterations': 10,
        'eg_xi': 0.1,
    },
    'evaluation': {
        'use_reweighting': True,  # ⚠️ Important cho balanced test
    }
}
```

---

## 📊 Key Concepts

### **1. Reweighting Cho Balanced Test**

**Problem:**
- Training data: long-tailed (IF=100)
- Test data: balanced (10 samples/class)

**Solution:**
```python
# Load class weights từ training distribution
class_weights = load_class_weights(splits_dir)  # [C]

# Convert to sample weights
sample_weights = class_weights[labels]  # [N]

# Use trong metrics
weighted_error = (errors * sample_weights).sum() / sample_weights.sum()
```

**Effect:**
- Tail classes nhận higher weights
- Metrics reflect true long-tail performance
- Không reweight → over-estimate tail accuracy

---

### **2. MAP Decision Rule**

```python
# L2R margin (mixture posterior)
m_L2R(x) = η̃₁(x) - η̃₂(x)

# Uncertainty penalty
U(x) = a·H(w) + b·Disagreement + d·H(η̃)

# MAP margin
m_MAP(x) = m_L2R(x) - γ·U(x)

# Rejection rule
accept  if  m_MAP(x) ≥ μ₁ - cost
reject  if  m_MAP(x) < μ₂ + cost
```

**Parameters:**
- **λ = μ₁ - μ₂**: separation between accept/reject thresholds
- **γ**: uncertainty penalty weight
- **ν**: sigmoid slope for soft acceptance (fixed-point only)
- **cost**: sweeps for RC curve

---

### **3. Two-Stage Optimization**

**Stage 1 (S1): Fixed-Point α**
```python
# Initialize α randomly
α = torch.rand(num_classes)

for iter in range(fp_iterations):
    # Soft acceptance rate per class
    p̂ₖ = sigmoid[ν·(m_MAP - μ)].mean()
    
    # Update α to match target
    α_new = (1 - target_rate) / p̂ₖ
    
    # EMA
    α = ema * α + (1 - ema) * α_new
```

**Stage 2 (S2): Grid Search**
```python
for λ in lambda_grid:
    for γ in gamma_grid:
        for ν in nu_grid:
            # Compute μ from λ
            μ = lambda_to_mu(λ, α)
            
            # Fixed-point α on S1
            α_opt = optimize_alpha(...)
            
            # Evaluate on S2
            error = compute_selective_error(S2, α_opt, μ, γ)
            
            if error < best_error:
                best = (λ, γ, ν, α_opt, μ)
```

---

### **4. EG-Outer for Worst-Group**

```python
# Initialize uniform weights
β = torch.ones(num_groups) / num_groups

for iter in range(eg_iterations):
    # Grid search với β-weighted error
    result = grid_search(objective='worst', beta=β)
    
    # Compute group errors
    errors_per_group = [...]
    
    # Update β (multiplicative weights)
    β = β * exp(ξ * errors_per_group)
    β = β / β.sum()
    
# Final: worst-group optimized parameters
```

---

## 🧪 Testing

### **Minimal Test**

```python
import torch
from src.models.map_selector import MAPSelector, MAPConfig

# Config
config = MAPConfig(num_classes=100, num_groups=2, group_boundaries=[50])
selector = MAPSelector(config)

# Fake data
posteriors = torch.randn(100, 100).softmax(dim=-1)
uncertainty = torch.rand(100)
labels = torch.randint(0, 100, (100,))

# Set parameters
alpha = torch.ones(100)
mu = torch.zeros(2)
selector.set_parameters(alpha=alpha, mu=mu, gamma=0.5, cost=0.0)

# Predictions
preds = selector.predict_class(posteriors, uncertainty)
accept = selector.predict_reject(posteriors, uncertainty)

print(f"Predictions: {preds}")
print(f"Accept rate: {accept.float().mean():.3f}")
```

---

## 📈 Interpreting Results

### **RC Curve Analysis**

```python
# Load results
with open('results/map_plugin/cifar100_lt_if100/rc_curve.json') as f:
    rc_data = json.load(f)

rejection_rates = rc_data['rejection_rates']
selective_errors = rc_data['selective_errors']
aurc = rc_data['aurc']

# Key points
for i, rej in enumerate([0.0, 0.2, 0.4]):
    idx = np.argmin(np.abs(rejection_rates - rej))
    print(f"Rej={rej:.1f}: Error={selective_errors[idx]:.4f}")
```

**Good RC Curve:**
- AURC càng thấp càng tốt
- Error giảm nhanh khi rejection tăng
- Smooth curve (không oscillate)

---

### **Group-wise Performance**

```python
# Load evaluation
with open('results/map_plugin/cifar100_lt_if100/evaluation_test.json') as f:
    results = json.load(f)

for g_id, metrics in results['group_metrics'].items():
    print(f"Group {g_id}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Count: {metrics['count']}")
    print(f"  Weight: {metrics['weight_sum']:.4f}")
```

**Worst-Group Objective:**
- Should reduce gap between head/tail
- May sacrifice head accuracy for tail

---

## ⚠️ Common Issues

### **1. Logits Not Found**

```
FileNotFoundError: Logits not found: outputs/logits/.../test_logits.pt
```

**Fix:**
```bash
# Generate expert logits first
python recompute_logits.py
```

---

### **2. Gating Checkpoint Missing**

```
FileNotFoundError: Gating checkpoint not found
```

**Fix:**
```bash
# Train gating first
python src/train/train_gating_map.py
```

---

### **3. class_weights.json Missing**

```
⚠️  class_weights.json not found, using uniform weights
```

**Fix:**
```bash
# Generate splits with class weights
python create_splits_fixed.py
```

**Or** disable reweighting:
```bash
python src/train/train_map_plugin.py --no_reweight
python src/train/eval_map_plugin.py --no_reweight
```

---

### **4. Grid Search Too Slow**

**Reduce grid size:**
```python
CONFIG = {
    'map': {
        'lambda_grid': list(np.linspace(-3.0, 3.0, 7)),  # 13→7
        'gamma_grid': [0.0, 1.0],                        # 4→2
        'nu_grid': [5.0],                                # 3→1
    }
}
```

**Total combinations:** 7 × 2 × 1 = 14 (vs 13 × 4 × 3 = 156)

---

## 🔬 Advanced Usage

### **Custom Objective Function**

```python
# In GridSearchOptimizer.search()

def custom_objective(errors_per_group, beta):
    """
    Example: Exponential penalty on tail
    """
    weights = torch.tensor([1.0, 2.0])  # Higher weight for tail
    return (errors_per_group * weights).sum()

# Modify optimizer
optimizer.objective_fn = custom_objective
```

---

### **Calibration Temperature**

```python
# In generate_mixture_posteriors()

# Before computing mixture
posteriors = torch.softmax(expert_logits / temperature, dim=-1)

# Typical: temperature ∈ [0.5, 2.0]
```

---

### **Multi-Stage Pipeline**

```bash
#!/bin/bash
# Full pipeline

# 1. Train experts
python train_experts.py

# 2. Generate logits
python recompute_logits.py

# 3. Train gating
python src/train/train_gating_map.py

# 4. Train MAP (balanced)
python src/train/train_map_plugin.py --objective balanced

# 5. Train MAP (worst-group)
python src/train/train_map_plugin.py --objective worst --eg_outer

# 6. Evaluate
python src/train/eval_map_plugin.py --split test --visualize
```

---

## 📚 References

**Implementation Files:**
- `src/models/map_selector.py`: Decision rules, fixed-point α
- `src/models/map_optimization.py`: Grid search, EG-outer, RC curves
- `src/train/train_map_plugin.py`: Training pipeline
- `src/train/eval_map_plugin.py`: Evaluation pipeline

**Documentation:**
- `docs/reweighting_explained.md`: Reweighting theory
- `TRAINING_PIPELINE_COMPLETE.md`: Full pipeline guide

---

## 🎓 Theory Summary

### **Objective Functions**

**Balanced:**
$$
R_{\text{bal}} = \sum_{k=1}^K \pi_k \cdot \mathbb{E}[\ell(x, y) | y=k, \text{accept}]
$$

**Worst-Group:**
$$
R_{\text{worst}} = \max_{g \in \{1, \ldots, G\}} \mathbb{E}[\ell(x, y) | g(y)=g, \text{accept}]
$$

### **MAP Margin**

$$
m_{\text{MAP}}(x) = m_{\text{L2R}}(x) - \gamma \cdot U(x)
$$

where:
- $m_{\text{L2R}}(x) = \eta_1(x) - \eta_2(x)$: L2R margin
- $U(x)$: uncertainty quantification
- $\gamma$: penalty weight

### **Decision Rule**

$$
h(x) = \begin{cases}
1 & \text{if } m_{\text{MAP}}(x) \geq \mu_1 - c \\
2 & \text{if } m_{\text{MAP}}(x) < \mu_2 + c \\
\emptyset & \text{otherwise (reject)}
\end{cases}
$$

---

## ✅ Checklist

**Before Training:**
- [ ] Experts trained (`checkpoints/experts/`)
- [ ] Logits exported (`outputs/logits/`)
- [ ] Gating trained (`checkpoints/gating_map/`)
- [ ] Splits created (`data/cifar100_lt_if100_splits_fixed/`)
- [ ] `class_weights.json` exists

**After Training:**
- [ ] `map_parameters.json` created
- [ ] `rc_curve.json` created
- [ ] AURC < 0.15 (reasonable)
- [ ] Tail accuracy improved

**Evaluation:**
- [ ] Test metrics computed
- [ ] Visualizations generated
- [ ] Group-wise metrics reasonable

---

## 🎉 Success Criteria

**Balanced Objective:**
- Overall accuracy ≥ 75%
- AURC ≤ 0.12
- Smooth RC curve

**Worst-Group Objective:**
- Tail accuracy ≥ 65%
- Head-tail gap ≤ 10%
- Worst-group error ≤ 0.35

---

**Happy Selecting! 🚀**
