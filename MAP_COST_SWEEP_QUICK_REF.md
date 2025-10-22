# MAP Cost Sweep - Quick Reference
## One-Page Cheat Sheet

---

## 🚀 QUICK START

```bash
# 1. Run cost sweep (VAL optimization + TEST evaluation)
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_cost_sweep.py --objective balanced
PYTHONPATH=$PWD:$PYTHONPATH python3 train_map_cost_sweep.py --objective worst

# 2. Analyze hull (optional)
python3 analyze_cost_sweep_hull.py --objective balanced

# 3. Check results
cat results/map_cost_sweep/cifar100_lt_if100/cost_sweep_balanced.json
```

---

## 📐 KEY FORMULAS

### Objectives
- **Balanced**: $R = \frac{1}{2}(e_H + e_T) + c \cdot \rho$
- **Worst**: $R = \max(e_H, e_T) + c \cdot \rho$

### Rejection Rule
$$m(x) = \text{conf}(x) - \theta - \gamma \cdot U(x)$$
- Accept if $m \geq 0$
- Reject if $m < 0$

### Uncertainty (Z-scored)
$$U' = \frac{U_{\text{raw}} - \mu_U}{\sigma_U}$$
- Computed on VAL, applied to TEST

### Group Error (with reweighting)
$$e_k = \frac{\sum_{i:g_i=k} w_i \cdot \mathbb{1}[\text{wrong}] \cdot A_i}{\sum_{i:g_i=k} w_i \cdot A_i}$$

---

## 📊 OUTPUT FILES

| File | Description |
|------|-------------|
| `cost_sweep_{obj}.json` | All results (VAL + TEST) |
| `aurc_curves_{obj}_test.png` | 3-panel plot (full, practical, comparison) |
| `hull_analysis_{obj}.json` | Hull AURC (optional) |
| `cost_sweep_hull_{obj}.png` | Hull visualization |

---

## 🎯 PAPER METRICS TO REPORT

### Primary Table
- **AURC (TEST)**: Main metric for model ranking
- **Operating points**: Error @ rejection rates {0, 0.1, 0.2, 0.5}
- **Group breakdown**: Head/Tail errors

### Ablations
- Reweighting on/off
- Balanced vs Worst comparison
- Hull vs standard AURC (optional)

---

## ⚙️ DEPLOYMENT CODE

```python
# Load config
config = results['results_per_cost'][best_idx]
theta, gamma = config['threshold'], config['gamma']
U_mu, U_sigma = config['uncertainty_mu'], config['uncertainty_sigma']

# Inference for new sample x
posteriors = softmax(expert_logits)  # [E, C]
weights = gating(posteriors)[0]      # [E]
mixture = (weights * posteriors).sum(0)  # [C]
U_raw = compute_uncertainty(...)
U = (U_raw - U_mu) / U_sigma  # ← CRITICAL!

margin = mixture.max() - theta - gamma * U
prediction = mixture.argmax() if margin >= 0 else REJECT
```

---

## 🔍 SANITY CHECKS

- [ ] Gating weights sum to 1: `assert torch.allclose(w.sum(), 1.0, atol=1e-5)`
- [ ] Mixture sums to 1: `assert torch.allclose(mix.sum(), 1.0, atol=1e-5)`
- [ ] Uncertainty normalized: mean≈0, std≈1 on VAL
- [ ] TEST uses same (μ, σ) as VAL
- [ ] AURC(TEST) > AURC(VAL) (slight generalization gap expected)
- [ ] Worst AURC > Balanced AURC (protects tail → higher overall error)

---

## 📈 EXPECTED RESULTS

### CIFAR-100-LT (IF=100)

| Metric | Balanced | Worst |
|--------|----------|-------|
| AURC (TEST) | ~0.15-0.18 | ~0.24-0.26 |
| Error @ ρ=0 | ~0.25 | ~0.25 |
| Error @ ρ=0.2 | ~0.19 | ~0.21 |
| Error @ ρ=0.5 | ~0.10 | ~0.14 |

**Interpretation**:
- Balanced: Lower overall error (optimizes mean)
- Worst: Higher overall error but protects tail (optimizes max)

---

## 🐛 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| NaN in gating weights | Check LayerNorm (should be disabled) |
| γ too large (>10) | Apply z-score to uncertainty |
| AURC(TEST) << AURC(VAL) | Bug in evaluation (should be slightly worse) |
| Balanced = Worst AURC | Check objective in RC curve (mean vs max) |
| High tail error | Increase rejection cost c |

---

## 📚 KEY FILES TO CHECK

1. **`src/models/map_selector_simple.py`**
   - Line 353: `mean(group_errors)` for Balanced
   - Line 366: `max(group_errors)` for Worst
   - Line 508-515: Objective-specific error in RC curve

2. **`train_map_cost_sweep.py`**
   - Line 295-298: Z-score normalization
   - Line 370: `gamma=best_result.gamma` (uses γ*)
   - Line 422-425: TEST uncertainty normalized with VAL stats

3. **`data/cifar100_lt_if100_splits_fixed/class_weights.json`**
   - Train frequencies: [0.046, ..., 0.0005]
   - Ratio: 100× (matches IF=100)

---

**Status**: ✅ All fixes applied, ready for paper submission  
**Last updated**: 2025-10-21
