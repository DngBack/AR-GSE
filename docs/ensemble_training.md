# AR-GSE Ensemble Training: Balance & Constrained Plugin Algorithms

## 📖 Tổng quan

Đây là giai đoạn cuối cùng và quan trọng nhất của hệ thống AR-GSE, thực hiện tối ưu hóa ensemble thông qua 2 thuật toán plugin:

1. **GSE-Balanced Plugin** (`gse_balanced_plugin.py`): Tối ưu hóa theo balanced/worst-case error
2. **GSE-Constrained Plugin** (`gse_constrained_plugin.py`): Tối ưu hóa với ràng buộc công bằng (fairness constraints)

Cả hai thuật toán đều dựa trên **primal-dual optimization** và **fixed-point updates** để tìm tham số tối ưu cho ensemble.

---

## 🎯 Mục tiêu chính

### Input chính:
- **Expert logits** đã được train và calibrated
- **Pre-trained gating network** từ giai đoạn trước
- **Data splits**: tuneV (S1) và val_lt (S2)

### Output mục tiêu:
- **α* (alpha)**: per-group scaling parameters [K,]
- **μ* (mu)**: per-group bias parameters [K,]
- **t* (threshold)**: decision threshold (scalar hoặc per-group)
- **Optimized ensemble model** với hiệu suất tối ưu

### Metrics tối ưu:
- **Balanced Error**: (1/K) Σ e_k - lỗi trung bình các nhóm
- **Worst-case Error**: max_k e_k - lỗi tệ nhất
- **Coverage**: tỷ lệ samples được chấp nhận
- **Fairness**: công bằng giữa các nhóm

---

## 🔄 GSE-Balanced Plugin Algorithm

### 1. Kiến trúc tổng quan

```python
# File: gse_balanced_plugin.py
# Core optimization loop:
for outer_iter in range(T):
    1. Fit threshold t on S1 for current (α, μ)
    2. Compute errors e_k on S2
    3. Update α via fixed-point method
    4. Update μ via grid search over λ
    5. Apply EMA smoothing and projection
```

### 2. Biến đầu vào chính

```python
CONFIG = {
    'balanced_params': {
        'T': 50,                    # Outer iterations
        'objective': 'balanced',    # 'balanced', 'worst', or 'hybrid'
        'lambda_grid': [-2.0..2.0], # Grid for μ optimization
        'alpha_min': 0.75,          # α constraints
        'alpha_max': 1.35,
        'ema_alpha': 0.7,          # EMA coefficient for smoothing
        'beta_floor': 0.85,        # Lower bound for β updates
        'patience': 8,             # Early stopping
        'convergence_tol': 1e-4,   # Convergence threshold
        'adaptive_grid': True,     # Expand grid if needed
        'hybrid_weight': 0.3,      # Weight for worst error in hybrid
    }
}
```

### 3. Input Data Flow

```python
def main():
    # 1. Load expert logits
    S1_loader, S2_loader = load_data_from_logits(CONFIG)
    # S1 (tuneV): ~2500 samples for threshold fitting  
    # S2 (val_lt): ~2500 samples for optimization
    
    # 2. Setup grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=20)
    # Tạo 2 nhóm: head (>20 samples) vs tail (≤20 samples)
    
    # 3. Load pre-trained GSE model
    model = AR_GSE(num_experts=3, num_classes=100, num_groups=2, gating_dim=24)
    # Load gating weights từ giai đoạn training trước
    
    # 4. Cache mixture posteriors
    eta_S1, y_S1 = cache_eta_mix(model, S1_loader, class_to_group)
    eta_S2, y_S2 = cache_eta_mix(model, S2_loader, class_to_group)
    # η̃(x) = Σ_e w^(e)(x) * p^(e)(y|x): mixture posterior
```

### 4. Core Algorithm Components

#### A. Cache Mixture Posteriors
```python
@torch.no_grad()
def cache_eta_mix(gse_model, loader, class_to_group):
    """
    Tính mixture posterior η̃(x) từ expert outputs và gating weights.
    
    Input:
        - logits: [B, E, C] từ E=3 experts
        - gating_net: pre-trained gating network
    
    Output:
        - eta: [N, C] mixture posteriors
        - labels: [N] ground truth
    
    Formula:
        η̃_y(x) = Σ_e w^(e)(x) * p^(e)(y|x)
        where:
        - w^(e)(x): gating weights [B, E]
        - p^(e)(y|x): expert posteriors [B, E, C]
    """
```

#### B. Margin Computation
```python
def compute_raw_margin(eta, alpha, mu, class_to_group):
    """
    Tính raw margin score cho selective prediction.
    
    Formula:
        margin = max_y α_{g(y)} * η̃_y - Σ_y (1/α_{g(y)} - μ_{g(y)}) * η̃_y
    
    Input:
        - eta: [N, C] mixture posteriors
        - alpha: [K] per-group scaling
        - mu: [K] per-group bias
        
    Output:
        - raw_margin: [N] margin scores
    """
    score = (alpha[class_to_group] * eta).max(dim=1).values
    coeff = 1.0 / alpha[class_to_group] - mu[class_to_group]  
    threshold_term = (coeff.unsqueeze(0) * eta).sum(dim=1)
    return score - threshold_term
```

#### C. Error Computation
```python
def balanced_error_on_S(eta, y, alpha, mu, t, class_to_group, K):
    """
    Tính balanced error trên split S.
    
    Steps:
    1. Compute margins và acceptance
    2. Tính per-group error rates
    3. Return balanced error = (1/K) Σ e_k
    
    Formula:
        e_k = 1 - (correct_accepted_k / total_accepted_k)
        balanced_error = (1/K) Σ_k e_k
    """
```

#### D. Parameter Updates

##### Alpha Update (Fixed-Point Method)
```python
def update_alpha_fixed_point(eta_S1, y_S1, alpha, mu, t, class_to_group, K, config):
    """
    Cập nhật α theo fixed-point iteration.
    
    Algorithm:
    1. Compute current acceptance rates per group
    2. Update: α_k^{new} = acceptance_rate_k + regularization
    3. Apply EMA smoothing: α = ema_alpha * α_old + (1-ema_alpha) * α_new
    4. Project to valid range [α_min, α_max]
    
    Input variables:
        - eta_S1: [N1, C] posteriors on tuning split
        - alpha: [K] current scaling parameters
        - mu: [K] current bias parameters
        
    Output:
        - alpha_new: [K] updated parameters
    """
    raw_margins = compute_raw_margin(eta_S1, alpha, mu, class_to_group)
    accepted = (raw_margins >= t)
    
    alpha_new = torch.zeros(K)
    for k in range(K):
        group_mask = (class_to_group[y_S1] == k)
        if group_mask.sum() > 0:
            acceptance_rate = (accepted & group_mask).sum().float() / group_mask.sum()
            alpha_new[k] = acceptance_rate + config['regularization']
    
    # EMA smoothing
    alpha = config['ema_alpha'] * alpha + (1 - config['ema_alpha']) * alpha_new
    return project_alpha(alpha, config['alpha_min'], config['alpha_max'])
```

##### Mu Update (Grid Search)
```python
def update_mu_grid_search(eta_S2, y_S2, alpha, t, class_to_group, K, config):
    """
    Tìm μ tối ưu thông qua grid search.
    
    Algorithm:
    1. For each λ in lambda_grid:
        - Set μ = [+λ/2, -λ/2] (for K=2)
        - Compute objective on S2
        - Track best μ
    2. Return μ with lowest objective
    
    Variables:
        - lambda_grid: [-2.0, -1.9, ..., 1.9, 2.0]
        - objective_type: 'balanced', 'worst', or 'hybrid'
    """
```

### 5. Objective Functions

```python
# Balanced Error (default)
objective = e_k.mean()

# Worst-case Error  
objective = e_k.max()

# Hybrid (weighted combination)
balanced_error = e_k.mean()
worst_error = e_k.max() 
objective = (1 - hybrid_weight) * balanced_error + hybrid_weight * worst_error
```

### 6. Output Results

```python
# Final optimized parameters
alpha_star = [1.0234, 0.8765]  # Per-group scaling
mu_star = [0.1234, -0.1234]    # Per-group bias  
t_star = 0.5678                # Decision threshold

# Performance metrics
balanced_error = 0.1234        # (1/K) Σ e_k
worst_error = 0.2345          # max_k e_k
coverage = 0.8567             # Acceptance rate

# Checkpoints saved to:
# ./checkpoints/argse_balance/cifar100_lt_if100/gse_balanced_plugin.ckpt
```

---

## 🔒 GSE-Constrained Plugin Algorithm

### 1. Kiến trúc Lagrangian Optimization

```python
# File: gse_constrained_plugin.py
# Lagrangian formulation:
L(α, μ, t, λ, ν) = (1/K) Σ e_k + λ(τ - (1/K) Σ cov_k) + Σ ν_k(e_k - δ)

# Where:
# - e_k: per-group error when accepting
# - cov_k: per-group coverage  
# - τ: minimum average coverage constraint (e.g., 0.65)
# - δ: maximum per-group error constraint  
# - λ: coverage multiplier (dual variable)
# - ν_k: fairness multipliers per group (dual variables)
```

### 2. Biến đầu vào chính

```python
CONFIG = {
    'constrained_params': {
        'T': 50,                   # Outer iterations
        'tau': 0.65,              # Coverage constraint: cov ≥ τ
        'delta_multiplier': 1.3,   # δ = delta_multiplier × avg_error
        'eta_dual': 0.05,         # Dual step size
        'eta_primal': 0.01,       # Primal step size
        'lambda_grid': [-2.0..2.0], # Grid for primal updates
        'warmup_iters': 10,       # Warmup before enforcing constraints
        'adaptive_delta': True,    # Adaptively adjust δ
        'patience': 8,            # Early stopping
    }
}
```

### 3. Core Algorithm Flow

```python
def gse_constrained_plugin(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K, config):
    """
    Main constrained optimization với primal-dual updates.
    
    Input Variables:
        - eta_S1, y_S1: tuning split for threshold fitting
        - eta_S2, y_S2: validation split for optimization  
        - class_to_group: [C] mapping classes to groups
        - K: number of groups (typically 2)
        
    Algorithm Flow:
    1. Initialize primal (α, μ, t) and dual (λ, ν) variables
    2. For each outer iteration:
        a. Fit threshold t on S1
        b. Compute metrics (e_k, cov_k) on S2  
        c. Update primal variables (α, μ) via grid search
        d. Update dual variables (λ, ν) via gradient ascent
        e. Check convergence
        
    Output:
        - (α*, μ*, t*): optimal primal variables
        - history: optimization trajectory
    """
```

### 4. Constraint Handling

#### A. Coverage Constraint
```python
# Constraint: average coverage ≥ τ
coverage_constraint = tau - cov_k.mean()  # ≤ 0 when satisfied

# Dual update: λ ← max(0, λ + η_dual × violation)
lambda_cov = max(0.0, lambda_cov + eta_dual * coverage_constraint)
```

#### B. Fairness Constraint  
```python
# Constraint: per-group error ≤ δ
fairness_violations = e_k - delta  # ≤ 0 when satisfied

# Dual update: ν_k ← max(0, ν_k + η_dual × violation_k)  
nu = torch.clamp(nu + eta_dual * fairness_violations, min=0.0)

# Adaptive δ adjustment
if adaptive_delta and current_avg_error > 0:
    delta = max(delta * 0.95, delta_multiplier * current_avg_error)
```

### 5. Primal-Dual Updates

#### Primal Update (α, μ)
```python
# Grid search over μ candidates
for lam in lambda_grid:
    mu_candidate = torch.tensor([+lam/2, -lam/2])  # For K=2
    
    # Fixed-point updates for α given μ
    alpha_candidate = alpha.clone()
    for fp_iter in range(3):
        # Compute acceptance rates
        raw_margins = compute_raw_margin(eta_S1, alpha_candidate, mu_candidate, class_to_group)
        accepted = (raw_margins >= t)
        
        # Update α based on acceptance rates
        for k in range(K):
            group_mask = (class_to_group[y_S1] == k)
            acceptance_rate = (accepted & group_mask).sum() / group_mask.sum()
            alpha_new[k] = acceptance_rate + regularization
        
        # EMA smoothing
        alpha_candidate = 0.7 * alpha_candidate + 0.3 * alpha_new
        alpha_candidate = project_alpha(alpha_candidate)
    
    # Evaluate Lagrangian objective
    e_k, cov_k, _, _ = compute_group_metrics(eta_S2, y_S2, alpha_candidate, mu_candidate, t)
    
    lagrangian = e_k.mean()  # Primal objective
    if outer_iter >= warmup_iters:
        lagrangian += lambda_cov * (tau - cov_k.mean())  # Coverage penalty
        lagrangian += (nu * torch.clamp(e_k - delta, min=0)).sum()  # Fairness penalty
    
    # Track best candidate
    if lagrangian < best_lagrangian:
        best_alpha, best_mu = alpha_candidate, mu_candidate
```

#### Dual Update (λ, ν)
```python
# Coverage multiplier update
lambda_cov = max(0.0, lambda_cov + eta_dual * coverage_violation)

# Per-group fairness multiplier update  
nu = torch.clamp(nu + eta_dual * fairness_violations, min=0.0)
```

### 6. Output Analysis

```python
# Final Results Example:
α* = [1.0145, 0.9823]     # Balanced scaling between groups
μ* = [0.0234, -0.0234]    # Small bias correction
t* = 0.4567               # Threshold for τ=0.65 coverage

# Metrics achieved:
balanced_error = 0.0987   # Primary objective
worst_error = 0.1234     # Worst group performance  
coverage = 0.653         # Satisfies τ ≥ 0.65
per_group_errors = [0.0876, 0.1098]  # Both ≤ δ
per_group_coverage = [0.645, 0.661]  # Balanced coverage

# Constraint satisfaction:
coverage_satisfied = True    # 0.653 ≥ 0.65 ✓
fairness_satisfied = True   # All e_k ≤ δ ✓
```

---

## 🔄 So sánh hai thuật toán

| Aspect | GSE-Balanced | GSE-Constrained |
|--------|--------------|-----------------|
| **Objective** | Minimize balanced/worst error | Minimize error subject to constraints |
| **Constraints** | None (unconstrained) | Coverage ≥ τ, Error ≤ δ |  
| **Optimization** | Direct minimization | Lagrangian dual method |
| **Complexity** | Simpler, faster | More complex, principled |
| **Use case** | General optimization | Fairness-aware deployment |
| **Parameters** | (α, μ, t) | (α, μ, t) + (λ, ν) |

---

## 📊 Evaluation và Metrics

### 1. Performance Metrics
```python
def evaluate_final_performance(eta_S2, y_S2, alpha_star, mu_star, t_star, class_to_group, K):
    """
    Đánh giá hiệu suất cuối cùng trên validation set.
    
    Computed Metrics:
    - Balanced Error: (1/K) Σ e_k  
    - Worst-case Error: max_k e_k
    - Coverage: fraction of samples accepted
    - Per-group Error: e_k for each group k
    - Per-group Coverage: cov_k for each group k
    - Accuracy on accepted: correct predictions / accepted predictions
    """
```

### 2. Convergence Analysis
```python
# Tracking optimization history:
history = {
    'balanced_error': [...],     # Balanced error per iteration
    'worst_error': [...],        # Worst error per iteration  
    'coverage': [...],           # Coverage per iteration
    'lagrangian': [...],         # Objective per iteration (constrained only)
    'lambda_cov': [...],         # Coverage multiplier (constrained only)
    'nu': [...],                # Fairness multipliers (constrained only)
    'alpha': [...],             # Alpha evolution
    'mu': [...]                 # Mu evolution
}
```

---

## 🚀 Usage Examples

### 1. Chạy GSE-Balanced Plugin
```bash
cd c:\Users\Admin\Documents\GitHub\AR-GSE
python -m src.train.gse_balanced_plugin

# Output:
# ✅ Loaded S1 (tuneV): 10 batches  
# ✅ Loaded S2 (val_lt): 10 batches
# ✅ Groups: 34 head classes, 66 tail classes
# ✅ Loaded pre-trained gating
# ✅ Cached η̃_S1: [2500, 100], y_S1: [2500]
# ✅ Cached η̃_S2: [2500, 100], y_S2: [2500]
# 
# === GSE Balanced Plugin ===
# Objective: balanced, T=50, EMA=0.70
# [1 ] bal=0.2134, worst=0.3456, obj=0.2134
# [2 ] bal=0.1987, worst=0.3201, obj=0.1987  
# ...
# [23] bal=0.0987, worst=0.1234, obj=0.0987
# Early stopping at iteration 23
# 
# α* = [1.0145, 0.9823]
# μ* = [0.0234, -0.0234]  
# t* = 0.4567
# 💾 Saved checkpoint to ./checkpoints/argse_balance/cifar100_lt_if100/gse_balanced_plugin.ckpt
```

### 2. Chạy GSE-Constrained Plugin
```bash 
python -m src.train.gse_constrained_plugin

# Output:
# === GSE Constrained Plugin ===
# Coverage constraint: τ ≥ 0.65
# Outer iterations: 50
# Initial δ = 0.267 (1.3× avg error)
# 
# [1 ] bal=0.2134, worst=0.3456, cov=0.580, L=0.2134
# [2 ] bal=0.1987, worst=0.3201, cov=0.623, L=0.1987
# ...  
# [15] bal=0.0987, worst=0.1234, cov=0.653, L=0.0956, λ=0.234, ν_max=0.123
# Early stopping at iteration 15
#
# ✅ Final Results:
# α* = [1.0145, 0.9823]
# μ* = [0.0234, -0.0234]
# Coverage = 0.653 (≥ 0.65 ✓)
# Per-group errors: [0.0876, 0.1098] (all ≤ 0.267 ✓)
# 💾 Saved checkpoint + plots
```

---

## 🔧 Configuration tuning

### 1. Key hyperparameters

#### GSE-Balanced:
- `T`: Số iterations (50 thường đủ)
- `ema_alpha`: EMA coefficient (0.7 cho stability)  
- `lambda_grid`: Range cho μ optimization ([-2, 2])
- `alpha_min/max`: Constraints cho α ([0.75, 1.35])
- `objective`: 'balanced' vs 'worst' vs 'hybrid'

#### GSE-Constrained:
- `tau`: Coverage constraint (0.65 = 65% minimum)
- `delta_multiplier`: Fairness constraint (1.3× avg error) 
- `eta_dual`: Dual step size (0.05 cho stability)
- `warmup_iters`: Iterations trước khi áp dụng constraints (10)

### 2. Troubleshooting

#### Common issues:
1. **Convergence problems**: Giảm eta_dual/eta_primal
2. **Unstable α updates**: Tăng ema_alpha 
3. **Coverage too low**: Giảm tau hoặc tăng delta_multiplier
4. **Poor fairness**: Điều chỉnh lambda_grid range

---

## 📁 File Structure

```
checkpoints/argse_balance/cifar100_lt_if100/
├── gse_balanced_plugin.ckpt      # Balanced plugin results
└── optimization_history.png      # Training curves

checkpoints/argse_constrained_plugin/cifar100_lt_if100/  
├── gse_constrained_plugin.ckpt   # Constrained plugin results
└── constrained_optimization_history.png  # Constraint satisfaction curves

outputs/logits/cifar100_lt_if100/  # Input expert logits
├── ce_baseline/
├── logitadjust_baseline/  
└── balsoftmax_baseline/
```

---

## 🎯 Summary

**GSE Ensemble Training** là giai đoạn cuối cùng kết hợp tất cả components của AR-GSE:

1. **Input**: Expert logits + Pre-trained gating + Data splits
2. **Process**: Plugin optimization với primal-dual methods  
3. **Output**: Optimal ensemble parameters (α*, μ*, t*)
4. **Result**: High-performance selective prediction system

Hai thuật toán plugin cung cấp flexibility:
- **Balanced**: Đơn giản, tối ưu performance trực tiếp
- **Constrained**: Phức tạp hơn nhưng đảm bảo fairness constraints

Kết quả cuối cùng là một AR-GSE model hoàn chỉnh có thể deploy cho selective prediction trên CIFAR-100-LT với hiệu suất cao và công bằng giữa các nhóm head/tail classes.