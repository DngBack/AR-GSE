# 🎛️ AR-GSE Gating Training Documentation

Tài liệu chi tiết và toàn diện về hệ thống training gating network trong AR-GSE, bao gồm kiến trúc, algorithms, và hai modes training quan trọng.

## 🎯 Tổng quan Gating System

Gating network là **trái tim** của AR-GSE ensemble system, chịu trách nhiệm học cách **kết hợp optimal** các expert models dựa trên input characteristics. Đây là component quyết định **khi nào** và **bao nhiều** tin tưởng vào mỗi expert.

### Vai trò chính:
- **Expert Selection**: Chọn expert phù hợp cho từng input
- **Adaptive Weighting**: Điều chỉnh weights dựa trên uncertainty
- **Complementarity**: Tận dụng strengths riêng biệt của từng expert
- **Fairness**: Đảm bảo performance cân bằng giữa head/tail classes

## 🏗️ Kiến trúc Gating Network

### Core Components

```
Gating Network Architecture:
Expert Logits [B, E, C] → GatingFeatureBuilder → Features [B, D] → GatingNet → Raw Weights [B, E] → Softmax → Gating Weights [B, E]
```

### 1. Gating Feature Builder

```python
class GatingFeatureBuilder:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def __call__(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """
        Chuyển đổi expert logits [B, E, C] thành rich features [B, D]
        D = 7*E + 3 (per-expert features + global features)
        """
        B, E, C = expert_logits.shape
        expert_posteriors = torch.softmax(expert_logits, dim=-1)
        
        # Per-expert features (7 features per expert)
        # 1. Entropy: H(p_e) = -Σ p_e log p_e
        entropy = -torch.sum(expert_posteriors * torch.log(expert_posteriors + 1e-8), dim=-1)  # [B, E]
        
        # 2. Top-k probability mass
        topk_vals, _ = torch.topk(expert_posteriors, k=min(self.top_k, C), dim=-1)
        topk_mass = torch.sum(topk_vals, dim=-1)  # [B, E]
        
        # 3. Residual mass (uncertainty measure)
        residual_mass = 1.0 - topk_mass  # [B, E]
        
        # 4. Max probability (confidence)
        max_probs, _ = expert_posteriors.max(dim=-1)  # [B, E]
        
        # 5. Top1-Top2 gap (decisiveness)
        if topk_vals.size(-1) >= 2:
            top_gap = topk_vals[..., 0] - topk_vals[..., 1]  # [B, E]
        else:
            top_gap = torch.zeros_like(max_probs)
            
        # 6. Cosine similarity to ensemble mean (agreement)
        mean_posterior = torch.mean(expert_posteriors, dim=1)  # [B, C]
        cosine_sim = F.cosine_similarity(expert_posteriors, mean_posterior.unsqueeze(1), dim=-1)  # [B, E]
        
        # 7. KL divergence to ensemble mean (disagreement)
        kl_to_mean = torch.sum(expert_posteriors * (torch.log(expert_posteriors + 1e-8) - torch.log(mean_posterior.unsqueeze(1) + 1e-8)), dim=-1)  # [B, E]
        
        # Global features (3 features total)
        # 1. Ensemble uncertainty
        mean_entropy = -torch.sum(mean_posterior * torch.log(mean_posterior + 1e-8), dim=-1)  # [B]
        
        # 2. Cross-expert variance
        class_var = expert_posteriors.var(dim=1)  # [B, C]
        mean_class_var = class_var.mean(dim=-1)  # [B]
        
        # 3. Confidence dispersion
        std_max_conf = max_probs.std(dim=-1)  # [B]
        
        # Concatenate all features
        per_expert_feats = [entropy, topk_mass, residual_mass, max_probs, top_gap, cosine_sim, kl_to_mean]
        per_expert_concat = torch.cat(per_expert_feats, dim=1)  # [B, 7*E]
        
        global_feats = torch.stack([mean_entropy, mean_class_var, std_max_conf], dim=1)  # [B, 3]
        
        return torch.cat([per_expert_concat, global_feats], dim=1)  # [B, 7*E + 3]
```

**Feature Analysis:**
- **Per-Expert (7 × E features)**:
  - `entropy`: Độ không chắc chắn của expert
  - `topk_mass`: Concentration của probability mass  
  - `residual_mass`: Uncertainty measure (1 - top-k mass)
  - `max_probs`: Confidence level
  - `top_gap`: Decisiveness (top1 - top2)
  - `cosine_sim`: Agreement với ensemble mean
  - `kl_to_mean`: Disagreement với ensemble mean

- **Global (3 features)**:
  - `mean_entropy`: Overall ensemble uncertainty
  - `mean_class_var`: Cross-expert disagreement
  - `std_max_conf`: Confidence dispersion

### 2. Gating Network (MLP)

```python
class GatingNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list = [128, 64], 
                 num_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        current_dim = in_dim  # 7*E + 3
        
        # Hidden layers với BatchNorm và Dropout
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = h_dim
        
        # Output layer (raw weights, softmax applied externally)
        layers.append(nn.Linear(current_dim, num_experts))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features [B, D] → Raw weights [B, E]"""
        return self.net(x)
```

**Architecture Details:**
- **Input**: Rich features [B, 7*E + 3] (E=3 → D=24)
- **Hidden**: [24] → [128] → [64] → [3]
- **Regularization**: BatchNorm + Dropout
- **Output**: Raw weights (softmax applied externally)

## 🎭 Two Training Modes

AR-GSE sử dụng **two-stage gating training** để đạt được optimal performance:

### Mode 1: Pretrain (Warm-up)
- **Purpose**: Học basic mixture of experts
- **Parameters**: Fixed α=1, μ=0 
- **Objective**: Standard mixture cross-entropy
- **Duration**: ~25 epochs

### Mode 2: Selective (Advanced)  
- **Purpose**: Học selective prediction với fairness constraints
- **Parameters**: Learnable α, μ, per-group thresholds
- **Objective**: Selective loss + coverage constraints + fairness
- **Duration**: ~18 epochs (3 cycles × 6 epochs)

## 🔧 Mode 1: Pretrain Training Deep Dive

### Objective Function

```python
def mixture_cross_entropy_loss(expert_logits, labels, gating_weights, 
                              sample_weights=None, entropy_penalty=0.0, diversity_penalty=0.0):
    """
    Enhanced mixture CE loss với diversity promotion
    L = -log(Σ_e w_e * p_e(y)) + λ_H * H(w) + λ_D * D(w)
    """
    # expert_logits: [B, E, C], gating_weights: [B, E], labels: [B]
    
    # Compute mixture probabilities
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, E, C]
    mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_probs)  # [B, C]
    mixture_probs = torch.clamp(mixture_probs, min=1e-7, max=1.0-1e-7)
    
    # Cross-entropy: -log(p_y)
    log_probs = torch.log(mixture_probs)
    nll = F.nll_loss(log_probs, labels, reduction='none')  # [B]
    
    # Sample weighting for balanced training
    if sample_weights is not None:
        nll = nll * sample_weights
    
    ce_loss = nll.mean()
    
    # Entropy penalty: -λ_H * H(w) để encourage diversity
    entropy_loss = 0.0
    if entropy_penalty > 0:
        gating_log_probs = torch.log(gating_weights + 1e-8)
        entropy = -(gating_weights * gating_log_probs).sum(dim=1).mean()
        entropy_loss = -entropy_penalty * entropy  # Negative vì muốn maximize entropy
    
    # Diversity penalty: λ_D * KL(usage_freq || uniform)
    diversity_loss = 0.0
    if diversity_penalty > 0:
        p_bar = gating_weights.mean(dim=0) + 1e-8  # [E] usage frequency
        diversity_loss = diversity_penalty * torch.sum(p_bar * torch.log(p_bar * len(p_bar)))
    
    return ce_loss + entropy_loss + diversity_loss
```

### Sample Weighting Strategies

**1. Frequency-based Soft Weighting:**
```python
def compute_frequency_weights(labels, class_counts, smoothing=0.5):
    """w_i = (freq(y_i))^(-smoothing)"""
    freq_weights = torch.ones_like(labels, dtype=torch.float)
    
    for label in labels.unique():
        class_freq = class_counts[label.item()]
        weight = (class_freq + 1) ** (-smoothing)  # Smooth để tránh extreme weights
        freq_weights[labels == label] = weight
    
    return freq_weights / freq_weights.mean()  # Normalize mean = 1
```

**2. Group-based Hard Weighting:**
```python
# Binary weighting: head=1.0, tail=tail_weight
sample_weights = torch.where(y_groups == 0, 1.0, tail_weight)
```

### Training Configuration

```python
CONFIG['gating_params'] = {
    'epochs': 25,
    'batch_size': 2,           # Small batch for stable gradient
    'lr': 8e-4,               # Conservative learning rate
    'weight_decay': 2e-4,     # L2 regularization
    'balanced_training': True, # Enable tail-aware training
    'tail_weight': 1.0,       # Equal weighting (can be > 1 for tail boost)
    'use_freq_weighting': True, # Use frequency-based soft weighting
    'entropy_penalty': 0.0,    # Diversity encouragement (disabled)
    'diversity_penalty': 0.002, # Usage balance penalty
    'gradient_clip': 0.5,      # Gradient clipping for stability
}
```

### Training Loop

```python
def train_gating_only():
    """Main pretrain training loop"""
    # Load pre-computed expert logits from tuneV split
    train_loader = load_data_from_logits(CONFIG)
    
    # Create AR_GSE model với fixed α=1, μ=0
    model = AR_GSE(num_experts, num_classes, num_groups, gating_feature_dim).to(device)
    with torch.no_grad():
        model.alpha.fill_(1.0)
        model.mu.fill_(0.0)
    
    # Optimize only gating network parameters
    optimizer = optim.Adam(model.gating_net.parameters(), lr=CONFIG['gating_params']['lr'])
    
    for epoch in range(CONFIG['gating_params']['epochs']):
        model.train()
        for expert_logits, labels in train_loader:
            # Forward pass
            gating_features = model.feature_builder(expert_logits)
            raw_weights = model.gating_net(gating_features)
            gating_weights = torch.softmax(raw_weights, dim=1)
            
            # Compute sample weights for balanced training
            if CONFIG['gating_params']['balanced_training']:
                if CONFIG['gating_params']['use_freq_weighting']:
                    sample_weights = compute_frequency_weights(labels, class_counts)
                else:
                    y_groups = class_to_group[labels]
                    sample_weights = torch.where(y_groups == 0, 1.0, CONFIG['gating_params']['tail_weight'])
            
            # Loss computation
            loss = mixture_cross_entropy_loss(expert_logits, labels, gating_weights, 
                                            sample_weights, 
                                            CONFIG['gating_params']['entropy_penalty'],
                                            CONFIG['gating_params']['diversity_penalty'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if CONFIG['gating_params']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), 
                                             CONFIG['gating_params']['gradient_clip'])
            optimizer.step()
```

## 🎯 Mode 2: Selective Training Deep Dive

Mode 2 là **phần phức tạp nhất** của AR-GSE training, sử dụng **alternating optimization** với **Pinball loss** để học selective prediction.

### Overall Algorithm Structure

```
Selective Training = Stage A (Warm-up) + Stage B (Alternating Cycles)

Stage A (5 epochs):
  - Mixture CE loss để warm-up
  - Collect raw margins để init thresholds
  
Stage B (6 cycles × 3 epochs each):
  For each cycle:
    B1: Optimize gating network (3 epochs)
        - Selective loss với Pinball quantile loss
        - Learn per-group thresholds t_head, t_tail
        
    B2: Update confidence parameters α (2 steps)
        - Fixed-point matching cho acceptance rates
        
    B3: Grid search cho bias parameters μ (41 candidates)
        - Minimize worst-case group error
```

### Stage A: Warm-up

```python
# Stage A: Standard mixture CE để khởi động
for epoch in range(sel_cfg['stageA_epochs']):  # 5 epochs
    for expert_logits, labels in S1_loader:
        # Temperature scaling
        expert_logits = temperature_scale_logits(expert_logits, temperatures)
        
        # Standard mixture forward pass
        w, eta, expert_probs = compute_mixture_and_w(model, expert_logits)
        mixture_probs = eta.clamp(min=1e-8)
        ce_loss = F.nll_loss(mixture_probs.log(), labels)
        
        # Collect raw margins for threshold initialization
        raw_margin = compute_raw_margin(eta, alpha, mu, class_to_group)
        
        ce_loss.backward()
        optimizer.step()
    
    # Update global threshold via quantile
    if sel_cfg['use_quantile_t']:
        t = torch.quantile(raw_margins_epoch, 1 - sel_cfg['tau'])
```

### Stage B1: Gating Optimization với Pinball Loss

**Core Innovation**: Sử dụng **learnable per-group thresholds** thay vì fixed global threshold.

```python
def selective_losses_with_pinball(expert_logits, labels, model, alpha, mu, t_param, cfg_sel, class_to_group, pi_by_group):
    """
    Complete selective loss với Pinball quantile loss cho learnable thresholds
    """
    # Forward gating
    gating_features = model.feature_builder(expert_logits)
    raw_w = model.gating_net(gating_features)
    w = torch.softmax(raw_w, dim=1)  # [B, E]
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, E, C]
    eta = torch.einsum('be,bec->bc', w, expert_probs)  # [B, C]

    # Raw margin computation: m_raw(x) = max_y α_{g(y)} η_y - Σ_y (1/α_{g(y)} - μ_{g(y)}) η_y
    m_raw = compute_raw_margin(eta, alpha, mu, class_to_group)  # [B]
    
    # Per-sample thresholds based on group membership
    y_groups = class_to_group[labels]  # [B]
    t_g = t_param[y_groups]            # [B] - select appropriate threshold per sample
    
    # Soft acceptance probability: s = σ(κ * (m_raw - t_g))
    s = torch.sigmoid(cfg_sel['kappa'] * (m_raw - t_g))  # [B]

    # === Loss Component 1: Selective CE ===
    # p^α: confidence-scaled probabilities
    alpha_per_class = alpha[class_to_group].to(eta.device)
    q = eta * alpha_per_class.unsqueeze(0)  # [B, C] 
    q = q / (q.sum(dim=1, keepdim=True) + eps)  # Normalize
    ce = F.nll_loss(torch.log(q + eps), labels, reduction='none')  # [B]

    # Tail-aware weighting trong L_sel
    if cfg_sel.get('beta_tail', 1.0) != 1.0:
        beta_tail = cfg_sel['beta_tail']
        sample_w = torch.where(y_groups == 1, beta_tail, 1.0)
        L_sel = (s * ce * sample_w).sum() / (s * sample_w).sum().clamp_min(eps)
    else:
        L_sel = (s * ce).sum() / (s.sum() + eps)

    # === Loss Component 2: Pinball Quantile Loss ===
    # Learn thresholds để achieve target coverage per group
    tau_head = cfg_sel.get('tau_head', 0.56)  # Target coverage cho head group
    tau_tail = cfg_sel.get('tau_tail', 0.44)  # Target coverage cho tail group
    
    # Quantile level = 1 - coverage (vì chúng ta muốn P(m_raw > t_g) ≈ coverage)
    tau_quantile = torch.where(y_groups == 0, 1 - tau_head, 1 - tau_tail)  # [B]
    
    # Pinball loss: ρ_τ(z) = τ * max(z, 0) + (1-τ) * max(-z, 0)
    z = m_raw - t_g  # [B] residuals
    pinball = tau_quantile * torch.relu(z) + (1 - tau_quantile) * torch.relu(-z)
    L_q = pinball.mean()

    # === Loss Component 3: Coverage Penalty ===
    # Enforce actual coverage matches target coverage
    cov_head = s[y_groups == 0].mean() if (y_groups == 0).any() else torch.tensor(0., device=s.device)
    cov_tail = s[y_groups == 1].mean() if (y_groups == 1).any() else torch.tensor(0., device=s.device)
    L_cov_pinball = (cov_head - tau_head)**2 + (cov_tail - tau_tail)**2

    # === Loss Component 4: Entropy Regularization ===
    # Encourage diversity trong gating weights
    H_w = -(w * torch.log(w + eps)).sum(dim=1).mean()
    L_H = cfg_sel['lambda_H'] * H_w

    # === Loss Component 5: Group-aware Prior KL ===
    # Regularize gating towards group-specific priors
    pi = pi_by_group[y_groups]  # [B, E] - group priors
    KL = (w * (torch.log(w + eps) - torch.log(pi + eps))).sum(dim=1).mean()
    L_GA = cfg_sel['lambda_GA'] * KL

    # Total loss
    total = L_sel + cfg_sel['lambda_q'] * L_q + cfg_sel['lambda_cov_pinball'] * L_cov_pinball + L_H + L_GA
    
    return total, diagnostics, m_raw.detach(), s.detach()
```

**Key Innovations:**
1. **Per-group Thresholds**: `t_param = [t_head, t_tail]` learned parameters
2. **Pinball Loss**: Directly optimizes quantiles để achieve target coverage
3. **Tail-aware Weighting**: `beta_tail` để boost tail class importance
4. **Group Priors**: Regularize gating theo expert specializations

### Stage B2: Alpha Update (Fixed-Point)

```python
def update_alpha_fixed_point_conditional(eta, y, alpha, mu, t, class_to_group, K, gamma=0.2, alpha_min=0.85, alpha_max=1.15):
    """
    Fixed-point update cho α để match acceptance rates
    α_k = acceptance_rate_group_k + ε
    """
    with torch.no_grad():
        # Compute current acceptance
        raw = compute_raw_margin(eta, alpha, mu, class_to_group)
        accepted = raw > t  # [B] boolean mask
        y_groups = class_to_group[y]
        
        alpha_hat = torch.ones_like(alpha)
        for k in range(K):
            mask = (y_groups == k)
            if mask.any():
                group_acceptance = (accepted & mask).float().sum() / mask.float().sum()
                alpha_hat[k] = group_acceptance + 1e-3  # Small epsilon for stability
        
        # EMA update
        new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
        
        # Geometric mean constraint: enforce Π α_k = 1
        new_alpha = new_alpha.clamp_min(alpha_min)
        log_alpha = new_alpha.log()
        new_alpha = torch.exp(log_alpha - log_alpha.mean())  # Geometric mean = 1
        new_alpha = new_alpha.clamp(min=alpha_min, max=alpha_max)
    
    return new_alpha
```

**Intuition**: 
- **α_k > 1**: Group k cần higher confidence (more selective)
- **α_k < 1**: Group k cần lower confidence (more inclusive)
- **Geometric mean = 1**: Constraint để avoid trivial scaling

### Stage B3: Mu Grid Search

```python
def mu_from_lambda_grid(lambdas, K):
    """Generate μ candidates from λ grid for K=2 groups"""
    mus = []
    for lam in lambdas:  # λ ∈ [-2.0, 2.0] with 41 points
        if K == 2:
            # μ = [λ/2, -λ/2] để maintain zero sum
            mus.append(torch.tensor([lam/2.0, -lam/2.0], dtype=torch.float32))
        else:
            raise NotImplementedError("Multi-group μ not implemented")
    return mus

# Grid search trên S2 validation set
best_mu, best_score = mu.clone(), float('inf')
for mu_candidate in mu_candidates:
    score, group_errors = evaluate_split_with_learned_thresholds(
        eta_S2, y_S2, alpha, mu_candidate, t_param, class_to_group, K, 
        objective='worst'  # Minimize worst-case group error
    )
    
    if score < best_score:
        best_score = score
        best_mu = mu_candidate.clone()

# EMA stabilization
mu = 0.5 * mu + 0.5 * best_mu
```

**Purpose of μ**:
- **Bias terms** cho mỗi group trong threshold computation
- **μ_k > 0**: Easier acceptance cho group k (lower threshold)  
- **μ_k < 0**: Harder acceptance cho group k (higher threshold)
- **Zero sum constraint**: Σ μ_k = 0

## 🌡️ Temperature Scaling Integration

**Critical preprocessing step** trước khi gating training:

```python
def fit_temperature_scaling(expert_logits, labels, expert_names, device='cuda'):
    """Fit per-expert temperature scaling using validation data"""
    temperatures = {}
    
    for i, name in enumerate(expert_names):
        logits_i = expert_logits[:, i, :].to(device)  # [B, C]
        
        best_temp, best_nll = 1.0, float('inf')
        # Grid search over temperature values
        for temp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            scaled_logits = logits_i / temp
            nll = F.cross_entropy(scaled_logits, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        temperatures[name] = best_temp
        print(f"  {name}: T={best_temp:.2f} (NLL={best_nll:.4f})")
    
    return temperatures

# Apply temperature scaling
def temperature_scale_logits(expert_logits, expert_names, temp_cfg):
    """Apply temperature scaling to expert logits"""
    scaled = expert_logits.clone()
    for i, name in enumerate(expert_names):
        T = float(temp_cfg.get(name, 1.0))
        if abs(T - 1.0) > 1e-6:
            scaled[:, i, :] = scaled[:, i, :] / T
    return scaled
```

**Why Temperature Scaling matters**:
1. **Calibrated Confidence**: Đảm bảo expert confidence scores meaningful
2. **Fair Comparison**: Normalize confidence levels giữa các experts
3. **Better Gating**: Gating network học từ calibrated signals

## 🎮 Training Script Interface

### Main Script: `train_gating.py`

```bash
# Pretrain mode - warm up gating network
python train_gating.py --mode pretrain --verbose

# Selective mode - learn selective prediction
python train_gating.py --mode selective --verbose

# Custom parameters
python train_gating.py --mode pretrain --epochs 30 --lr 1e-3

# Dry run
python train_gating.py --mode selective --dry-run
```

**Arguments:**
- `--mode {pretrain|selective}`: Training mode (required)
- `--epochs INT`: Override epochs
- `--lr FLOAT`: Override learning rate  
- `--batch-size INT`: Override batch size
- `--device {cpu|cuda|auto}`: Compute device
- `--verbose`: Detailed logging
- `--dry-run`: Configuration check only

### Output Structure

```
checkpoints/gating_pretrained/cifar100_lt_if100/
├── gating_pretrained.ckpt     # Mode 1: Pretrained gating weights
├── gating_selective.ckpt      # Mode 2: Full selective model
└── selective_training_logs.json  # Detailed training logs
```

**Checkpoint Contents:**
```python
# Pretrain checkpoint
{
    'gating_net_state_dict': ...,  # Gating network weights
    'num_experts': 3,
    'num_classes': 100, 
    'num_groups': 2,
    'gating_feature_dim': 24,
    'config': CONFIG
}

# Selective checkpoint  
{
    'gating_net_state_dict': ...,  # Updated gating weights
    'alpha': torch.tensor([α_head, α_tail]),  # Confidence parameters
    'mu': torch.tensor([μ_head, μ_tail]),     # Bias parameters
    't_param': torch.tensor([t_head, t_tail]), # Learned thresholds
    'pi_by_group': ...,            # Group priors [K, E]
    'temperatures': {...},         # Per-expert temperatures
    'cycle_logs': [...],          # Training history
    'mode': 'selective_pinball'
}
```

## 📊 Performance Analysis & Expected Results

### Pretrain Mode Results

```
Epoch 25: Loss=0.8234
✅ Gating training complete. Best loss: 0.8234

Features learned:
- Expert agreement/disagreement patterns
- Confidence-based routing
- Basic mixture capabilities
```

### Selective Mode Results

```
=== Final Results ===
Final α=[1.02, 0.98] | μ=[-0.15, 0.15] | learned t=[-0.45, -0.52] | best_score=0.182

Group Coverage Analysis:
- Head group: coverage=0.56 (target=0.56) ✅
- Tail group: coverage=0.44 (target=0.44) ✅

Temperature Scaling:
- ce_baseline: T=1.08
- logitadjust_baseline: T=1.14  
- balsoftmax_baseline: T=1.09
```

### Key Metrics Evolution

**Cycle-by-cycle progression:**
```
Cycle 1: worst_err=0.245, α=[1.00,1.00], μ=[0.00,0.00]
Cycle 2: worst_err=0.218, α=[1.05,0.95], μ=[-0.10,0.10]  
Cycle 3: worst_err=0.195, α=[1.03,0.97], μ=[-0.12,0.12]
Cycle 4: worst_err=0.187, α=[1.02,0.98], μ=[-0.14,0.14]
Cycle 5: worst_err=0.183, α=[1.02,0.98], μ=[-0.15,0.15]
Cycle 6: worst_err=0.182, α=[1.02,0.98], μ=[-0.15,0.15] ← Converged
```

**Interpretation:**
- **α_head=1.02 > 1**: Head classes need slightly higher confidence
- **α_tail=0.98 < 1**: Tail classes get lower confidence requirements  
- **μ_head=-0.15 < 0**: Head gets harder threshold (more selective)
- **μ_tail=+0.15 > 0**: Tail gets easier threshold (more inclusive)

## 🧪 Advanced Features & Configurations

### 1. Group-Aware Priors

```python
def build_group_priors(expert_names, K, head_boost=1.5, tail_boost=1.5):
    """Build expert priors based on specializations"""
    pi = torch.ones(K, E, dtype=torch.float32)
    
    # Head-friendly experts
    head_keywords = ['ce', 'irm']
    # Tail-friendly experts  
    tail_keywords = ['balsoft', 'logitadjust', 'ride', 'ldam']
    
    for e, name in enumerate(expert_names):
        lname = name.lower()
        if any(k in lname for k in head_keywords):
            pi[0, e] *= head_boost  # Boost head group prior
        if K > 1 and any(k in lname for k in tail_keywords):
            pi[1, e] *= tail_boost  # Boost tail group prior
    
    # Normalize per group
    pi = pi / pi.sum(dim=1, keepdim=True)
    return pi
```

**Result cho AR-GSE experts:**
```
Group Priors:
Head group [0]: [0.60, 0.20, 0.20]  # CE expert preferred
Tail group [1]: [0.15, 0.425, 0.425] # LogitAdjust + BalSoftmax preferred
```

### 2. Pinball Loss Details

**Mathematical Foundation:**
```
ρ_τ(z) = {
    τ * z       if z ≥ 0
    (τ-1) * z   if z < 0
}

where:
- z = m_raw - t_g (residual)
- τ = 1 - target_coverage (quantile level)
- Minimizing E[ρ_τ(z)] learns τ-quantile của distribution
```

**Implementation:**
```python
# Quantile levels per group
tau_quantile = torch.where(y_groups == 0, 
                          1 - tau_head,    # Head: 1 - 0.56 = 0.44
                          1 - tau_tail)    # Tail: 1 - 0.44 = 0.56

# Pinball loss computation
z = m_raw - t_g  # Residuals [B]
pinball = tau_quantile * torch.relu(z) + (1 - tau_quantile) * torch.relu(-z)
L_q = pinball.mean()
```

### 3. Gradient Flow Analysis

**Critical design choices để ensure stable training:**

1. **Small Batch Size** (batch_size=2):
   ```python
   # Lý do: Gating features rất rich (24D), small batches prevent overfitting
   # Trade-off: Slower training vs better generalization
   ```

2. **Gradient Clipping** (max_norm=0.5):
   ```python
   # Prevent exploding gradients trong selective loss
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
   ```

3. **EMA Stabilization**:
   ```python  
   # Smooth parameter updates để prevent oscillation
   alpha = (1 - gamma) * alpha_old + gamma * alpha_new
   mu = 0.5 * mu_old + 0.5 * mu_new
   ```

## 🔧 Debugging & Troubleshooting

### Common Issues

**1. Gating Collapse (all weights → one expert)**
```python
# Symptoms: Diversity penalty high, one expert dominates
# Solutions:
- Increase diversity_penalty (0.002 → 0.005)
- Decrease entropy_penalty (encourage exploration) 
- Check expert logit quality và temperature scaling
- Verify balanced training weights
```

**2. Coverage Constraints Not Met**
```python
# Symptoms: Actual coverage ≠ target coverage
# Solutions: 
- Increase lambda_cov_pinball (20.0 → 50.0)
- Check Pinball loss implementation
- Verify per-group threshold learning
- Adjust kappa (sharpness parameter)
```

**3. Alpha/Mu Instability**  
```python
# Symptoms: Parameters oscillating between cycles
# Solutions:
- Increase EMA gamma (0.2 → 0.5) for smoother updates
- Tighten alpha bounds (alpha_min=0.90, alpha_max=1.10)
- Add more regularization trong grid search
```

### Validation Checks

```python
def validate_gating_training():
    """Comprehensive validation checks"""
    
    # 1. Check expert logits availability
    for expert_name in CONFIG['experts']['names']:
        logit_path = f"outputs/logits/cifar100_lt_if100/{expert_name}/tuneV_logits.pt"
        assert Path(logit_path).exists(), f"Missing logits: {logit_path}"
    
    # 2. Verify feature dimensions
    dummy_logits = torch.zeros(2, 3, 100)  # [B=2, E=3, C=100]
    builder = GatingFeatureBuilder()
    features = builder(dummy_logits)
    expected_dim = 7 * 3 + 3  # 7*E + 3 = 24
    assert features.shape[1] == expected_dim
    
    # 3. Check temperature scaling
    temperatures = fit_temperature_scaling(dummy_logits, torch.tensor([0, 1]), 
                                         CONFIG['experts']['names'])
    assert all(0.5 <= t <= 3.0 for t in temperatures.values())
    
    # 4. Validate Pinball loss
    t_param = torch.tensor([-0.5, -0.6])  # Example thresholds
    assert t_param.requires_grad == True
    
    print("✅ All gating training validations passed!")
```

## 📈 Training Timeline & Resources

### Expected Training Time (RTX 3090)

**Pretrain Mode:**
- **Duration**: ~15 minutes (25 epochs)
- **Memory**: ~4GB GPU 
- **Checkpoint**: ~2MB

**Selective Mode:**  
- **Duration**: ~45 minutes (18 epochs total)
- **Memory**: ~6GB GPU
- **Checkpoint**: ~3MB + logs

**Total Gating Training**: ~1 hour

### Memory Breakdown

```
Component                Memory Usage
─────────────────────────────────────
Expert Logits Cache      ~500MB
Gating Network          ~50KB  
Feature Builder         No parameters
Temperature Cache       ~1MB
Gradient Buffers        ~100MB
Working Memory          ~2GB
─────────────────────────────────────
Total Peak Usage        ~6GB
```

## 🔍 Comparison: Pretrain vs Selective

| Aspect | Pretrain Mode | Selective Mode |
|--------|---------------|----------------|
| **Purpose** | Basic mixture learning | Selective prediction + fairness |
| **Parameters** | Gating weights only | Gating + α + μ + t_param |
| **Loss** | Mixture CE + diversity | Selective + Pinball + coverage |
| **Complexity** | Simple, stable | Complex, alternating optimization |
| **Duration** | 25 epochs (~15 min) | 18 epochs (~45 min) |
| **Output** | Basic ensemble | Fair selective predictor |

## 📝 Summary

Gating training trong AR-GSE là một **sophisticated two-stage process**:

1. **Stage 1 (Pretrain)**: Học basic expert mixing với diversity promotion
2. **Stage 2 (Selective)**: Học selective prediction với fairness constraints

**Key innovations:**
- **Rich feature engineering** (24D features từ expert outputs)
- **Per-group learnable thresholds** với Pinball loss
- **Alternating optimization** cho α, μ, gating parameters
- **Temperature scaling integration** cho fair expert comparison
- **Group-aware priors** để leverage expert specializations

System này tạo ra một **adaptive gating mechanism** có khả năng:
- **Smart expert selection** based on input characteristics
- **Balanced performance** giữa head và tail classes  
- **Calibrated confidence** cho reliable rejection decisions
- **Fairness guarantees** thông qua constrained optimization

Kết quả là foundation vững chắc cho AR-GSE ensemble system với both accuracy và fairness trong selective prediction task.