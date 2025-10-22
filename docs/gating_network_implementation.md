# Gating Network cho MAP (Mixture-Aware Plug-in) với L2R

## 📚 Tổng quan

Implementation đầy đủ của **Gating Network** theo ý tưởng MAP + L2R cho long-tailed classification với selective rejection.

### Kiến trúc tổng thể

```
Expert Logits → Posteriors → Gating Network → Mixture Posterior → MAP Selector → Accept/Reject
   [N,E,C]        [N,E,C]      (features)        [N,C]         (margin-γU)
```

## 🎯 Mục tiêu

1. **Học trọng số mixture** để tối đa hóa likelihood của phân phối hỗn hợp
2. **Tận dụng diversity** của experts qua uncertainty/disagreement features
3. **Cân bằng load** giữa các experts (tránh collapse khi dùng Top-K)
4. **Cung cấp uncertainty signal** cho MAP margin: `m_MAP(x) = m_L2R(x) - γ·U(x)`

## 📁 Files đã triển khai

### 1. `src/models/gating_network_map.py`
**Gating Network Core - Architecture & Feature Extraction**

#### Components:

##### a) `UncertaintyDisagreementFeatures`
Tính các features uncertainty/disagreement từ expert posteriors:

**Per-expert features:**
- `expert_entropy`: H(p^(e)) - độ không chắc chắn của mỗi expert
- `expert_confidence`: max(p^(e)) - confidence cao nhất
- `expert_margin`: p_1 - p_2 (top-1 vs top-2 gap)

**Disagreement features:**
- `disagreement_ratio`: tỉ lệ experts bất đồng về top-1 prediction
- `mean_pairwise_kl`: trung bình KL divergence giữa các cặp experts

**Mixture features:**
- `mixture_entropy`: H(η̃) - entropy của phân phối hỗn hợp
- `posterior_variance`: variance của posteriors giữa experts
- `mutual_information`: I(Y; E | X) ≈ H(η̃) - mean(H(p^(e)))

**Lý do:**
> Deep Ensembles cho thấy các đại lượng này tương quan mạnh với sai số & overconfidence. Dùng làm feature giúp router "biết khi nào nên dè chừng".

##### b) `GatingFeatureExtractor`
Concat tất cả features thành vector đầu vào:
```
Input:  posteriors [B, E, C]
Output: features   [B, D] với D = E*C + 3*E + 5
```

##### c) `GatingMLP`
MLP nông (2-3 layers) với **LayerNorm** (stable hơn BatchNorm cho batch nhỏ):
```python
Input → Linear → LayerNorm → ReLU → Dropout → ... → Output
[B,D]                                                  [B,E]
```

##### d) Routing Strategies

**Dense Softmax (Jordan & Jacobs 1994):**
```python
w_φ(x) = softmax(g(x))  # All experts used
```

**Noisy Top-K (Shazeer et al. 2017):**
```python
1. Add Gaussian noise: g' = g + N(0, σ²)
2. Select Top-K experts
3. Softmax & renormalize over K
```

##### e) `GatingNetwork` (Complete)
Kết hợp tất cả:
```python
posteriors → feature_extractor → mlp → router → weights
  [B,E,C]         [B,D]           [B,E]  [B,E]
```

**Methods:**
- `forward(posteriors)`: returns `(weights, aux_outputs)`
- `get_mixture_posterior(posteriors, weights)`: tính η̃(x) = Σ w_e · p^(e)

##### f) `compute_uncertainty_for_map`
Tính U(x) cho MAP margin:
```
U(x) = a·H(w_φ) + b·Disagree({p^(e)}) + d·H(η̃)
```

---

### 2. `src/models/gating_losses.py`
**Loss Functions - Maximum Likelihood + Regularization**

#### Loss Components:

##### a) `MixtureNLLLoss` (CORE - Bắt buộc)
Maximum likelihood cho mixture model:
```python
L_mix = -E_{(x,y)} log(Σ_e w_φ(x)_e · p^(e)(y|x))
```

**Lý thuyết:**
- Proper scoring rule → khuyến khích calibration
- Nhất quán với EM algorithm cho HME (Jordan & Jacobs 1994)
- Khả vi → backprop end-to-end

##### b) `LoadBalancingLoss` (Cho Top-K routing)
Switch Transformer (Fedus et al. 2021):
```python
L_LB = α · N · Σ_i f_i · P_i

f_i = fraction of tokens routed to expert i (hard assignment)
P_i = average router probability for expert i
α ≈ 1e-2 (balancing coefficient)
```

**Intuition:**
- Expert i nhận nhiều tokens (f_i cao) VÀ có xác suất cao (P_i cao) → penalty
- Khuyến khích phân bổ đều → tránh collapse

##### c) `EntropyRegularizer` (Tùy chọn)
```python
L_H = -H(w) = Σ_e w_e log(w_e)  (maximize entropy)
hoặc
L_H = H(w)                       (minimize entropy)
```

Trong MAP: maximize entropy → khuyến khích diversity

##### d) `GatingLoss` (Combined)
```python
L_total = L_mix + λ_LB·L_LB + λ_H·L_H

Defaults:
- λ_LB = 1e-2  (Switch Transformer recommendation)
- λ_H = 0.01   (light regularization)
```

#### Utility Functions:

##### `compute_gating_metrics`
Monitor metrics:
- `gating_entropy`: H(w̄) - diversity của routing
- `load_std/max/min`: cân bằng expert usage
- `mixture_acc`: accuracy của mixture
- `effective_experts`: exp(H(mean_weights)) - số experts thực sự được dùng

---

### 3. `src/train/train_gating_map.py`
**Training Script - End-to-End Pipeline**

#### Workflow:

```
1. Load expert logits (đã calibrated)
   ├─ gating split: training
   └─ val split: validation (balanced)

2. Convert logits → posteriors

3. Initialize GatingNetwork
   ├─ Feature extractor
   ├─ MLP
   └─ Router (dense/top-k)

4. Training loop
   ├─ Forward: posteriors → weights
   ├─ Loss: L_mix + L_LB + L_H
   ├─ Backward + optimizer step
   └─ Validation every N epochs

5. Save best model (by balanced_acc)

6. Export checkpoints & training history
```

#### Configuration:

```python
CONFIG = {
    'gating': {
        # Architecture
        'hidden_dims': [256, 128],
        'dropout': 0.1,
        'activation': 'relu',
        
        # Routing
        'routing': 'dense',  # or 'top_k'
        'top_k': 2,
        
        # Training
        'epochs': 100,
        'batch_size': 128,
        'lr': 1e-3,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        
        # Loss
        'lambda_lb': 1e-2,
        'lambda_h': 0.01,
        
        # Long-tail
        'use_class_weights': True,
    }
}
```

#### Key Functions:

- `load_expert_logits()`: load từ disk
- `create_dataloaders()`: tạo train/val loaders
- `train_one_epoch()`: train loop với gradient clipping
- `validate()`: compute metrics trên val set
- `compute_group_accuracies()`: head/tail accuracy

---

### 4. `src/train/test_gating_map.py`
**Test Suite - Verification**

Tests:
1. ✅ Uncertainty feature extraction
2. ✅ Gating network forward pass
3. ✅ Loss computation
4. ✅ Metrics computation
5. ✅ U(x) for MAP
6. ✅ Gradient flow
7. ✅ Real expert logits (if available)

Run: `python src/train/test_gating_map.py`

---

## 🚀 Usage

### Step 1: Train Experts (Already done)
```bash
python src/train/train_expert.py
```

Output: expert logits tại `outputs/logits/cifar100_lt_if100/`

### Step 2: Test Gating Implementation
```bash
python src/train/test_gating_map.py
```

Expected: All tests pass ✅

### Step 3: Train Gating Network

**Dense routing (all experts):**
```bash
python src/train/train_gating_map.py --routing dense
```

**Top-K routing (sparse):**
```bash
python src/train/train_gating_map.py --routing top_k --top_k 2 --lambda_lb 1e-2
```

**Custom hyperparameters:**
```bash
python src/train/train_gating_map.py \
    --routing dense \
    --epochs 150 \
    --batch_size 64 \
    --lr 5e-4 \
    --lambda_lb 1e-2 \
    --lambda_h 0.02
```

### Step 4: Check Results

Checkpoints:
- `checkpoints/gating_map/cifar100_lt_if100/best_gating.pth`
- `checkpoints/gating_map/cifar100_lt_if100/final_gating.pth`

Training history:
- `results/gating_map/cifar100_lt_if100/training_history.json`

---

## 📊 Expected Behavior

### Gating Entropy
- **High entropy** (close to log(E)): experts được dùng đều → diversity cao
- **Low entropy**: chỉ một vài experts dominant → có thể collapse

### Load Balancing
- **Std of expert usage < 0.1**: cân bằng tốt
- **Max usage < 0.5**: không có expert nào quá dominant

### Mixture Accuracy
- **> Individual expert accs**: ensemble effect
- **Balanced acc** (head & tail gần nhau): fair performance

### Effective Experts
- Close to E: tất cả experts có ích
- Close to 1: collapse (chỉ 1 expert được dùng)

---

## 🔬 Lý thuyết nền tảng

### 1. Mixture of Experts (MoE)
```
p(y|x) = Σ_e w_φ(x)_e · p^(e)(y|x)

w_φ(x): gating function (input-dependent weights)
p^(e)(y|x): expert e's prediction
```

**Huấn luyện:** Maximum likelihood
```
max_φ E_{(x,y)} log p(y|x) = E log(Σ_e w_φ(x)_e · p^(e)(y|x))
```

**References:**
- Jordan & Jacobs (1994): Hierarchical Mixtures of Experts and EM
- Jacobs et al. (1991): Adaptive Mixtures of Local Experts

### 2. Load Balancing (Switch Transformer)
**Problem:** Top-K routing có thể collapse → chỉ dùng 1-2 experts

**Solution:** Auxiliary loss khuyến khích cân bằng
```
L_LB = α · N · Σ_i f_i · P_i
```

**References:**
- Fedus et al. (2021): Switch Transformers - Scaling to Trillion Parameters
- Shazeer et al. (2017): Outrageously Large Neural Networks (Sparsely-Gated MoE)

### 3. Uncertainty Quantification (Deep Ensembles)
**Key Insight:** Entropy/disagreement của ensemble tương quan với error

**Application:** Dùng làm features cho gating + U(x) cho MAP margin

**References:**
- Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty
- Gal & Ghahramani (2016): Dropout as Bayesian Approximation

### 4. MAP Margin với Uncertainty Penalty
```
m_MAP(x) = m_L2R(x) - γ·U(x)

m_L2R: margin từ L2R (tuyến tính trên η̃)
U(x): uncertainty từ mixture (H(w), disagreement, H(η̃))
γ: penalty coefficient
```

**Intuition:**
- Khi experts tranh cãi (high U) → margin giảm → dễ reject hơn
- Tận dụng "wisdom of crowd": nếu ensemble không chắc → nên thận trọng

---

## 🛠️ Customization

### Thêm features mới
Edit `UncertaintyDisagreementFeatures.compute()`:
```python
# Ví dụ: thêm max KL divergence
max_kl = torch.max(kl_to_mean, dim=-1)[0]  # [B]
features['max_expert_kl'] = max_kl
```

### Thay đổi routing strategy
Tạo class mới kế thừa `nn.Module`:
```python
class CustomRouter(nn.Module):
    def forward(self, logits):
        # Custom logic
        return weights
```

Update `GatingNetwork.__init__`:
```python
elif routing == 'custom':
    self.router = CustomRouter(...)
```

### Thêm loss term
Edit `GatingLoss.forward()`:
```python
# Ví dụ: diversity loss khác
loss_diversity = some_diversity_metric(weights)
total_loss += lambda_div * loss_diversity
```

---

## 🐛 Troubleshooting

### Issue: Gating collapse (chỉ 1 expert được dùng)
**Solutions:**
1. Tăng `lambda_h` (entropy regularization): 0.01 → 0.05
2. Tăng `lambda_lb` (load-balancing): 1e-2 → 5e-2
3. Giảm learning rate: 1e-3 → 5e-4
4. Thêm warmup: `warmup_epochs = 10`

### Issue: Training không ổn định (loss spike)
**Solutions:**
1. Enable gradient clipping: `grad_clip = 1.0`
2. Dùng LayerNorm thay BatchNorm (đã default)
3. Giảm learning rate
4. Tăng batch size

### Issue: Mixture acc thấp hơn best expert
**Nguyên nhân:** Gating chưa học tốt, routing không optimal

**Solutions:**
1. Train lâu hơn: epochs 100 → 200
2. Simplify architecture: hidden_dims=[128] thay vì [256,128]
3. Check features: in ra uncertainty metrics để xem có informative không

### Issue: Validation acc tốt nhưng test acc kém
**Nguyên nhân:** Overfitting

**Solutions:**
1. Tăng dropout: 0.1 → 0.3
2. Tăng weight_decay: 1e-4 → 5e-4
3. Early stopping: save best model on val, stop khi không cải thiện

---

## 📈 Next Steps

Sau khi train gating, bạn có thể:

1. **Integrate với MAP Selector:**
   ```python
   # Load gating
   gating = GatingNetwork(...)
   gating.load_state_dict(torch.load('best_gating.pth'))
   
   # Compute mixture & uncertainty
   weights, _ = gating(posteriors)
   mixture = gating.get_mixture_posterior(posteriors, weights)
   U = compute_uncertainty_for_map(posteriors, weights, mixture)
   
   # MAP margin
   m_MAP = m_L2R(mixture, alpha, mu, c) - gamma * U
   
   # Decision
   accept = (m_MAP >= 0)
   ```

2. **Optimize α, μ, γ trên S1/S2:** (theo pipeline MAP)

3. **EG-outer cho worst-group:** (nếu cần R_max)

4. **Conformal Risk Control:** (finite-sample guarantee)

---

## 📚 References

### Mixture of Experts
1. Jordan & Jacobs (1994): Hierarchical Mixtures of Experts and the EM Algorithm
2. Jacobs et al. (1991): Adaptive Mixtures of Local Experts

### Sparse MoE & Load Balancing
3. Shazeer et al. (2017): Outrageously Large Neural Networks (Sparsely-Gated MoE)
4. Fedus et al. (2021): Switch Transformers
5. Lepikhin et al. (2020): GShard - Scaling Giant Models with Conditional Computation

### Uncertainty & Calibration
6. Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty
7. Guo et al. (2017): On Calibration of Modern Neural Networks
8. Kull et al. (2019): Beyond Temperature Scaling (Dirichlet calibration)

### Selective Classification
9. Geifman & El-Yaniv (2017): Selective Classification for Deep Neural Networks
10. Geifman & El-Yaniv (2019): SelectiveNet - Learning to Abstain
11. Stutz et al. (2023): Learning to Reject with L2R

### Conformal Prediction
12. Angelopoulos et al. (2024): Conformal Risk Control

---

## ✅ Checklist Implementation

- [x] Feature extraction (uncertainty/disagreement)
- [x] Gating MLP với LayerNorm
- [x] Dense routing (softmax)
- [x] Noisy Top-K routing
- [x] Mixture NLL loss
- [x] Load-balancing loss (Switch)
- [x] Entropy regularization
- [x] Combined loss function
- [x] Training script với warmup/scheduler
- [x] Validation & metrics
- [x] Group-wise accuracy (head/tail)
- [x] Checkpoint saving
- [x] U(x) computation cho MAP
- [x] Test suite
- [x] Documentation

---

## 📧 Contact & Support

Nếu có vấn đề, hãy check:
1. Test suite: `python src/train/test_gating_map.py`
2. Loss values: NLL ~2-3, LB ~0.01, Entropy ~1-2
3. Metrics: mixture_acc > max(expert_accs), effective_experts ≈ E

Happy training! 🚀
