# Reweighting for Long-tail Evaluation

## 🎯 Mục đích

**Vấn đề**: Test/val splits là **balanced** (~10 samples/class), nhưng train là **imbalanced** (IF=100).

**Giải pháp**: Reweight test samples để metrics phản ánh performance trên **train distribution**.

---

## 📐 Công thức

### Sample Weights

$$w_i = \text{freq}_{\text{train}}(y_i) = \frac{n_{\text{train}}(y_i)}{N_{\text{train}}}$$

Trong đó:
- $n_{\text{train}}(c)$: Số samples của class $c$ trong train set
- $N_{\text{train}}$: Tổng số samples trong train

### Group Error (có reweighting)

$$e_k(w) = \frac{\sum_{i: g_i=k} w_i \cdot \mathbb{1}[\hat{y}_i \neq y_i] \cdot A_i}{\sum_{i: g_i=k} w_i \cdot A_i + \varepsilon}$$

Trong đó:
- $g_i \in \{\text{head}, \text{tail}\}$: Group của sample $i$
- $A_i$: Accept indicator (1 nếu không reject, 0 nếu reject)
- $w_i$: Train frequency weight
- $\varepsilon$: Small constant để tránh division by zero

---

## 🔢 Ví dụ Cụ thể

### CIFAR-100-LT (IF=100)

**Train distribution**:
- Class 0 (head): 500 samples → $w_0 = 500/10847 = 0.0461$
- Class 99 (tail): 5 samples → $w_{99} = 5/10847 = 0.0005$
- Ratio: $w_0 / w_{99} = 100$

**Test distribution**: Balanced (~10 samples/class)

### Tính Group Error

Giả sử test có 10 samples được accept:

**Head group (classes 0-68)**:
```python
labels = [5, 10, 20, 30, 40, 50]  # 6 samples
predictions = [5, 10, 22, 30, 42, 51]  # 3 wrong
weights = [0.046, 0.044, 0.042, 0.040, 0.038, 0.037]

# Numerator: Σ w_i * 1[wrong] * A_i
numerator = 0 + 0 + 0.042 + 0 + 0.038 + 0.037 = 0.117

# Denominator: Σ w_i * A_i
denominator = 0.046 + 0.044 + 0.042 + 0.040 + 0.038 + 0.037 = 0.247

# Error
e_head = 0.117 / 0.247 = 0.474 (47.4%)
```

**Tail group (classes 69-99)**:
```python
labels = [70, 75, 80, 99]  # 4 samples
predictions = [71, 75, 82, 99]  # 2 wrong
weights = [0.0011, 0.0008, 0.0006, 0.0005]

# Numerator
numerator = 0.0011 + 0 + 0.0006 + 0 = 0.0017

# Denominator
denominator = 0.0011 + 0.0008 + 0.0006 + 0.0005 = 0.0030

# Error
e_tail = 0.0017 / 0.0030 = 0.567 (56.7%)
```

### So sánh với Uniform Weights

**Không có reweighting** (tất cả $w_i = 1$):
```python
e_head = 3/6 = 50.0%
e_tail = 2/4 = 50.0%
```

**Có reweighting** (weights = train freq):
```python
e_head = 47.4%  # Thấp hơn vì head samples có weight cao
e_tail = 56.7%  # Cao hơn vì tail samples có weight thấp
```

---

## 🧠 Ý nghĩa

### Tại sao cần reweight?

**Scenario 1: Không reweight**
- Test balanced → mỗi class đóng góp bằng nhau vào metric
- Tail classes (50 classes) chiếm 50% metric
- **Không phản ánh thực tế**: Train có 99% head, 1% tail!

**Scenario 2: Có reweight**
- Head samples có weight cao → ảnh hưởng nhiều đến metric
- Tail samples có weight thấp → ảnh hưởng ít
- **Phản ánh đúng**: Performance trên distribution thực tế (imbalanced)

### Group boundaries

Trong code, groups được định nghĩa:
- **Head**: Classes 0-68 (69 classes)
- **Tail**: Classes 69-99 (31 classes)

Tại sao 69-31 chứ không phải 50-50?
- Train distribution không phải step function
- Classes 0-68 có nhiều samples hơn đáng kể
- Boundary tại class 69 phân chia tốt hơn

---

## 💻 Implementation

### Load weights

```python
def load_sample_weights(splits_dir, split_name, device='cpu'):
    # Load train frequencies
    with open(f'{splits_dir}/class_weights.json') as f:
        class_weights = json.load(f)  # [w_0, w_1, ..., w_99]
    
    # Convert to tensor
    class_weights = torch.tensor(class_weights, device=device)
    
    # Get per-sample weights
    labels = load_labels(splits_dir, split_name, device)
    sample_weights = class_weights[labels]  # [w_i for i in samples]
    
    return sample_weights
```

### Compute metrics với reweighting

```python
# src/models/map_selector_simple.py
def compute_selective_metrics(..., sample_weights):
    # Group error với reweighting
    for k in range(num_groups):
        group_mask = (label_groups == k)
        
        # Numerator: Σ w_i * 1[wrong] * A_i
        errors_weighted = (accept[group_mask] & ~correct[group_mask]).float() 
                         * sample_weights[group_mask]
        
        # Denominator: Σ w_i * A_i
        total_weighted = accept[group_mask].float() * sample_weights[group_mask]
        
        # Group error
        group_errors[k] = errors_weighted.sum() / (total_weighted.sum() + eps)
```

---

## 📊 Kết quả mong đợi

Với reweighting, metrics sẽ:
- **Gần với train performance hơn** (vì distribution giống train)
- **Head errors ảnh hưởng nhiều hơn** (vì head chiếm phần lớn train)
- **Tail errors ít quan trọng hơn** (vì tail chỉ chiếm 1% train)

Ví dụ:
```
Without reweight (uniform):
  e_head = 40%, e_tail = 60%
  R_balanced = (40% + 60%) / 2 = 50%
  
With reweight (train freq):
  e_head = 40%, e_tail = 60%
  R_balanced (weighted) ≈ 40.2%  (vì head có weight cao hơn)
```

---

## ✅ Checklist

Để đảm bảo reweighting đúng:

- [x] `class_weights.json` chứa train frequencies (không phải inverse!)
- [x] Head classes có weight cao (~0.046)
- [x] Tail classes có weight thấp (~0.0005)
- [x] Ratio = 100x (match imbalance factor)
- [x] Group boundaries = [69] (69 head, 31 tail)
- [x] `compute_selective_metrics()` nhận `sample_weights` parameter
- [x] Công thức: `e_k = Σ(w*error*A) / Σ(w*A)`

---

## 📚 References

1. Long-tail learning papers thường dùng **balanced test** để fair comparison
2. Reweighting test → metrics reflect **train distribution**
3. Alternative: Report metrics on **imbalanced test** (giống train), nhưng khó tạo
4. Công thức trong paper: "evaluate on balanced test with importance weighting"
