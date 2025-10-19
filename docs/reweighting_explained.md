# Reweighting Explained - Ví dụ Chi tiết

## Vấn đề

Dataset CIFAR-100-LT với IF=100 có phân phối long-tail:
- **Head classes** (class 0-19): ~500 samples/class  
- **Mid classes** (class 20-79): ~50-500 samples/class
- **Tail classes** (class 80-99): ~5 samples/class

Nhưng validation sets (val, tunev, test) là **balanced**: 10 samples/class

❌ **Vấn đề**: Nếu tính accuracy thông thường trên balanced val set, metrics không phản ánh performance thực tế trên long-tail data!

## Solution: Reweighting

Dùng class weights từ training distribution để reweight metrics trên balanced val set.

---

## Ví dụ Cụ thể

### Setup
```python
# CIFAR-100-LT IF=100
Total training samples: 10,847
- Class 0 (head): 500 samples → weight = 500/10847 = 0.0461
- Class 50 (mid): 50 samples → weight = 50/10847 = 0.0046  
- Class 99 (tail): 5 samples → weight = 5/10847 = 0.00046

# Validation set (balanced)
- Each class: 10 samples
- Total: 1,000 samples
```

### Scenario: Model predictions on val set

```python
# Predictions on 1,000 balanced validation samples
Correct predictions:
- Class 0 (head): 9/10 correct → acc = 90%
- Class 50 (mid): 7/10 correct → acc = 70%
- Class 99 (tail): 3/10 correct → acc = 30%
```

---

## Method 1: Standard Accuracy (WRONG for long-tail)

```python
# Mỗi class đóng góp như nhau
total_correct = 9 + 7 + 3 = 19
total_samples = 10 + 10 + 10 = 30
accuracy = 19/30 = 63.3%
```

❌ **Vấn đề**: Class 99 (tail, chỉ 5 samples trong train) đóng góp bằng class 0 (head, 500 samples)!

---

## Method 2: Reweighted Accuracy (CORRECT)

```python
# Load class weights from training distribution
class_weights = {
    0: 0.0461,   # head class
    50: 0.0046,  # mid class  
    99: 0.00046, # tail class
    ...
}

# Calculate per-class accuracy
class_acc = {
    0: 9/10 = 0.90,   # 90% correct
    50: 7/10 = 0.70,  # 70% correct
    99: 3/10 = 0.30,  # 30% correct
}

# Reweight by training distribution
weighted_acc = sum(class_acc[c] * class_weights[c] for c in classes)
             = 0.90 * 0.0461 + 0.70 * 0.0046 + 0.30 * 0.00046
             = 0.04149 + 0.00322 + 0.000138
             = 0.04485 (trên 3 classes này)

# Full calculation với 100 classes
weighted_acc = Σ(class_acc[c] * class_weights[c]) for c = 0 to 99
```

✅ **Kết quả**: Metrics phản ánh đúng performance trên long-tail distribution!

---

## Code Implementation

### 1. Load Class Weights

```python
def load_class_weights(splits_dir, num_classes=100):
    """Load class weights from training distribution."""
    weights_path = os.path.join(splits_dir, 'class_weights.json')
    
    with open(weights_path, 'r') as f:
        data = json.load(f)
    
    # Support both list and dict formats
    if isinstance(data, list):
        weights = torch.tensor(data, dtype=torch.float32)
    else:
        weights = torch.tensor([data[str(i)] for i in range(num_classes)])
    
    # Normalize to sum to 1
    weights = weights / weights.sum()
    
    return weights
```

### 2. Reweighted Accuracy Calculation

```python
def compute_reweighted_accuracy(logits, targets, class_weights):
    """
    Compute accuracy reweighted by class distribution.
    
    Args:
        logits: [N, C] model predictions
        targets: [N] ground truth labels
        class_weights: [C] weights from training distribution
    
    Returns:
        reweighted_acc: float, accuracy weighted by class distribution
    """
    preds = logits.argmax(dim=1)  # [N]
    
    num_classes = logits.size(1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    # Count correct per class
    for c in range(num_classes):
        mask = (targets == c)
        class_total[c] = mask.sum()
        class_correct[c] = (preds[mask] == c).sum()
    
    # Per-class accuracy
    class_acc = torch.zeros(num_classes)
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c] / class_total[c]
    
    # Reweight by training distribution
    reweighted_acc = (class_acc * class_weights).sum()
    
    return reweighted_acc.item()
```

### 3. Example Usage in Validation

```python
# In validation loop
def validate_model(model, val_loader, class_weights):
    """Validate with reweighted metrics."""
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            logits = model(data)
            all_logits.append(logits)
            all_targets.append(target)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Standard accuracy (on balanced val)
    preds = all_logits.argmax(dim=1)
    standard_acc = (preds == all_targets).float().mean()
    
    # Reweighted accuracy (simulates long-tail)
    reweighted_acc = compute_reweighted_accuracy(
        all_logits, all_targets, class_weights
    )
    
    print(f"Standard Accuracy: {standard_acc:.4f}")
    print(f"Reweighted Accuracy: {reweighted_acc:.4f}")
    
    return reweighted_acc
```

---

## Ví dụ Thực tế với 100 Classes

### Training Distribution (class_weights.json)
```json
{
  "0": 0.0461,   // 500 samples (head)
  "1": 0.0456,   // 495 samples
  ...
  "50": 0.0046,  // 50 samples (mid)
  ...
  "99": 0.00046  // 5 samples (tail)
}
```

### Validation Set (balanced)
```
- Class 0: 10 samples
- Class 1: 10 samples
...
- Class 99: 10 samples
Total: 1,000 samples
```

### Model Performance
```python
# Giả sử model có head bias
Head classes (0-19):  8/10 correct avg → 80% per class
Mid classes (20-79):  6/10 correct avg → 60% per class
Tail classes (80-99): 2/10 correct avg → 20% per class

# Standard accuracy (WRONG)
total_correct = 20*8 + 60*6 + 20*2 = 160 + 360 + 40 = 560
standard_acc = 560/1000 = 56.0%

# Reweighted accuracy (CORRECT)
# Head classes contribute more (higher weights)
head_contrib = 0.80 * sum(weights[0:20])
mid_contrib = 0.60 * sum(weights[20:80])  
tail_contrib = 0.20 * sum(weights[80:100])

# Approximate calculation
sum(weights[0:20]) ≈ 0.90   # head has ~90% of data
sum(weights[20:80]) ≈ 0.095 # mid has ~9.5%
sum(weights[80:100]) ≈ 0.005 # tail has ~0.5%

reweighted_acc = 0.80*0.90 + 0.60*0.095 + 0.20*0.005
               = 0.72 + 0.057 + 0.001
               = 77.8%

# Reweighted accuracy HIGHER vì model tốt ở head classes 
# (chiếm phần lớn training data)
```

---

## Khi nào dùng Standard vs Reweighted?

### Standard Accuracy
- Đo performance trên balanced distribution
- Mọi class đóng góp như nhau
- **Use case**: Khi muốn đảm bảo fairness across classes

### Reweighted Accuracy  
- Đo performance trên long-tail distribution
- Classes weighted theo training frequency
- **Use case**: Khi muốn optimize cho real-world performance

---

## So sánh với Training Loss

### Training Loss (on long-tail data)
```python
# Naturally weighted by data distribution
loss = CrossEntropyLoss()(logits, targets)
# Head classes appear 100x more → dominate loss
```

### Validation Accuracy (on balanced data)
```python
# Option 1: Standard (all classes equal weight)
acc_standard = (preds == targets).mean()

# Option 2: Reweighted (match training distribution)
acc_reweighted = compute_reweighted_accuracy(logits, targets, class_weights)
```

✅ **Reweighted validation accuracy is consistent with training objective!**

---

## Trong AR-GSE Pipeline

### Expert Training
```python
# Train on expert split (long-tail, 9,719 samples)
train_loss = train_on_expert_split()

# Validate on balanced val (1,000 samples) with reweighting
val_acc_reweighted = validate_with_reweighting(class_weights)
```

### Gating Training  
```python
# Pretrain on gating split (long-tail, 1,128 samples)
train_loss = train_on_gating_split()

# Selective training on tunev+val (balanced, 2,000 samples) with reweighting
val_acc_reweighted = validate_with_reweighting(class_weights)
```

### Plugin Training
```python
# Optimize on S1=tunev, S2=val (both balanced)
# But minimize reweighted error
error_s1 = compute_reweighted_error(S1_logits, S1_targets, class_weights)
error_s2 = compute_reweighted_error(S2_logits, S2_targets, class_weights)
```

### Final Evaluation
```python
# Test on balanced test set (8,000 samples) with reweighting
test_acc_reweighted = evaluate_with_reweighting(class_weights)
```

---

## Key Insights

1. **Clean data, realistic metrics**: Train/eval on balanced splits (no duplication), but reweight to reflect long-tail performance

2. **Consistency**: Reweighted validation matches training objective (both favor head classes proportionally)

3. **Fair comparison**: All stages (expert, gating, plugin, eval) use same reweighting → comparable metrics

4. **Interpretability**: Reweighted accuracy tells you actual performance on long-tail test distribution

---

## Common Mistakes

❌ **Mistake 1**: Use standard accuracy on balanced val
```python
acc = (preds == targets).mean()  # Treats all classes equally
```

❌ **Mistake 2**: Assume balanced val represents long-tail performance
```python
# 80% on balanced val != 80% on long-tail test!
```

✅ **Correct**: Always reweight when evaluating on balanced data for long-tail task
```python
acc = compute_reweighted_accuracy(logits, targets, class_weights)
```

---

## Summary

**Reweighting formula**:
```
reweighted_acc = Σ(class_acc[c] * class_weight[c]) for all classes c

where:
- class_acc[c] = accuracy on class c (from balanced val)
- class_weight[c] = frequency of class c in training set
```

**Effect**: Metrics on balanced validation reflect performance on long-tail test distribution.

**Benefits**: 
- Clean balanced data (no duplication)
- Realistic metrics (reflects long-tail)
- Consistent across pipeline stages
- Easy to implement (just load weights and apply formula)
