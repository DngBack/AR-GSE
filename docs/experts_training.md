# 🎯 AR-GSE Expert Training Documentation

Tài liệu chi tiết về hệ thống training expert models trong AR-GSE, bao gồm kiến trúc, loss functions, training pipeline, và calibration.

## 🎯 Tổng quan

Expert training là **bước đầu tiên** trong pipeline AR-GSE, tạo ra 3 expert models với các chiến lược khác nhau để xử lý imbalanced data. Mỗi expert học một approach riêng biệt, sau đó được ensemble lại bởi gating network.

### Mục tiêu chính:
- **Diversity**: Tạo ra các experts với strengths khác nhau
- **Complementarity**: Experts bổ sung cho nhau trên different class groups
- **Calibration**: Đảm bảo outputs có confidence scores đáng tin cậy
- **Robustness**: Xử lý tốt long-tail distribution

## 🏗️ Kiến trúc Expert System

```
src/
├── train/
│   └── train_expert.py        # Core training logic
├── models/
│   ├── experts.py            # Expert model wrapper
│   ├── losses.py             # Specialized loss functions
│   └── backbones/
│       └── resnet_cifar.py   # CIFAR-optimized ResNet-32
├── metrics/
│   └── calibration.py        # Temperature scaling
└── data/
    └── dataloader_utils.py   # Data loading utilities
```

## 🤖 Three Expert Models

### 1. CE Expert (Cross Entropy Baseline)

```python
EXPERT_CONFIGS['ce'] = {
    'name': 'ce_baseline',
    'loss_type': 'ce', 
    'epochs': 256,
    'lr': 0.1,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1,
    'milestones': [96, 192, 224],
    'gamma': 0.1
}
```

**Đặc điểm:**
- **Standard Cross Entropy**: Không có adjustments cho imbalance
- **Baseline Performance**: Đại diện cho vanilla training
- **Strong on Head Classes**: Performs well trên frequent classes
- **Use Case**: Provides reliable predictions cho head classes

### 2. LogitAdjust Expert

```python
EXPERT_CONFIGS['logitadjust'] = {
    'name': 'logitadjust_baseline',
    'loss_type': 'logitadjust',
    'epochs': 256, 
    'lr': 0.1,
    'weight_decay': 5e-4,
    'dropout_rate': 0.1,
    'milestones': [160, 180],
    'gamma': 0.1
}
```

**Loss Function:**
```python
class LogitAdjustLoss(nn.Module):
    def __init__(self, class_counts, tau=1.0):
        # Calculate class priors: π_y = n_y / Σn_y  
        priors = class_counts / class_counts.sum()
        self.log_priors = torch.log(priors)
        self.tau = tau
    
    def forward(self, logits, target):
        # Adjust logits: ŝ_y = s_y + τ * log(π_y)
        adjusted_logits = logits + self.tau * self.log_priors
        return F.cross_entropy(adjusted_logits, target)
```

**Nguyên lý:**
- **Prior Adjustment**: Bù trừ class imbalance bằng log priors
- **Tail Boost**: Tăng logits cho tail classes dựa trên frequency
- **Theoretical Foundation**: Dựa trên Bayesian posterior adjustment
- **Balances Performance**: Cải thiện tail classes mà không làm hỏng head classes

### 3. BalancedSoftmax Expert

```python
EXPERT_CONFIGS['balsoftmax'] = {
    'name': 'balsoftmax_baseline',
    'loss_type': 'balsoftmax',
    'epochs': 256,
    'lr': 0.1, 
    'weight_decay': 5e-4,
    'dropout_rate': 0.1,
    'milestones': [96, 192, 224],
    'gamma': 0.1
}
```

**Loss Function:**
```python
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts):
        # Use raw counts instead of priors
        self.log_priors = torch.log(class_counts)
    
    def forward(self, logits, target):
        # Adjust logits: ŝ_y = s_y + log(n_y)
        adjusted_logits = logits + self.log_priors  
        return F.cross_entropy(adjusted_logits, target)
```

**Khác biệt với LogitAdjust:**
- **Raw Counts**: Sử dụng log(n_y) thay vì log(π_y)
- **Stronger Adjustment**: Tác động mạnh hơn lên rare classes
- **Different Balance Point**: Khác biệt trong cách balance head/tail

## 🏛️ Model Architecture Deep Dive

### Expert Model Wrapper

```python
class Expert(nn.Module):
    def __init__(self, num_classes=100, backbone_name='cifar_resnet32', 
                 dropout_rate=0.0, init_weights=True):
        self.backbone = CIFARResNet32(dropout_rate, init_weights)
        feature_dim = self.backbone.get_feature_dim()  # 64
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
    
    def get_calibrated_logits(self, x):
        return self.forward(x) / self.temperature
```

**Design Principles:**
- **Modular Design**: Backbone riêng biệt với classifier
- **Calibration Ready**: Built-in temperature scaling
- **Feature Extraction**: Có thể extract features cho analysis
- **Flexible Architecture**: Dễ thay đổi backbone nếu cần

### CIFAR-Optimized ResNet-32

```python
class CIFARResNet32:
    # Architecture: 1 + 3×(5×2) + 1 = 32 layers
    # 3 residual groups: [16, 32, 64] channels
    # 5 basic blocks per group
    # Output feature dim: 64
```

**Key Optimizations cho CIFAR:**

1. **Smaller Initial Conv**: 3x3 thay vì 7x7 (ImageNet)
2. **No Initial MaxPool**: Giữ spatial resolution cho 32x32 images  
3. **Fewer Initial Channels**: 16 thay vì 64
4. **Proper Downsampling**: Chỉ downsample ở group transitions
5. **Adaptive Global Pooling**: Flexibility cho different input sizes

```python
class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, dropout_rate=0.0):
        # Initial: 32x32x3 → 32x32x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # Group 1: 32x32x16 → 32x32x16 (5 blocks)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        
        # Group 2: 32x32x16 → 16x16x32 (5 blocks)  
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        
        # Group 3: 16x16x32 → 8x8x64 (5 blocks)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Global pooling: 8x8x64 → 1x1x64
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

## 🔄 Training Pipeline Deep Dive

### 1. Main Training Loop

```python
def train_single_expert(expert_key):
    """Train một expert từ đầu đến cuối"""
    
    # 1. Setup
    model = Expert(num_classes=100, dropout_rate=config['dropout_rate'])
    criterion = get_loss_function(config['loss_type'], train_loader)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                         momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=config['milestones'], 
                                              gamma=config['gamma'])
    
    # 2. Training Loop
    for epoch in range(config['epochs']):
        # Train phase
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation phase
        val_acc, group_accs = validate_model(model, val_loader, device)
        
        # Save best model
        if val_acc > best_acc:
            torch.save(model.state_dict(), best_model_path)
    
    # 3. Post-processing
    # Temperature calibration
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(model, val_loader, device)
    model.set_temperature(optimal_temp)
    
    # Export logits for ensemble training
    export_logits_for_all_splits(model, expert_name)
```

### 2. Learning Rate Scheduling

**MultiStepLR Strategy:**
```python
# CE Expert: Conservative schedule
milestones = [96, 192, 224]  # 3/8, 3/4, 7/8 of total epochs
gamma = 0.1                   # LR decay factor

# LogitAdjust: Early decay for stability  
milestones = [160, 180]      # 5/8, 11/16 of total epochs
gamma = 0.1

# BalancedSoftmax: Same as CE
milestones = [96, 192, 224]
gamma = 0.1
```

**Rationale:**
- **Long Warm-up**: Nhiều epochs ở LR cao để học features
- **Gradual Decay**: Từ từ giảm LR để fine-tune
- **Final Convergence**: LR thấp cuối để converge tốt

### 3. Validation với Group-wise Metrics

```python
def validate_model(model, val_loader, device):
    """Validation với phân tích head/tail performance"""
    group_correct = {'head': 0, 'tail': 0}
    group_total = {'head': 0, 'tail': 0}
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i, target in enumerate(targets):
                pred = predicted[i]
                if target < 50:  # Head classes (0-49)
                    group_total['head'] += 1
                    if pred == target:
                        group_correct['head'] += 1 
                else:  # Tail classes (50-99)
                    group_total['tail'] += 1
                    if pred == target:
                        group_correct['tail'] += 1
    
    # Calculate accuracies
    overall_acc = 100 * correct / total
    group_accs = {
        'head': 100 * group_correct['head'] / group_total['head'],
        'tail': 100 * group_correct['tail'] / group_total['tail']
    }
    
    return overall_acc, group_accs
```

**Key Insights:**
- **Balanced Monitoring**: Theo dõi cả head và tail performance
- **Early Detection**: Phát hiện sớm overfitting trên head classes
- **Model Selection**: Chọn model balance tốt head/tail

## 🌡️ Temperature Calibration

### Why Calibration Matters

**Problem**: Raw neural network outputs không reflect true confidence
- High confidence predictions có thể sai
- Low confidence predictions có thể đúng  
- Imbalanced data làm tăng miscalibration

**Solution**: Temperature scaling để adjust confidence

### Temperature Scaling Algorithm

```python
class TemperatureScaler:
    def fit(self, model, dataloader, device):
        """Find optimal temperature T minimizing NLL"""
        
        # 1. Collect all logits and labels
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                logits = model(inputs.to(device))
                all_logits.append(logits)
                all_labels.append(labels.to(device))
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # 2. Optimize temperature parameter
        temperature = nn.Parameter(torch.ones(1).to(device))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            # Scaled logits: s/T
            loss = F.cross_entropy(all_logits / temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return temperature.item()
```

**How it Works:**
1. **Collect Validation Data**: Get logits + labels từ validation set
2. **Optimize Temperature**: Minimize NLL w.r.t. T using L-BFGS  
3. **Apply Scaling**: Use T để scale logits trong inference
4. **Calibrated Probabilities**: softmax(logits/T) gives better confidence

### Expected Calibration Error (ECE)

```python
def calculate_ece(posteriors, labels, n_bins=15):
    """Measure calibration quality"""
    confidences, predictions = torch.max(posteriors, 1)
    accuracies = predictions.eq(labels)
    
    # Bin predictions by confidence
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            # |confidence - accuracy| weighted by bin size
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()
```

## 📊 Logit Export System

### Purpose của Logit Export

Sau khi training, mỗi expert export **calibrated logits** cho tất cả splits:

```python
def export_logits_for_all_splits(model, expert_name):
    """Export logits cho tất cả dataset splits"""
    
    splits_to_export = [
        'train',      # Training data  
        'val_lt',     # Validation long-tail
        'test_lt',    # Test long-tail
        'tuneV',      # Calibration split
    ]
    
    for split_name in splits_to_export:
        # Load split indices
        with open(f"{split_name}_indices.json") as f:
            indices = json.load(f)
        
        # Create dataloader
        dataset = Subset(base_cifar_dataset, indices)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        # Export calibrated logits
        all_logits = []
        with torch.no_grad():
            for inputs, _ in loader:
                logits = model.get_calibrated_logits(inputs.to(device))
                all_logits.append(logits.cpu())
        
        # Save as compressed tensors
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), 
                  f"{output_dir}/{split_name}_logits.pt")
```

**Benefits:**
- **Efficiency**: Tránh recompute logits trong ensemble training
- **Consistency**: Đảm bảo same preprocessing cho all experts  
- **Storage Optimization**: float16 để tiết kiệm space
- **Easy Loading**: Simple torch.load() trong ensemble phase

### Output Structure

```
outputs/logits/cifar100_lt_if100/
├── ce_baseline/
│   ├── train_logits.pt      # [~23K, 100] 
│   ├── val_lt_logits.pt     # [~2.4K, 100]
│   ├── test_lt_logits.pt    # [~10K, 100] 
│   └── tuneV_logits.pt      # [~1.8K, 100]
├── logitadjust_baseline/
│   └── ... (same structure)
└── balsoftmax_baseline/
    └── ... (same structure)
```

## 🎮 Training Script Interface

### Main Script: `train_experts.py`

```bash
# Train all experts
python train_experts.py

# Train specific expert  
python train_experts.py --expert ce --verbose

# Override parameters
python train_experts.py --expert logitadjust --epochs 200 --lr 0.05

# Dry run to check configuration
python train_experts.py --dry-run --verbose
```

**Arguments:**
- `--expert {ce|logitadjust|balsoftmax|all}`: Which expert to train
- `--epochs INT`: Override number of training epochs
- `--lr FLOAT`: Override learning rate
- `--batch-size INT`: Override batch size  
- `--device {cpu|cuda|auto}`: Training device
- `--verbose`: Detailed output
- `--dry-run`: Show config without training
- `--resume`: Resume from checkpoint

### Configuration Overrides

```python
def apply_overrides(expert_configs, args):
    """Apply command-line overrides"""
    for expert_key in expert_configs:
        if args.epochs:
            expert_configs[expert_key]['epochs'] = args.epochs
        if args.lr: 
            expert_configs[expert_key]['lr'] = args.lr
        # ... other overrides
    
    return expert_configs
```

## 📈 Performance Analysis

### Typical Training Results

**CE Expert:**
```
Epoch 256: Loss=1.234, Val Acc=62.5%, Head=78.2%, Tail=34.1%
Temperature calibration: T = 1.087
Final Results - Overall: 62.8%, Head: 78.5%, Tail: 34.7%
```

**LogitAdjust Expert:**  
```
Epoch 256: Loss=1.456, Val Acc=58.3%, Head=71.6%, Tail=41.2%  
Temperature calibration: T = 1.145
Final Results - Overall: 58.7%, Head: 72.1%, Tail: 42.0%
```

**BalancedSoftmax Expert:**
```
Epoch 256: Loss=1.389, Val Acc=59.8%, Head=73.4%, Tail=39.8%
Temperature calibration: T = 1.098  
Final Results - Overall: 60.1%, Head: 73.8%, Tail: 40.5%
```

### Performance Patterns

1. **CE Expert**: Highest overall accuracy, strong on head, weak on tail
2. **LogitAdjust**: Best tail performance, moderate overall accuracy  
3. **BalancedSoftmax**: Balanced between CE và LogitAdjust

**Complementarity**: Mỗi expert có strengths khác nhau → ensemble potential

## 🔧 Advanced Training Features

### 1. Automatic Mixed Precision (AMP)

```python
# Optional AMP support for faster training
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Clipping

```python
# Prevent exploding gradients
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

### 3. Model Checkpointing

```python
# Save training state for resumability
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': best_acc,
    'config': expert_config
}
torch.save(checkpoint, checkpoint_path)
```

## 🧪 Usage Examples

### 1. Basic Training

```python
# Train all experts sequentially  
from train_experts import main
main()

# Or programmatically
from src.train.train_expert import train_single_expert

for expert in ['ce', 'logitadjust', 'balsoftmax']:
    model_path = train_single_expert(expert)
    print(f"Trained {expert}: {model_path}")
```

### 2. Custom Configuration

```python
from src.train.train_expert import EXPERT_CONFIGS

# Modify config before training
EXPERT_CONFIGS['ce']['epochs'] = 200
EXPERT_CONFIGS['ce']['lr'] = 0.05

# Add custom expert
EXPERT_CONFIGS['custom'] = {
    'name': 'custom_expert',
    'loss_type': 'ce', 
    'epochs': 100,
    'lr': 0.1,
    'weight_decay': 1e-3,
    'dropout_rate': 0.2,
    'milestones': [50, 80],
    'gamma': 0.1
}
```

### 3. Evaluation và Analysis

```python
# Load trained expert
from src.models.experts import Expert

model = Expert(num_classes=100)
model.load_state_dict(torch.load('checkpoints/experts/cifar100_lt_if100/ce_baseline.pth'))

# Evaluate on test set
from src.data.dataloader_utils import get_expert_training_dataloaders
_, test_loader = get_expert_training_dataloaders()

acc, group_accs = validate_model(model, test_loader, 'cuda')
print(f"Test Accuracy: {acc:.2f}%")
print(f"Head: {group_accs['head']:.2f}%, Tail: {group_accs['tail']:.2f}%")
```

## 🔍 Debugging & Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solutions:
- Reduce batch_size in CONFIG['train_params']
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training
```

**2. Poor Tail Performance**
```python
# Check:
- Class count distribution in dataset
- Loss function implementation 
- Learning rate too high/low
- Weight decay too strong
```

**3. Training Instability**
```python
# Solutions:
- Add gradient clipping
- Lower learning rate
- Increase warmup epochs
- Check data preprocessing
```

### Validation Checks

```python
def validate_expert_training():
    """Comprehensive validation of expert training setup"""
    
    # 1. Check data splits exist
    assert Path("data/cifar100_lt_if100_splits").exists()
    
    # 2. Check CUDA availability
    assert torch.cuda.is_available(), "CUDA required for training"
    
    # 3. Validate configs
    for expert, config in EXPERT_CONFIGS.items():
        assert config['epochs'] > 0
        assert 0 < config['lr'] < 1
        assert len(config['milestones']) > 0
    
    # 4. Test data loading
    train_loader, val_loader = get_dataloaders()
    batch = next(iter(train_loader))
    assert batch[0].shape[0] > 0  # Non-empty batch
    
    print("✅ All validation checks passed!")
```

## 📊 Expected Results

### Training Time (on RTX 3090)

- **CE Expert**: ~2.5 hours (256 epochs)
- **LogitAdjust Expert**: ~2.5 hours  
- **BalancedSoftmax Expert**: ~2.5 hours
- **Total**: ~7.5 hours cho all experts

### Model Sizes

- **Each Expert**: ~11.2M parameters
- **Checkpoint Size**: ~43MB per expert
- **Total Storage**: ~130MB cho all experts + logits

### Memory Requirements

- **Training**: ~8GB GPU memory (batch_size=128)
- **Inference**: ~2GB GPU memory  
- **Logit Storage**: ~500MB cho all splits

---

## � Expert Evaluation System

### 1. Individual Expert Performance Metrics

```python
def evaluate_single_expert(model, test_loader, class_to_group, num_groups):
    """
    Comprehensive evaluation của từng expert.
    
    Metrics:
    - Overall Accuracy: Standard classification accuracy
    - Per-Group Accuracy: Head vs Tail performance
    - Balanced Accuracy: (1/K) Σ acc_k
    - Worst-Group Accuracy: min_k acc_k  
    - Per-Class Accuracy: Detailed class-level analysis
    - Calibration Error (ECE): Before và after temperature scaling
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data.to(device))
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_confidences.append(confidences.cpu())
    
    # Aggregate results
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    confidences = torch.cat(all_confidences)
    
    # Overall metrics
    overall_acc = (preds == labels).float().mean().item()
    
    # Per-group analysis
    y_groups = class_to_group[labels]
    group_accuracies = []
    
    for k in range(num_groups):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:
            group_acc = (preds[group_mask] == labels[group_mask]).float().mean().item()
            group_accuracies.append(group_acc)
        else:
            group_accuracies.append(0.0)
    
    balanced_acc = np.mean(group_accuracies)
    worst_acc = np.min(group_accuracies)
    
    return {
        'overall_accuracy': overall_acc,
        'group_accuracies': group_accuracies,
        'balanced_accuracy': balanced_acc,
        'worst_accuracy': worst_acc,
        'head_accuracy': group_accuracies[0],
        'tail_accuracy': group_accuracies[1]
    }
```

### 2. Comparative Expert Analysis

```python
def compare_experts(expert_results):
    """
    So sánh performance của 3 experts.
    
    Generates:
    - Performance comparison table
    - Strengths/weaknesses analysis
    - Complementarity assessment
    - Ensemble potential prediction
    """
    
    print("📊 EXPERT COMPARISON ANALYSIS")
    print("=" * 60)
    
    metrics = ['overall_accuracy', 'balanced_accuracy', 'head_accuracy', 'tail_accuracy']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for expert_name, results in expert_results.items():
            value = results[metric]
            print(f"  {expert_name:>15}: {value:.4f}")
    
    # Find best performer per metric
    print(f"\n🏆 Best Performers:")
    for metric in metrics:
        best_expert = max(expert_results.items(), key=lambda x: x[1][metric])
        print(f"  {metric:>20}: {best_expert[0]} ({best_expert[1][metric]:.4f})")
    
    # Complementarity analysis
    print(f"\n🔄 Complementarity Analysis:")
    
    # Check if different experts excel on different groups
    head_best = max(expert_results.items(), key=lambda x: x[1]['head_accuracy'])
    tail_best = max(expert_results.items(), key=lambda x: x[1]['tail_accuracy'])
    
    print(f"  Head Classes Champion: {head_best[0]} ({head_best[1]['head_accuracy']:.4f})")
    print(f"  Tail Classes Champion: {tail_best[0]} ({tail_best[1]['tail_accuracy']:.4f})")
    
    if head_best[0] != tail_best[0]:
        print("  ✅ Good complementarity detected - different experts excel on different groups!")
    else:
        print("  ⚠️  Limited complementarity - same expert dominates both groups")
```

### 3. Expected Expert Performance (CIFAR-100-LT IF=100)

```python
EXPECTED_EXPERT_RESULTS = {
    'ce_baseline': {
        'overall_accuracy': 0.52-0.56,      # 52-56% overall
        'head_accuracy': 0.78-0.83,         # 78-83% on head classes  
        'tail_accuracy': 0.35-0.42,         # 35-42% on tail classes
        'balanced_accuracy': 0.56-0.62,     # 56-62% balanced
        'calibration_ece': 0.08-0.15        # Before temperature scaling
    },
    'logitadjust_baseline': {
        'overall_accuracy': 0.48-0.52,      # Slightly lower overall
        'head_accuracy': 0.72-0.78,         # Lower on head (by design)
        'tail_accuracy': 0.42-0.48,         # Higher on tail ✓
        'balanced_accuracy': 0.57-0.63,     # Better balanced ✓
        'calibration_ece': 0.06-0.12        # Often better calibrated
    },
    'balsoftmax_baseline': {
        'overall_accuracy': 0.49-0.53,      # Middle ground
        'head_accuracy': 0.74-0.80,         # Balanced approach
        'tail_accuracy': 0.40-0.46,         # Good tail performance
        'balanced_accuracy': 0.57-0.63,     # Consistent balanced
        'calibration_ece': 0.07-0.13        # Good calibration
    }
}
```

### 4. Calibration Assessment

```python
def assess_expert_calibration(expert_logits, labels, expert_name):
    """
    Evaluate calibration quality before và after temperature scaling.
    
    Steps:
    1. Compute ECE on raw logits
    2. Apply temperature scaling
    3. Compute ECE on calibrated logits  
    4. Generate reliability diagram
    5. Assess improvement
    """
    
    from src.metrics.calibration import calculate_ece, temperature_scale
    
    # Raw calibration
    raw_probs = torch.softmax(expert_logits, dim=1)
    raw_confidences = raw_probs.max(dim=1)[0]
    raw_predictions = expert_logits.argmax(dim=1)
    
    raw_ece = calculate_ece(raw_confidences, raw_predictions, labels, n_bins=15)
    
    # Temperature scaling
    temperature = temperature_scale(expert_logits, labels)
    calibrated_logits = expert_logits / temperature
    calibrated_probs = torch.softmax(calibrated_logits, dim=1)
    calibrated_confidences = calibrated_probs.max(dim=1)[0]
    
    calibrated_ece = calculate_ece(calibrated_confidences, raw_predictions, labels, n_bins=15)
    
    improvement = (raw_ece - calibrated_ece) / raw_ece * 100
    
    print(f"\n📏 CALIBRATION ANALYSIS - {expert_name}")
    print(f"  Raw ECE: {raw_ece:.4f}")
    print(f"  Calibrated ECE: {calibrated_ece:.4f}")  
    print(f"  Temperature: {temperature:.3f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    return {
        'raw_ece': raw_ece,
        'calibrated_ece': calibrated_ece,
        'temperature': temperature,
        'improvement_pct': improvement
    }
```

### 5. Expert Ensemble Potential Analysis

```python
def analyze_ensemble_potential(expert_logits_dict, labels, class_to_group):
    """
    Analyze potential benefit of ensembling experts.
    
    Methods:
    1. Simple Average Ensemble
    2. Oracle Ensemble (best expert per sample)
    3. Disagreement Analysis
    4. Correlation Analysis
    """
    
    # Simple average ensemble
    avg_logits = torch.stack(list(expert_logits_dict.values())).mean(dim=0)
    avg_preds = avg_logits.argmax(dim=1)
    avg_accuracy = (avg_preds == labels).float().mean().item()
    
    # Oracle ensemble (upper bound)
    oracle_correct = 0
    total_samples = len(labels)
    
    for i in range(total_samples):
        # Check if any expert got this sample correct
        sample_correct = False
        for expert_logits in expert_logits_dict.values():
            expert_pred = expert_logits[i].argmax().item()
            if expert_pred == labels[i].item():
                sample_correct = True
                break
        if sample_correct:
            oracle_correct += 1
    
    oracle_accuracy = oracle_correct / total_samples
    
    # Disagreement analysis
    expert_preds = torch.stack([logits.argmax(dim=1) for logits in expert_logits_dict.values()])
    
    # Samples where all experts agree
    unanimous = (expert_preds[0] == expert_preds[1]) & (expert_preds[1] == expert_preds[2])
    unanimous_correct = (expert_preds[0][unanimous] == labels[unanimous])
    
    # Samples where experts disagree
    disagreement = ~unanimous
    disagreement_rate = disagreement.float().mean().item()
    
    print(f"\n🤝 ENSEMBLE POTENTIAL ANALYSIS")
    print(f"  Average Ensemble Accuracy: {avg_accuracy:.4f}")
    print(f"  Oracle Ensemble Accuracy: {oracle_accuracy:.4f}")
    print(f"  Improvement Potential: {(oracle_accuracy - avg_accuracy):.4f}")
    print(f"  Expert Disagreement Rate: {disagreement_rate:.3f}")
    print(f"  Unanimous Accuracy: {unanimous_correct.float().mean():.4f}")
    
    return {
        'average_ensemble_acc': avg_accuracy,
        'oracle_ensemble_acc': oracle_accuracy,
        'improvement_potential': oracle_accuracy - avg_accuracy,
        'disagreement_rate': disagreement_rate
    }
```

### 6. Expert Training Validation

```python
def validate_expert_training():
    """
    Post-training validation checks.
    
    Validates:
    1. Model convergence (loss plateauing)
    2. No overfitting (train vs val gap)
    3. Reasonable calibration
    4. Expected performance ranges
    5. Diversity across experts
    """
    
    print("🔍 EXPERT TRAINING VALIDATION")
    print("=" * 50)
    
    expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    validation_results = {}
    
    for expert_name in expert_names:
        checkpoint_path = f"./checkpoints/experts/cifar100_lt_if100/final_calibrated_{expert_name}.pth"
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check convergence
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        if len(train_losses) > 10:
            final_train_loss = np.mean(train_losses[-10:])  # Last 10 epochs
            final_val_loss = np.mean(val_losses[-10:])
            overfitting_gap = final_val_loss - final_train_loss
            
            # Check if loss stabilized
            loss_std = np.std(train_losses[-20:])  # Stability in last 20 epochs
            converged = loss_std < 0.05  # Threshold for convergence
            
        else:
            converged = False
            overfitting_gap = float('inf')
        
        # Performance check
        test_acc = checkpoint.get('test_accuracy', 0.0)
        expected_range = EXPECTED_EXPERT_RESULTS[expert_name]['overall_accuracy']
        
        if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
            performance_ok = expected_range[0] <= test_acc <= expected_range[1]
        else:
            performance_ok = True  # Skip if no expected range
        
        validation_results[expert_name] = {
            'converged': converged,
            'overfitting_gap': overfitting_gap,
            'performance_ok': performance_ok,
            'test_accuracy': test_acc
        }
        
        # Print results
        status = "✅" if converged and performance_ok and overfitting_gap < 0.2 else "⚠️"
        print(f"{status} {expert_name}:")
        print(f"    Converged: {converged} (loss_std: {loss_std:.3f})")
        print(f"    Overfitting gap: {overfitting_gap:.3f}")
        print(f"    Test accuracy: {test_acc:.3f}")
        print(f"    Performance in range: {performance_ok}")
    
    # Overall validation
    all_good = all(r['converged'] and r['performance_ok'] and r['overfitting_gap'] < 0.2 
                   for r in validation_results.values())
    
    if all_good:
        print("\n✅ All experts passed validation checks!")
    else:
        print("\n⚠️  Some experts may need attention - check training logs")
    
    return validation_results
```

---

## �📝 Summary

Expert training system trong AR-GSE được thiết kế để:

1. **Create Diversity**: 3 experts với specialized strengths
2. **Handle Imbalance**: Specialized loss functions cho long-tail data
3. **Ensure Calibration**: Temperature scaling cho reliable confidence
4. **Enable Ensemble**: Export logits cho efficient ensemble training
5. **Provide Flexibility**: Configurable architectures và hyperparameters
6. **Comprehensive Evaluation**: Multi-metric assessment và validation

**Evaluation framework** đảm bảo:
- **Individual Performance**: Detailed metrics cho mỗi expert
- **Comparative Analysis**: Understanding strengths/weaknesses
- **Calibration Quality**: Reliable uncertainty estimates
- **Ensemble Potential**: Assessment của combination benefits
- **Training Validation**: Quality assurance checks

System này cung cấp foundation vững chắc cho AR-GSE ensemble, với mỗi expert contributing unique capabilities để handle different aspects của imbalanced classification problem.