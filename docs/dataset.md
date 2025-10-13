# 📊 AR-GSE Dataset Documentation

Tài liệu chi tiết về hệ thống xử lý dữ liệu trong AR-GSE, bao gồm luồng tạo dataset, phân chia dữ liệu, và các utility functions.

## 🎯 Tổng quan

AR-GSE sử dụng **CIFAR-100-LT** (Long-Tail) - một phiên bản imbalanced của CIFAR-100 với phân bố exponential decay để mô phỏng dữ liệu thực tế có sự mất cân bằng giữa các classes.

### Đặc điểm chính:
- **Base Dataset**: CIFAR-100 (100 classes, 32x32 RGB images)
- **Imbalance Factor**: 100 (class đầu có 100x samples so với class cuối)
- **Long-tail Distribution**: Exponential decay profile theo Cao et al., 2019
- **Methodology**: Proportional duplication cho val/test để giữ phân bố

## 🏗️ Kiến trúc Dataset System

```
src/data/
├── enhanced_datasets.py    # Core dataset creation logic
├── dataloader_utils.py     # DataLoader utilities & convenience functions  
├── datasets.py            # Basic CIFAR-100-LT utilities
├── splits.py              # Legacy split creation (deprecated)
├── groups.py              # Class grouping utilities (head/tail)
└── __init__.py
```

## 📈 Long-tail Distribution Profile

### Exponential Decay Formula
```python
n_i = n_max * (IF)^(-i/(C-1))
```

Trong đó:
- `n_i`: Số samples cho class i
- `n_max`: Số samples tối đa (500 cho CIFAR-100)
- `IF`: Imbalance Factor (default: 100)
- `C`: Tổng số classes (100)
- `i`: Class index (0 → 99)

### Phân bố thực tế:
- **Class 0** (head): 500 samples
- **Class 50** (medium): ~22 samples  
- **Class 99** (tail): 5 samples
- **Imbalance Factor**: 100:1

## 🔄 Data Pipeline Flow

### 1. Dataset Creation (`enhanced_datasets.py`)

```python
def create_full_cifar100_lt_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits",
    val_ratio: float = 0.2,
    tunev_ratio: float = 0.15,
    seed: int = 42
)
```

**Pipeline Steps:**

#### Step 1: Long-tail Training Set
```python
def create_longtail_train(cifar_train_dataset, imb_factor=100, seed=42)
```
- Subsampling từ CIFAR-100 train (50K → ~23K samples)
- Áp dụng exponential profile cho từng class
- Đảm bảo ít nhất 1 sample/class

#### Step 2: Proportional Val/Test Creation
```python
def create_proportional_test_val_with_duplication(
    cifar_test_dataset, 
    train_class_counts,
    val_ratio=0.2
)
```
- **Duplication Strategy**: Nhân đôi samples khi cần thiết
- **Proportional Matching**: Val/Test giữ nguyên tỷ lệ như Train
- **Target Size**: ~12K combined (val+test) với duplication

#### Step 3: TuneV Split Creation
```python  
def create_tunev_from_test(test_indices, test_targets, tunev_ratio=0.15)
```
- **Purpose**: Calibration và gating network pretraining
- **Source**: Subset từ test set (không leakage)
- **Size**: 15% của test set (~1.8K samples)

### 2. Final Split Structure

| Split | Purpose | Size | Source | Distribution |
|-------|---------|------|---------|-------------|
| **train** | Expert training | ~23K | CIFAR train | Long-tail (IF=100) |
| **val_lt** | Validation | ~2.4K | CIFAR test + duplication | Matches train proportions |
| **test_lt** | Final evaluation | ~10K | CIFAR test + duplication | Matches train proportions |
| **tuneV** | Calibration & gating | ~1.8K | Subset from test_lt | Matches train proportions |

## 🛠️ Core Components Deep Dive

### 1. CIFAR100LTDataset Class

```python
class CIFAR100LTDataset(Dataset):
    def __init__(self, cifar_dataset, indices, transform=None):
        self.cifar_dataset = cifar_dataset  # Original CIFAR-100
        self.indices = indices              # Subset indices
        self.transform = transform          # Data augmentation
```

**Đặc điểm:**
- **Flexible Indexing**: Mapping từ subset index → original CIFAR index
- **Transform Support**: Riêng biệt cho train/eval
- **Memory Efficient**: Không copy data, chỉ lưu indices

### 2. Data Transforms

```python
def get_cifar100_transforms():
    # Training augmentation (basic following paper)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
        )
    ])
    
    # Evaluation (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[...], std=[...])
    ])
```

**Design Principles:**
- **Minimal Augmentation**: Theo Menon et al., 2021a specification
- **Consistent Normalization**: Sử dụng CIFAR-100 statistics
- **Separate Transforms**: Train vs eval riêng biệt

### 3. Duplication Strategy

**Problem**: Original CIFAR test có 100 samples/class (balanced), nhưng cần tạo long-tail test.

**Solution**: Intelligent duplication
```python
if target_count <= available_in_test:
    # No duplication needed  
    sampled_indices = np.random.choice(cls_indices_in_test, target_count, replace=False)
else:
    # Need duplication
    duplication_factor = ceil(target_count / available_in_test)
    duplicated_pool = np.tile(cls_indices_in_test, duplication_factor)
    sampled_indices = np.random.choice(duplicated_pool, target_count, replace=False)
```

**Ví dụ:**
- **Head class**: Cần 240 samples, có 100 available → duplicate 3x → sample 240
- **Tail class**: Cần 3 samples, có 100 available → sample 3 (no duplication)

## 🔧 DataLoader Utilities

### 1. CIFAR100LTDataModule Class

**Central data management class:**
```python
class CIFAR100LTDataModule:
    def __init__(self, data_dir="data", splits_dir="data/cifar100_lt_if100_splits", 
                 batch_size=128, num_workers=4):
        # Load base CIFAR datasets
        # Setup transforms 
        # Initialize containers
        
    def setup_datasets(self):
        # Load all split indices from JSON
        # Create CIFAR100LTDataset objects
        
    def get_dataloader(self, split, batch_size=None, shuffle=None):
        # Create DataLoader with appropriate settings
        # Auto-configure shuffle (True for train, False for others)
        # Handle batch size overrides
```

**Features:**
- **Auto-configuration**: Smart defaults cho mỗi split
- **Flexible batch sizes**: Override capability
- **Memory management**: Pin memory, num_workers optimization
- **Drop last**: Chỉ cho training để đảm bảo batch size consistency

### 2. Convenience Functions

#### Expert Training
```python
def get_expert_training_dataloaders(batch_size=128, num_workers=4):
    # Return: (train_loader, val_loader)
    # Uses train + val_lt for realistic validation
```

#### AR-GSE Training  
```python
def get_argse_training_dataloaders(batch_size=64, num_workers=4):
    # Return: (tunev_loader, val_loader, test_loader)
    # Smaller batch size for complex AR-GSE training
```

#### Calibration
```python
def get_calibration_dataloader(batch_size=128, num_workers=4):
    # Return: calibration_loader (using val_lt)
    # For temperature scaling of expert models
```

## 👥 Class Grouping System

### 1. Threshold-based Grouping (Primary)

```python
def get_class_to_group_by_threshold(class_counts, threshold=20):
    # Group 0 (Head): classes với count > threshold
    # Group 1 (Tail): classes với count <= threshold
    return class_to_group  # Shape: [100]
```

**Ví dụ với threshold=20:**
- **Head group**: Classes 0-43 (44 classes, >20 samples each)
- **Tail group**: Classes 44-99 (56 classes, ≤20 samples each)

### 2. Ratio-based Grouping (Alternative)

```python
def get_class_to_group(class_counts, K=2, head_ratio=0.5):
    # Chia classes theo tỷ lệ dựa trên sample count ranking
    # K=2: binary head/tail
    # K>2: multiple groups với quantile division
```

## 📁 File Structure & Persistence

### JSON Split Files
```
data/cifar100_lt_if100_splits/
├── train_indices.json      # Training set indices (~23K)
├── val_lt_indices.json     # Validation set indices (~2.4K) 
├── test_lt_indices.json    # Test set indices (~10K)
└── tuneV_indices.json      # Calibration set indices (~1.8K)
```

**Format:**
```json
[1245, 2891, 5672, 8934, ...]  // List of CIFAR-100 indices
```

### Analysis & Statistics
```python
def analyze_distribution(indices, targets, name, train_counts=None):
    # Print detailed statistics:
    # - Total samples
    # - Head/tail class counts & percentages  
    # - Imbalance factor
    # - Group distribution (Head, Medium, Low, Tail)
    # - Comparison with train proportions (if provided)
```

## ⚡ Performance Optimizations

### 1. Memory Efficiency
- **Index-based**: Chỉ lưu indices, không duplicate image data
- **Lazy loading**: Images được load on-demand
- **Transform caching**: Transforms được apply runtime

### 2. Training Speed
- **Pin memory**: GPU transfer acceleration
- **Num workers**: Multi-process data loading  
- **Drop last**: Consistent batch sizes
- **Prefetch**: Background data loading

### 3. Reproducibility
- **Fixed seeds**: Consistent splits across runs
- **Deterministic sampling**: Reproducible random operations
- **Version control**: JSON files track exact splits used

## 🧪 Usage Examples

### 1. Basic Dataset Creation
```python
from src.data.enhanced_datasets import create_full_cifar100_lt_splits

# Create complete dataset splits
datasets, splits = create_full_cifar100_lt_splits(
    imb_factor=100,
    output_dir="data/cifar100_lt_if100_splits",
    val_ratio=0.2,
    tunev_ratio=0.15,
    seed=42
)
```

### 2. DataLoader Usage in Training
```python
from src.data.dataloader_utils import get_expert_training_dataloaders

# For expert training
train_loader, val_loader = get_expert_training_dataloaders(
    batch_size=128, 
    num_workers=4
)

for epoch in range(epochs):
    for images, labels in train_loader:
        # Training step
        ...
    
    # Validation
    with torch.no_grad():
        for images, labels in val_loader:
            # Validation step
            ...
```

### 3. Custom DataModule Usage
```python
from src.data.dataloader_utils import CIFAR100LTDataModule

# Flexible setup
data_module = CIFAR100LTDataModule(
    batch_size=64,
    num_workers=8,
    splits_dir="custom/splits/path"
)

data_module.setup_datasets()
data_module.print_dataset_stats()

# Get specific loaders
train_loader = data_module.get_dataloader('train', shuffle=True)
test_loader = data_module.get_dataloader('test', batch_size=256)
```

### 4. Class Grouping
```python
from src.data.groups import get_class_to_group_by_threshold
from src.data.dataloader_utils import CIFAR100LTDataModule

# Get class counts
data_module = CIFAR100LTDataModule()
data_module.setup_datasets()
class_counts = data_module.get_train_class_counts_list()

# Create grouping
class_to_group = get_class_to_group_by_threshold(class_counts, threshold=20)
print(f"Head classes: {(class_to_group == 0).sum()}")  # ~44
print(f"Tail classes: {(class_to_group == 1).sum()}")  # ~56
```

## 🔍 Debugging & Validation

### 1. Dataset Integrity Checks
```python
def test_dataloaders():
    """Verify all dataloaders work correctly"""
    data_module = CIFAR100LTDataModule(num_workers=0)  # Single thread for debugging
    data_module.setup_datasets() 
    data_module.print_dataset_stats()
    
    # Test batch loading
    for split in ['train', 'val', 'test', 'tunev']:
        loader = data_module.get_dataloader(split)
        batch = next(iter(loader))
        images, labels = batch
        assert images.shape[0] > 0
        assert len(labels) == images.shape[0]
```

### 2. Distribution Validation
- **Proportionality check**: Val/test distributions match train
- **No data leakage**: TuneV subset không overlap với training
- **Class coverage**: Mỗi class có ít nhất 1 sample
- **Duplication verification**: Head classes được duplicate đúng cách

### 3. Common Issues & Solutions

**Problem**: `FileNotFoundError` - Split files not found
**Solution**: Run `python create_splits.py` trước khi training

**Problem**: Memory issues với large batch sizes
**Solution**: Giảm `batch_size` hoặc `num_workers`

**Problem**: Imbalanced validation metrics
**Solution**: Sử dụng `val_lt` thay vì balanced validation

**Problem**: Data leakage giữa splits  
**Solution**: TuneV được tạo từ test, không từ train

## 📊 Dataset Statistics

### CIFAR-100-LT (IF=100) Typical Statistics:

**Train Set (~23K samples):**
- Head classes (0-43): 500-21 samples each
- Tail classes (44-99): 20-5 samples each  
- Total: 22,906 samples
- Imbalance factor: 100:1

**Val Set (~2.4K samples):** 
- Matches train proportions với duplication
- Head: 240-10 samples per class
- Tail: 10-2 samples per class

**Test Set (~10K samples):**
- Similar proportions to train
- Larger size for robust evaluation
- With duplication để maintain distributions

**TuneV Set (~1.8K samples):**
- 15% of test set
- Used for calibration và gating pretraining  
- Maintains long-tail proportions

---

## 📝 Summary

AR-GSE data system được thiết kế để:

1. **Realistic Simulation**: Long-tail distribution phản ánh real-world data
2. **Principled Methodology**: Following established papers (Cao et al., Menon et al.)  
3. **Flexible & Extensible**: Easy customization và extension
4. **Performance Optimized**: Efficient memory usage và fast loading
5. **Reproducible**: Deterministic splits với seed control

Hệ thống này cung cấp foundation vững chắc cho AR-GSE ensemble training trên imbalanced data, đảm bảo công bằng và hiệu quả trong việc đánh giá model performance.