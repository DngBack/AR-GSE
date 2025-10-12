# CIFAR-100-LT Dataset Splits Creation Scripts

Bộ script để tạo các splits dữ liệu cho CIFAR-100-LT (Long-Tail) dataset.

## Scripts có sẵn

### 1. `create_splits.py` - Script cơ bản

Script đơn giản để tạo splits với cấu hình mặc định.

**Cách sử dụng:**
```bash
python create_splits.py
```

**Cấu hình mặc định:**
- Imbalance Factor: 100
- Output Directory: `data/cifar100_lt_if100_splits/`
- Validation Ratio: 0.2 (20%)
- TuneV Ratio: 0.15 (15%)
- Random Seed: 42

### 2. `create_splits_advanced.py` - Script nâng cao

Script với khả năng tùy chỉnh các tham số qua command-line arguments.

**Cách sử dụng:**
```bash
# Sử dụng cấu hình mặc định
python create_splits_advanced.py

# Tùy chỉnh imbalance factor
python create_splits_advanced.py --imb-factor 50

# Tùy chỉnh thư mục output
python create_splits_advanced.py --output-dir "data/custom_splits"

# Tùy chỉnh tỷ lệ validation và tuneV
python create_splits_advanced.py --val-ratio 0.25 --tunev-ratio 0.1

# Thay đổi random seed
python create_splits_advanced.py --seed 123

# Kết hợp nhiều tham số với verbose output
python create_splits_advanced.py --imb-factor 200 --val-ratio 0.3 --seed 100 --verbose
```

**Các tham số có sẵn:**
- `--imb-factor`: Hệ số mất cân bằng (default: 100.0)
- `--output-dir`: Thư mục đầu ra (default: "data/cifar100_lt_if100_splits")
- `--val-ratio`: Tỷ lệ validation set (default: 0.2)
- `--tunev-ratio`: Tỷ lệ tuneV set (default: 0.15)
- `--seed`: Random seed (default: 42)
- `--verbose`: Hiển thị thông tin chi tiết
- `--help`: Hiển thị hướng dẫn sử dụng

## Output

Cả hai script sẽ tạo ra:

### Files JSON
- `train_indices.json`: Indices cho training set
- `val_lt_indices.json`: Indices cho validation set
- `test_lt_indices.json`: Indices cho test set  
- `tuneV_indices.json`: Indices cho tuning/validation set

### Dataset Objects
Trả về dictionary chứa PyTorch Dataset objects:
- `train`: Training dataset với augmentation
- `val`: Validation dataset
- `test`: Test dataset
- `tunev`: TuneV dataset

## Methodology

Scripts sử dụng phương pháp:

1. **Train Set**: Tạo long-tail distribution với exponential profile
2. **Val/Test Sets**: Tạo phân bố tỷ lệ với train set thông qua duplication
3. **TuneV Set**: Subset từ test set để tránh data leakage

## Thống kê Distribution

Sau khi tạo, scripts sẽ hiển thị:
- Số lượng samples cho mỗi split
- Imbalance factor thực tế
- Phân bố theo nhóm classes (Head, Medium, Low, Tail)
- So sánh tỷ lệ giữa các splits

## Lỗi thường gặp

1. **Import Error**: Đảm bảo thư mục `src/` tồn tại với các modules cần thiết
2. **Download Error**: Kiểm tra kết nối internet để tải CIFAR-100
3. **Permission Error**: Đảm bảo có quyền ghi vào thư mục output
4. **Disk Space**: Cần khoảng 500MB+ cho CIFAR-100 và splits

## Ví dụ sử dụng

### Tạo splits cho các IF khác nhau
```bash
# IF = 50
python create_splits_advanced.py --imb-factor 50 --output-dir "data/cifar100_lt_if50_splits"

# IF = 200  
python create_splits_advanced.py --imb-factor 200 --output-dir "data/cifar100_lt_if200_splits"
```

### Thử nghiệm với tỷ lệ khác nhau
```bash
# Tăng validation set lên 30%
python create_splits_advanced.py --val-ratio 0.3

# Giảm tuneV set xuống 10%
python create_splits_advanced.py --tunev-ratio 0.1
```