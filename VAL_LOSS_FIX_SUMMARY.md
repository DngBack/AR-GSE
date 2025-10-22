# Tổng kết fix val loss issue

## Vấn đề
Val loss = 2.02 trong khi train NLL = 0.005 (chênh 400x)

## Nguyên nhân
Load-balancing loss đang được bật cho **dense routing**, điều này không đúng:
- Dense routing: tất cả experts được dùng với weights > 0
- Load-balancing loss: thiết kế cho **sparse routing (top-k)** để tránh collapse
- Khi apply LB cho dense với top_k=1 → tính sai logic

## Các thay đổi đã thực hiện

### 1. Tắt load-balancing cho dense routing
File: `src/train/train_gating_map.py`

```python
# Trước:
use_load_balancing=config['gating']['use_load_balancing'],

# Sau:
use_load_balancing=config['gating']['use_load_balancing'] and config['gating']['routing'] == 'top_k',
```

**Lý do**: Dense routing không cần load-balancing vì không có routing collapse.

### 2. Thêm logging cho loss components trong validation
```python
# In ra cả NLL và total loss để debug
print(f"Val Loss: {val_metrics['loss']:.4f} (NLL={val_metrics['nll']:.4f})")
```

### 3. Thêm info print về loss configuration
```python
print(f"\n⚙️  Loss Configuration:")
print(f"   Mixture NLL: ✓")
print(f"   Load-balancing: {'✓' if use_lb else '✗ (disabled for dense routing)'}")
print(f"   Entropy reg: {'✓' if config['gating']['use_entropy_reg'] else '✗'}")
```

## Kết quả mong đợi

Sau khi fix:
- **Dense routing**: Val loss ≈ Train NLL + small entropy term (0.005-0.01)
- **Top-K routing**: Val loss = NLL + LB + Entropy (có thể cao hơn chút)

## Lý thuyết MoE về load-balancing

**Khi nào dùng load-balancing?**
- ✓ Sparse routing (Top-1, Top-K): Tránh routing collapse
- ✗ Dense routing (Softmax over all): Không cần vì tất cả experts luôn được dùng

**Switch Transformer (Fedus et al. 2021)**:
- Top-1 routing với load-balancing loss
- Công thức: L_LB = α · E · Σ_i f_i · P_i
- f_i: fraction of tokens routed to expert i
- P_i: average router probability for expert i

**Dense MoE (Jordan & Jacobs 1994)**:
- Softmax weights over all experts
- Không cần load-balancing
- Có thể dùng entropy regularization để khuyến khích diversity
