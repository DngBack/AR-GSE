# Ghi chú chi tiết: Quan hệ giữa `threshold (t)` và `cost (c)` trong đánh giá AURC

## 1. Bối cảnh
Trong phiên bản đánh giá AURC cũ, ta quét biến `cost` từ 0→1 và áp điều kiện nhận mẫu dạng:
```
margin + c >= 0   ⇔   margin >= -c
```
Sau đó chuyển sang **dynamic threshold sweep**, trực tiếp quét `t` (ngưỡng) trên phân vị margin để kiểm soát coverage, vì phân phối margin lệch mạnh về phía âm. Điều này làm xuất hiện các giá trị âm ở cột vốn đang đặt tên là `cost`, gây nhầm lẫn.

## 2. Hai biểu diễn tương đương của luật chấp nhận
| Biểu diễn | Dạng điều kiện | Biến quét | Ghi chú |
|-----------|----------------|-----------|--------|
| Cost-form | margin + c >= 0 | c ≥ 0     | Dễ diễn giải như "phạt" cho reject, nhưng coverage thay đổi phi tuyến nếu margin lệch |
| Threshold | margin ≥ t      | t ∈ ℝ (thường âm) | Quét trực tiếp phân vị giúp coverage trải đều |

## 3. Quan hệ chuyển đổi
Ta luôn có:

$$ t = -c \quad \text{và} \quad c = -t $$

Vì margin chủ yếu âm (ví dụ min ≈ -1.25, max ≈ 0.38), để chấp nhận nhiều hơn ta phải **giảm ngưỡng** (làm t nhỏ hơn, ví dụ -1.0) → tương đương tăng `c` (≈ 1.0) trong cost-form.

| Tình huống | margin phân bố | threshold t | cost c | Coverage |
|------------|----------------|-------------|--------|----------|
| Rất chọn lọc | Margin phần lớn < t | t gần 0 âm nhẹ (≈ -0.28) | c ≈ 0.28 | Thấp (~5%) |
| Trung bình | t ≈ -0.70 | c ≈ 0.70 | Vừa (~50%) |
| Rộng | t ≈ -1.05 | c ≈ 1.05 | Cao (~95%) |

## 4. Công thức coverage và risk
- Coverage: \( \text{cov}(t) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\text{margin}_i \ge t] \)
- Risk (Error trên phần chấp nhận): \( \text{risk}(t) = \frac{\sum_{i=1}^N \mathbf{1}[\text{margin}_i \ge t] \cdot \mathbf{1}[y_i \neq \hat{y}_i]}{\sum_{i=1}^N \mathbf{1}[\text{margin}_i \ge t]} \)
- AURC (full): tích phân xấp xỉ: \( \int_0^1 \text{risk}(\text{cov})\, d\text{cov} \) bằng quy tắc hình thang trên tập điểm quét.

## 5. Vì sao xuất hiện "cost âm"
Khi bật dynamic threshold sweep nhưng vẫn dùng tên cột `cost`, thực chất dữ liệu đang là `t` (ngưỡng) → thường âm. Nếu người đọc tưởng đó là `c`, sẽ thấy "c âm" vô lý. Giải pháp: đổi tên cột thành `threshold` và nếu cần thêm `cost_equivalent = -threshold` để ánh xạ ngược.

## 6. Quy ước đặt tên cột khuyến nghị
Khi `dynamic_enabled = True`:
- `threshold`: giá trị t được quét (có thể âm).
- `coverage`: tỉ lệ chấp nhận.
- `risk`: sai số trên phần chấp nhận.
- `cost_equivalent`: (tùy chọn) = `-threshold` (luôn ≥ 0 nếu muốn so sánh với phiên bản cũ).

Khi `dynamic_enabled = False` (cost sweep cũ):
- `cost`: giá trị quét (≥ 0).
- `coverage`, `risk` giữ nguyên.

## 7. Chú thích figure nên có
Thêm một textbox nhỏ ở subplot bar chart hoặc góc trên bên phải:
```
Threshold sweep: accept if margin ≥ t
Cost-form equivalence: c = -t (c ≥ 0)
```
→ Giúp người đọc mới hiểu ngay chuyển đổi.

## 8. Gợi ý đổi tiêu đề / legend
| Thành phần | Hiện tại | Đề xuất |
|------------|----------|---------|
| Subplot 1 title | Error vs Rejection Rate (0-1) | Risk (Error) vs Proportion of Rejections (0–1) |
| Subplot 2 title | Error vs Rejection Rate (0-0.8) | Zoom: Risk vs Rejections (0–0.8) |
| Subplot 3 title | AURC Comparison (Full vs 0.2-1.0) | Threshold Sweep AURC (Full vs 0.2–1.0) |
| Bar legend | Full (0-1) / Practical (0.2-1.0) | Full Range / Practical Range |
| Annotation | Worst full AURC +X% vs Balanced | Giữ, thêm dòng chuyển đổi t↔c |

## 9. Patch code mẫu (chỉ minh hoạ, không phải diff đầy đủ)
```python
# Trong hàm plot_aurc_curves(...)
ax3.set_title('Threshold Sweep AURC (Full vs 0.2-1.0)', fontweight='bold')
# Thêm ghi chú chuyển đổi
ax3.text(0.02, 0.02, 'Accept if margin ≥ t\nCost-form: c = -t', transform=ax3.transAxes,
         fontsize=9, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
```

Trong phần lưu CSV:
```python
row = {
    'metric': metric,
    'threshold': sweep_val,
    'coverage': coverage,
    'risk': risk,
    'cost_equivalent': -sweep_val
}
```

## 10. Checklist nhanh để đồng bộ
- [x] Thêm cột `threshold` thay vì `cost` khi dynamic.
- [x] (Tuỳ chọn) Thêm `cost_equivalent` để truy vết.
- [ ] Sửa tiêu đề subplot 3.
- [ ] Thêm textbox chú thích t↔c.
- [ ] Rà soát lại mọi print: tránh nói "cost" khi đang dynamic sweep.
- [ ] README / docs bổ sung mục này (file hiện tại đáp ứng cơ bản).

## 11. FAQ
**Q:** Có nên chuẩn hoá margin về [0,1] trước khi quét?  
**A:** Không bắt buộc; quét theo phân vị trực tiếp bảo toàn thứ tự và tương quan với quyết định chấp nhận.

**Q:** Nếu phân bố margin thay đổi sau fine-tune gating?  
**A:** Chỉ cần xây lại lưới `threshold` bằng percentiles trên validation mới; không cần đổi công thức.

**Q:** Có cần thêm điểm coverage đúng 0 hoặc 1?  
**A:** Hàm tính AURC đã chèn endpoint nếu thiếu; vẫn nên đảm bảo lưới đủ rộng để không nội suy méo.

**Q:** Vì sao AURC thực tế (0.2–1.0) luôn nhỏ hơn full?  
**A:** Vì phần coverage rất thấp (0–0.2) thường có rủi ro cao hoặc không ổn định; loại bỏ làm giảm diện tích dưới đường.

---
*File này nhằm làm rõ mọi nhầm lẫn giữa `cost` và `threshold` trong pipeline đánh giá AR-GSE.*
