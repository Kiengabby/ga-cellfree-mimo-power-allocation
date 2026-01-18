# Contributing Guidelines

## 🎯 Mục đích dự án

Đây là đề tài nghiên cứu của sinh viên Trường Đại học Bách Khoa Hà Nội về ứng dụng Genetic Algorithm trong tối ưu hóa phân bổ công suất cho mạng Cell-Free Massive MIMO.

## 📋 Yêu cầu

- Python 3.8+
- NumPy 1.21+
- Matplotlib 3.4+

## 🔧 Setup môi trường phát triển

```bash
# Clone repository
git clone https://github.com/your-username/ga-cellfree-mimo-power-allocation.git
cd ga-cellfree-mimo-power-allocation

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Chạy test

```bash
# Chạy code chính
python src/ga_power_allocation.py

# Chạy phiên bản đơn giản
python src/ga_power_allocation_simple.py

# So sánh biến thể
python src/ga_variant_comparison.py
```

## 📝 Code Style

- Sử dụng comment tiếng Việt để giải thích logic
- Follow PEP 8 cho Python code
- Đặt tên biến có ý nghĩa rõ ràng
- Thêm docstring cho functions và classes

## 🐛 Báo lỗi

Nếu phát hiện lỗi, vui lòng tạo Issue với thông tin:
- Mô tả lỗi
- Các bước tái hiện
- Kết quả mong đợi vs thực tế
- Môi trường (Python version, OS)

## 💡 Đề xuất cải tiến

Chúng tôi hoan nghênh mọi đóng góp để cải thiện thuật toán:
- Tối ưu hóa hiệu năng
- Thêm phương pháp chọn lọc mới
- Cải tiến toán tử lai ghép/đột biến
- Thêm visualization mới

## 📧 Liên hệ

- Hoàng Mạnh Kiên: kien.hm215068@sis.hust.edu.vn
- Trần Trung Đức: duc.tt210210@sis.hust.edu.vn

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.
