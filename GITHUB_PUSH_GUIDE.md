# 🚀 HƯỚNG DẪN PUSH LÊN GITHUB

## Bước 1: Tạo repository trên GitHub

1. Truy cập: https://github.com/new
2. Điền thông tin:
   - **Repository name**: `ga-cellfree-mimo-power-allocation`
   - **Description**: `Genetic Algorithm for Power Allocation in Cell-Free Massive MIMO Networks`
   - **Visibility**: Chọn **Public** (để thầy xem được)
   - **KHÔNG** tick "Initialize with README" (vì đã có rồi)
3. Click **Create repository**

## Bước 2: Kết nối local repo với GitHub

Mở Terminal và chạy các lệnh sau:

```bash
cd "/Users/manhkien/Documents/Kỹ thuật truyền thông"

# Thêm remote repository (thay YOUR_USERNAME bằng username GitHub của bạn)
git remote add origin https://github.com/YOUR_USERNAME/ga-cellfree-mimo-power-allocation.git

# Đổi tên branch từ master sang main (chuẩn hiện tại)
git branch -M main

# Push code lên GitHub
git push -u origin main
```

## Bước 3: Verify trên GitHub

1. Reload trang GitHub repository
2. Kiểm tra các file đã được push:
   - ✅ README.md với badges đẹp
   - ✅ src/ folder với 3 files Python
   - ✅ results/ folder với 4 ảnh
   - ✅ docs/ folder với LaTeX
   - ✅ LICENSE, requirements.txt, .gitignore

## Bước 4: Cập nhật README với link đúng

Sau khi push, cập nhật dòng clone trong README.md:

```bash
# Thay YOUR_USERNAME bằng username thật của bạn
git clone https://github.com/YOUR_USERNAME/ga-cellfree-mimo-power-allocation.git
```

Commit và push lại:
```bash
git add README.md
git commit -m "docs: update clone URL"
git push
```

## Bước 5: Gửi link cho thầy

📧 **Gửi email cho thầy với nội dung:**

```
Kính gửi Thầy Trịnh Văn Chiến,

Em là Hoàng Mạnh Kiên (20215068) và Trần Trung Đức (20210210).

Em xin gửi Thầy link source code đề tài:
🔗 https://github.com/YOUR_USERNAME/ga-cellfree-mimo-power-allocation

Repository bao gồm:
- Source code Python với comment tiếng Việt đầy đủ
- Kết quả thực nghiệm (biểu đồ)
- Báo cáo LaTeX
- Hướng dẫn cài đặt và chạy code

Em xin chân thành cảm ơn Thầy!

Trân trọng,
Hoàng Mạnh Kiên & Trần Trung Đức
```

## 📝 Lưu ý quan trọng

### Nếu bị lỗi authentication:
GitHub không còn cho phép push bằng password. Bạn cần dùng **Personal Access Token**:

1. Vào: https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Chọn quyền: `repo` (full control)
4. Copy token (chỉ hiện 1 lần!)
5. Khi push, dùng token thay cho password

### Nếu muốn dùng SSH:
```bash
# Tạo SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Thêm vào GitHub: https://github.com/settings/keys

# Đổi remote sang SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ga-cellfree-mimo-power-allocation.git
```

## 🎉 Hoàn thành!

Repository của bạn giờ đã:
- ✅ Cấu trúc chuyên nghiệp
- ✅ README đẹp với badges
- ✅ Code có comment đầy đủ
- ✅ Kết quả thực nghiệm
- ✅ License MIT
- ✅ .gitignore chuẩn Python
- ✅ requirements.txt đầy đủ

Chúc bạn bảo vệ thành công! 🎓
