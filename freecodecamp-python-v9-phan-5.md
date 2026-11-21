# freeCodeCamp — Python v9, phần 5: Classes and Objects (313/588)

> Học ngày 21/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 5: **toàn khóa Python 313/588** (đầu phiên 240/588, tăng 73 bước). Số liệu đã xác minh độc lập trên trang tổng quan khóa học lúc 16:26 ngày 21/11/2026.
> Bài tiếp theo: "Object-Oriented Programming (OOP)" (0/124 bước).
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.

## 1. Classes and Objects (72/73)

### Lý thuyết (4/4 bài)

- Class là khuôn mẫu để tạo object; object là một thực thể cụ thể có thuộc tính (attributes) và hành vi (methods).
- `__init__` khởi tạo object với các thuộc tính ban đầu; `self` trỏ tới chính object đang làm việc.
- Methods là hàm gắn trong class, dùng để thao tác với dữ liệu của object.

### Workshop: Build a Musical Instrument Inventory (13/13 bước)

- Viết chương trình quản lý kho nhạc cụ bằng class: thêm, tìm, liệt kê nhạc cụ theo loại, giá, số lượng.

### Lab: Build a Planet Class (19/19 test)

- Viết class `Planet` với các thuộc tính và phương thức mô tả hành tinh; tất cả 19 test đều pass.

### Workshop: Build an Email Simulator (53/53 bước)

- Viết chương trình mô phỏng hộp thư: gửi, nhận, liệt kê, đánh dấu đã đọc email.
- Lỗi đã sửa trong lúc làm: lỗi thụt lề (indentation) ở Step 36–37 và lỗi f-string ở Step 46.

### Review (100%)

- "Classes and Objects Review" hoàn thành 100%.

### Certification Project: Build a Budget App ✅

- Viết class `Category` với các phương thức `deposit`, `withdraw`, `get_balance`, `transfer`, `check_funds`, `__str__`, và hàm `create_spend_chart` vẽ biểu đồ chi tiêu theo danh mục.
- Kết quả: **100%, cả 25 tests đều pass, đã nộp** (trang hồ sơ ghi ngày hoàn thành 21/11/2026).

### Quiz — CHƯA LÀM (1 bước còn lại của mục)

- "Classes and Objects Quiz" (10 câu, cần đúng ít nhất 9/10) đã bị bỏ qua trong phiên này vì người làm nhầm tưởng là bước thi cử. Thực tế đây là quiz cuối mục như các quiz trước (Python Basics 20/20, Install Python 10/10, Loops 19/20, Dictionaries 20/20, Error Handling 10/10) — KHÔNG phải kỳ thi có giám sát. **Sẽ làm bù đầu phiên sau**, rồi học tiếp mục OOP.

## Áp dụng cho công việc của Niên

- Class/object: mô hình hoá đúng việc Niên làm hằng ngày — một video là một object (tiêu đề, mô tả, trạng thái, lượt xem), một kênh là một object (danh sách video, lịch đăng), một cài đặt là một object.
- Budget App: mẫu quản lý tiền/điểm/credit theo danh mục (credit Flow, điểm Thematic, ngân sách từng kênh) — đúng bài toán theo dõi chi tiêu của Niên.
- Email Simulator: mẫu xử lý hộp thư — gửi/nhận/đánh dấu thư, gần với việc Niên đọc và phân loại email, tin nhắn các phòng.

## Tiến độ tổng

| Mục | Xong/Tổng |
|---|---|
| Python Basics | 144/144 ✅ |
| Install Python | 5/5 ✅ |
| Loops and Sequences | 31/31 ✅ |
| Dictionaries and Sets | 52/53 (Step 22 bỏ qua do lỗi trang) |
| Build a User Configuration Manager | Completed ✅ |
| Error Handling | 7/7 ✅ |
| Classes and Objects | 72/73 (Quiz chưa làm, làm bù phiên sau) |
| Build a Budget App | Completed ✅ |
| Toàn khóa Python v9 | 313/588 |

Mục tiếp theo: **Object-Oriented Programming (OOP) (0/124 bước)** — dự án tiếp theo là Build a Polygon Area Calculator.
