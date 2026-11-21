# freeCodeCamp — Python v9, phần 6: Object-Oriented Programming (439/588)

> Học ngày 21–22/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 6: **toàn khóa Python 439/588** (đầu phiên 313/588, tăng 126 bước). Số liệu đã xác minh độc lập trên trang tổng quan khóa học lúc 04:47 ngày 22/11/2026.
> Bài tiếp theo: "Linear Data Structures" (0/32 bước), rồi dự án chứng chỉ "Build a Hash Table".
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.
> Không có bước nào bị bỏ qua trong phiên này: không gặp bước cần token, thi có giám sát hay lỗi trang. Cửa sổ quyên góp của freeCodeCamp chỉ đóng bằng "Ask me later".

## 1. Classes and Objects — hoàn thành (73/73)

### Quiz — đúng 10/10 (100%)

- "Classes and Objects Quiz" (10 câu) — làm bù đầu phiên 6, đúng hết 10/10. Đây là quiz cuối mục bình thường, không phải kỳ thi có giám sát.
- Mục Classes and Objects từ 72/73 lên 73/73.

## 2. Object-Oriented Programming — hoàn thành (124/124)

### Lý thuyết: Understanding OOP and Encapsulation (2/2 bài)

- OOP là cách tổ chức code quanh các object: gom dữ liệu (attributes) và hành vi (methods) vào cùng một class.
- Encapsulation (đóng gói): giấu trạng thái bên trong của object, chỉ cho bên ngoài tương tác qua methods công khai — giúp code dễ bảo trì, tránh sửa trực tiếp thuộc tính gây lỗi.

### Workshop: Build a Salary Tracker (41/41 bước)

- Viết chương trình quản lý lương bằng class: lưu nhân viên, tính lương theo giờ làm và phụ cấp, in bảng tổng kết.

### Lab: Build a Game Character Stats Tracker (44/44 test)

- Viết class quản lý chỉ số nhân vật game (máu, sức mạnh, cấp độ…); tất cả 44 test đều pass.

### Lý thuyết: Understanding Inheritance and Polymorphism (3/3 bài)

- Inheritance (kế thừa): class con dùng lại code của class cha và bổ sung/thay đổi hành vi riêng — tránh lặp code.
- Polymorphism (đa hình): nhiều class khác nhau cùng có một method tên giống nhau, gọi cùng một cách nhưng chạy khác nhau tùy object.

### Workshop: Build a Media Catalogue (42/42 bước)

- Viết chương trình quản lý danh mục media (phim, nhạc, sách) bằng class cha và các class con kế thừa.

### Lý thuyết: Understanding Abstraction (1/1 bài, đúng 3/3 câu)

- Abstraction (trừu tượng hóa): chỉ cho người dùng thấy những gì cần dùng, giấu chi tiết cài đặt bên trong — ví dụ dùng method mà không cần biết nó tính toán ra sao.

### Workshop: Build a Discount Calculator (31/31 bước)

- Viết chương trình tính giá sau chiết khấu cho từng loại người dùng; terminal in ra: "Best price for Wireless Mouse for Premium user: $40.00".

### Lab: Build a Player Interface (25/25 test)

- Viết class giao diện người chơi; tất cả 25 test đều pass.

### Review + Quiz

- "OOP Review" hoàn thành 100%.
- "OOP Quiz" đúng 20/20 (100%).

### Certification Project: Build a Polygon Area Calculator ✅

- Viết class `Rectangle` với các phương thức `set_width`, `set_height`, `get_area`, `get_perimeter`, `get_diagonal`, `get_picture`, `get_amount_inside`, `__str__`.
- Viết class `Square` kế thừa `Rectangle`, thêm `set_side`, ghi đè `set_width`/`set_height` (đặt cạnh hình vuông), và `__str__` riêng.
- Kết quả: **100%, cả 22 tests đều pass, đã nộp** — trang khóa học hiển thị "Completed Certification Project".

## 3. Áp dụng vào công việc

- Tư duy OOP (class, kế thừa, đóng gói) dùng ngay khi viết công cụ/app Python: tách từng phần thành class riêng (ví dụ class quản lý video, class quản lý phụ đề) thay vì viết hàm rời rạc.
- Dự án Polygon Area Calculator là mẫu tốt cho bài toán hình học trong app tính toán điện (diện tích, chu vi) — cùng một kiểu class kế thừa.
