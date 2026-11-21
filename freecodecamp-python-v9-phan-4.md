# freeCodeCamp — Python v9, phần 4: Dictionaries and Sets + Error Handling (240/588)

> Học ngày 21/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 4: **toàn khóa Python 240/588** (đầu phiên 180/588, tăng 60 bước). Số liệu đã xác minh độc lập trên trang tổng quan khóa học lúc 14:37 ngày 21/11/2026.
> Bài tiếp theo: "Classes and Objects" (0/73 bước).
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.

## 1. Dictionaries and Sets (52/53)

### Lý thuyết (7 bài)

- **Working with Dictionaries and Sets (3 bài):** dictionary là tập hợp cặp key–value, viết trong `{}`, tra cứu theo key thay vì theo vị trí như list; set là tập hợp không trùng lặp, dùng để lọc trùng và kiểm tra thuộc tính nhanh.
- **Working with Modules (1 bài):** chia code thành nhiều file/module và `import` dùng lại — cách tổ chức chương trình lớn cho gọn.
- **Working with Regular Expressions (3 bài):** regex — mẫu để tìm và trích chuỗi theo quy tắc (ví dụ tìm số điện thoại, email, mã PIN trong văn bản).

### Workshop: Build a Medical Data Validator (Step 1–44, trừ Step 22)

- Viết chương trình kiểm tra định dạng dữ liệu y tế, tìm các bản ghi không hợp lệ.
- **Step 22 đã BỎ QUA (lỗi của trang freeCodeCamp, không phải lỗi code):** yêu cầu in `print(find_invalid_records(**medical_records[0]))`; code trong editor đã đúng tuyệt đối (đã gõ lại cả hàm và dòng print) nhưng test liên tục báo chưa in, trong khi console chỉ hiện "Valid format.". Đã thử chờ, bấm Check lại nhiều lần, tải lại trang, điều hướng trong ứng dụng, Reset + gõ lại toàn bộ — đều không được. Thăm dò cho thấy bước thực thi dùng code seed cũ (thêm `print('ZZZSYNC')` không xuất hiện trong output, `print(1/0)` không gây lỗi). Seed của Step 23 đã bao gồm sẵn đáp án chuẩn của Step 22 (hàm + dòng print) nên học tiếp bình thường; workshop hiển thị 97% vì Step 22 còn "Not Passed". Dictionaries and Sets vì thế còn 52/53.
- **"Dictionaries and Sets Review" (100%) + "Dictionaries and Sets Quiz" — 20/20.**

### Certification Project: Build a User Configuration Manager ✅

- Viết chương trình quản lý cài đặt người dùng: `test_settings` đầy đủ và 4 hàm `add_setting`, `update_setting`, `delete_setting`, `view_settings`.
- Kết quả: **100%, cả 27 tests đều pass.**

## 2. Error Handling (7/7)

### Lý thuyết "Understanding Error Handling" (4 bài, đều 100%)

- **What Are Some Common Error Messages in Python?** — các lỗi thường gặp (TypeError, ValueError, IndexError…) và cách đọc thông báo lỗi.
- **What Are Some Good Debugging Techniques in Python?** — kỹ thuật gỡ lỗi: đọc traceback, in giá trị trung gian, thu hẹp phạm vi lỗi.
- **How Does Exception Handling Work?** — `try`/`except`: bọc đoạn code có thể lỗi, chương trình không sập khi gặp lỗi đã lường trước.
- **What Is the Raise Statement and How Does It Work?** — `raise`: tự mình phát lỗi có chủ đích khi dữ liệu không hợp lệ.

### Lab: Debug an ISBN Validator (20/20 test)

- Sửa chương trình kiểm tra mã ISBN bị lỗi sẵn:
  - Sửa `TypeError` (gọi `len(isbn, length)` sai cú pháp).
  - Sửa lỗi off-by-one (`isbn[0:length-1]`, `isbn[length-1]`).
  - Sửa `IndentationError` ở dòng gọi `validate_isbn` trong `main()`.
  - Thêm `try`/`except IndexError` ("Enter comma-separated values.") và `ValueError` ("Length must be a number.") trong `main()`, `try`/`except ValueError` ("Invalid character was found.") trong `validate_isbn`; comment dòng `main()` cuối file.
- Kết quả: **100%, cả 20 tests đều pass.**

### Review + Quiz

- "Error Handling Review" (100%) + "Error Handling Quiz" — **10/10**.

## Áp dụng cho công việc của Niên

- Dictionary: lưu cài đặt/trạng thái theo tên (ví dụ `settings["video_42"]`, cấu hình từng kênh, từng phòng ban) — đúng bài User Configuration Manager vừa làm.
- Regex: trích mã, số điện thoại, link từ văn bản/log — thay cho cách tách chuỗi thủ công khi canh giveaway, đọc log tải video.
- Modules: chia Niên Tool và các script phụ trợ thành từng file nhỏ, import dùng lại thay vì viết một file dài.
- `try`/`except`: script tải video, sao lưu Drive gặp lỗi mạng/file hỏng thì báo lỗi gọn và chạy tiếp, thay vì sập cả tiến trình — đúng lỗi từng gặp khi backup timeout.

## Tiến độ tổng

| Mục | Xong/Tổng |
|---|---|
| Python Basics | 144/144 ✅ |
| Install Python | 5/5 ✅ |
| Loops and Sequences | 31/31 ✅ |
| Dictionaries and Sets | 52/53 (Step 22 bỏ qua do lỗi trang) |
| Build a User Configuration Manager | Completed ✅ |
| Error Handling | 7/7 ✅ |
| Toàn khóa Python v9 | 240/588 |

Mục tiếp theo: **Classes and Objects (0/73 bước)**.
