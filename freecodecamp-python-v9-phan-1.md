# freeCodeCamp — Python v9, phần 1: Python Basics (62/144)

> Học ngày 21/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 1: **Python Basics 62/144, toàn khóa Python 62/588.**
> Bài tiếp theo: "Build a Bill Splitter" (Workshop), Step 1.
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.

## 1. Introduction to Python (1 bài)

- Python là gì: ngôn ngữ lập trình bậc cao, cú pháp gần tiếng Anh, đọc hiểu nhanh, dùng rộng rãi trong tự động hoá, xử lý dữ liệu, web, AI.
- Ứng dụng thường gặp trong công việc: viết script tự động hoá (xử lý file, video, âm thanh), tính toán kỹ thuật, làm công cụ nội bộ — đúng hướng Niên đang làm (ffmpeg, TTS, tool CLI).

## 2. Understanding Variables and Data Types (4 bài)

- Biến là tên gắn với một giá trị; đặt tên rõ nghĩa, viết thường, nối bằng gạch dưới (`ten_bien`).
- Các kiểu dữ liệu cơ bản:
  - `str` — chuỗi ký tự, viết trong dấu nháy đơn hoặc kép: `"Niên"`.
  - `int` — số nguyên: `5`, `2026`.
  - `float` — số thực: `3.5`, `9.99`.
  - `bool` — đúng/sai: `True`, `False`.
  - `None` — không có giá trị.
- Kiểm tra kiểu bằng `type()`. Python tự suy kiểu khi gán — không cần khai báo kiểu trước.
- Chuyển kiểu khi cần: `str()`, `int()`, `float()` — ví dụ nối số vào chuỗi phải đổi số sang chuỗi trước.

## 3. Lab: Build a Report Card Printer (Step 1–10)

- Bài lab đầu tiên: viết chương trình in ra một phiếu báo điểm (report card) — tên học sinh, các môn, điểm số — bằng `print()` và biến.
- Thực hành: khai báo biến cho từng thông tin, in ra theo đúng định dạng yêu cầu, dùng f-string để chèn biến vào chuỗi.

## 4. Introduction to Strings (4 bài)

- Chuỗi là dãy ký tự; nối chuỗi bằng `+`, lặp chuỗi bằng `*`.
- f-string: đặt `f` trước chuỗi, chèn biến/trị trong `{}` — cách in biến vào câu gọn nhất trong Python:
  - `f"Tên: {ten}"`
- Chuỗi có thứ tự (index bắt đầu từ 0), lấy ký tự bằng chỉ số, lấy đoạn bằng slicing.
- Hàm `len()` đếm độ dài chuỗi; `input()` đọc dữ liệu người dùng gõ vào (luôn trả về chuỗi).

## 5. Lab: Build an Employee Profile Generator (Step 1–18)

- Viết chương trình tạo hồ sơ nhân viên: nhận tên, tuổi, chức vụ, lương… rồi in ra một hồ sơ gọn gàng, đúng định dạng.
- Thực hành: biến + chuỗi + f-string + `input()` + chuyển kiểu (`int()` cho tuổi/lương khi cần tính toán).

## 6. Lab: Build an Employee Profile Validator (Step 1–22)

- Viết chương trình kiểm tra tính hợp lệ của hồ sơ nhân viên: tên không được rỗng, tuổi phải trong khoảng hợp lý, các trường bắt buộc phải có giá trị…
- Thực hành: điều kiện `if/else`, kiểm tra độ dài chuỗi, kiểm tra kiểu/giá trị, in thông báo lỗi rõ ràng khi dữ liệu không hợp lệ.
- Lưu ý kỹ thuật gặp trong lúc làm: con trỏ soạn thảo đôi khi rơi vào dòng bị xuống dòng tự động làm lệch code (Step 3, 10–11, 15) — phải sửa lại cho đúng rồi mới qua kiểm tra của freeCodeCamp.

## 7. Numbers and Mathematical Operations (3 bài)

- Phép toán số học: `+` `-` `*` `/` (chia lấy thực), `//` (chia lấy phần nguyên), `%` (chia lấy dư), `**` (lũy thừa).
- Thứ tự ưu tiên phép toán như toán học thông thường; dùng ngoặc `()` để ép thứ tự.
- Định dạng số trong f-string (bài cuối phần 1: "How Do You Format Numbers in an F-String?"):
  - `f"{gia:.2f}"` — in số thực với 2 chữ số thập phân.
  - `f"{so:,}"` — thêm dấu phân cách hàng nghìn.
  - Căn lề, thêm số 0 ở đầu khi cần.
- Làm tròn: `round(so, n)` — làm tròn tới n chữ số thập phân.

## Áp dụng cho công việc của Niên

- Python là ngôn ngữ của các script tự động hoá Niên đang dùng hằng ngày (ffmpeg, TTS, xử lý file video/âm thanh, tool CLI).
- f-string + định dạng số: dùng ngay khi in log, báo cáo tiến độ, tạo tên file có số thứ tự (video 43, batch…).
- Validator (kiểm tra dữ liệu đầu vào): mẫu để viết các bước kiểm tra trong tool — ví dụ kiểm tra file video đã đủ, tên file đúng chuẩn, trước khi đăng.
- Bài tiếp theo "Build a Bill Splitter" sẽ luyện chia tiền/tính toán — gần với tính toán điện (chia tải, chia công suất).

## Tiến độ tổng

| Mục | Xong/Tổng |
|---|---|
| Python Basics | 62/144 |
| Toàn khóa Python v9 | 62/588 |

Các mục sau trong Python Basics: Build a Bill Splitter → Loops and Sequences → Dictionaries and Sets → … (xem trang khóa học).
