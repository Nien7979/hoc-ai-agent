# freeCodeCamp — Python v9, phần 3: Install Python + Loops and Sequences (180/588)

> Học ngày 21/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 3: **toàn khóa Python 180/588** (đầu phiên 144/588, tăng 36 bước).
> Bài tiếp theo: "Dictionaries and Sets" (0/53 bước).
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.

## 1. Install Python (5/5)

- Lý thuyết "Installing Python and Running Code Locally" (3 bài):
  - Cài đặt, cấu hình và dùng Python trong môi trường máy mình.
  - Cách chạy script Python (chạy file `.py` từ dòng lệnh).
  - Cách dùng Python Interactive Shell (gõ lệnh thử ngay, không cần tạo file).
- "Python Installation Review" + "Python Installation Quiz" — **10/10**.

## 2. Loops and Sequences (31/31)

### Lý thuyết "Working with Loops and Sequences" (9 bài)

- **List (danh sách):** dãy giá trị có thứ tự, viết trong `[]`, đổi được nội dung (thêm, xoá, sửa phần tử). Các phương thức thường dùng: `append()` (thêm cuối), `remove()`, `pop()`, `sort()`…
- **Tuple:** giống list nhưng viết trong `()` và KHÔNG đổi được sau khi tạo — dùng cho dữ liệu cố định (ví dụ toạ độ, cấu hình).
- **Vòng lặp:** `for` chạy qua từng phần tử của một dãy; `while` chạy lặp khi điều kiện còn đúng.
- **`range()`:** tạo dãy số để dùng trong vòng lặp (ví dụ `range(5)` cho 0–4).
- **`enumerate()`:** lặp kèm cả số thứ tự và giá trị; **`zip()`:** lặp đồng thời hai danh sách song song.
- **List comprehension:** tạo list mới gọn trong một dòng, ví dụ `[x * 2 for x in danh_sach]`.
- **Lambda:** hàm ẩn danh một dòng, ví dụ `lambda x: x * 2` — hay dùng làm tham số cho `sort()`, `map()`, `filter()`.

### Workshop: Build a PIN Extractor (Step 1–19)

- Viết chương trình trích mã PIN (dãy số) ẩn trong các đoạn thơ.
- Kết quả cuối in ra: `['5202', '3346', '50000']`.
- Lưu ý kỹ thuật gặp trong lúc làm: ở Step 17, test của freeCodeCamp yêu cầu đưa toàn bộ nội dung hàm (kể cả `secret_code = ''` và `return`) vào trong vòng lặp `for poem in poems:`; ở Step 19, PIN của đoạn thơ đầu phải là `'5202'`, nên dòng thơ thứ ba giữ đúng là "white and" (không thêm từ "bright").

### Lab: Build a Number Pattern Generator (8/8 test)

- Viết chương trình sinh ra dãy mẫu số theo quy tắc cho trước.
- Kết quả: **8/8 test đạt**.

### Review + Quiz

- "Loops and Sequences Review" + "Loops and Sequences Quiz" — **19/20**.
- Trang hiện thông báo: "Nicely done. You just completed Loops and Sequences."

## Áp dụng cho công việc của Niên

- List + vòng lặp: xử lý danh sách file video, danh sách bài đăng, danh sách link — đúng việc Niên làm hằng ngày (duyệt từng video trong queue, từng file trong thư mục backup).
- `enumerate()` + `zip()`: đánh số thứ tự video (video 37, 38…) khi duyệt song song hai danh sách (tên file + tiêu đề).
- List comprehension: lọc nhanh danh sách (ví dụ chỉ lấy file `.mp4` trong một thư mục) trong một dòng.
- Lambda + `sort()`: sắp xếp video theo ngày, theo lượt xem khi làm báo cáo.
- PIN Extractor: mẫu trích thông tin từ văn bản — gần với việc trích số liệu từ log, trích mã từ trang web khi canh giveaway.

## Tiến độ tổng

| Mục | Xong/Tổng |
|---|---|
| Python Basics | 144/144 ✅ |
| Install Python | 5/5 ✅ |
| Loops and Sequences | 31/31 ✅ |
| Toàn khóa Python v9 | 180/588 |

Mục tiếp theo: **Dictionaries and Sets (0/53 bước)**.
