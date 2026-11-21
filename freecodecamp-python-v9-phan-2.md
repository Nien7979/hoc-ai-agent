# freeCodeCamp — Python v9, phần 2: Python Basics (144/144) ✅

> Học ngày 21/11/2026 trên tài khoản chính fcc-07869125 (lehuunien95@gmail.com).
> Tiến độ cuối phần 2: **Python Basics 144/144 (100%), toàn khóa Python 144/588.**
> Phiên này làm từ 62/144 lên 144/144 — 82 bước, gồm 4 workshop, 3 lab, 2 cụm lý thuyết, Review và Quiz.
> Bài tiếp theo: "Install Python" (mục tiếp theo sau Python Basics, 0/5 bước).
> Quy tắc làm bài: gõ code từng ký tự, không dán đáp án, không tự điền; theory/quiz bấm "Submit and go to next challenge", lab bấm "Check Your Code" rồi mới Submit.

## 1. Workshop: Build a Bill Splitter (Step 1–8)

- Chương trình chia tiền hoá đơn: nhận tổng tiền, số người, tiền tip… rồi tính mỗi người phải trả bao nhiêu.
- Thực hành: biến + số học + định dạng số trong f-string (in tiền với 2 chữ số thập phân) — nối tiếp phần 1.

## 2. Booleans and Conditionals (lý thuyết, 3 bài)

- Kiểu `bool`: chỉ có hai giá trị `True` và `False`.
- Toán tử so sánh: `==` `!=` `<` `>` `<=` `>=` — kết quả trả về `True`/`False`.
- Toán tử logic: `and` (và), `or` (hoặc), `not` (phủ định) — kết hợp nhiều điều kiện.
- Rẽ nhánh: `if` / `elif` / `else` — chương trình chọn một hướng chạy tuỳ điều kiện đúng hay sai.
- Đây là nền tảng của mọi bước kiểm tra dữ liệu trong các lab sau (tên rỗng? tuổi hợp lệ? số liệu trong khoảng?).

## 3. Workshop: Build a Movie Ticket Booking Calculator (Step 1–21)

- Chương trình tính tiền vé xem phim: giá vé theo độ tuổi/loại vé, số lượng vé, giảm giá…
- Thực hành: điều kiện `if/elif/else` nhiều nhánh + tính toán số học + in kết quả đúng định dạng.

## 4. Lab: Build a Travel Weather Planner (24/24 test)

- Chương trình gợi ý kế hoạch du lịch theo thời tiết: nhận thông tin thời tiết rồi đưa ra gợi ý phù hợp (mang ô, mặc ấm…).
- Kết quả: **24/24 test đạt** — toàn bộ trường hợp kiểm tra đều đúng.

## 5. Understanding Functions and Scope (lý thuyết, 2 bài)

- Hàm (`def`): gói một đoạn code thành một khối có tên, gọi lại được nhiều lần thay vì viết lặp.
- Tham số và giá trị trả về (`return`): hàm nhận đầu vào, xử lý, trả kết quả về cho chỗ gọi.
- Phạm vi biến (scope): biến trong hàm là biến cục bộ (chỉ dùng được trong hàm), biến ngoài hàm là biến toàn cục — tránh đặt trùng tên gây nhầm.
- Đây là bước ngoặt: từ viết code chạy thẳng một lượt sang viết hàm tái sử dụng — đúng cách Niên viết tool (mỗi lệnh là một hàm).

## 6. Workshop: Build a Kitchen Inventory Tracker (Step 1–17)

- Chương trình quản lý kho bếp: theo dõi nguyên liệu, số lượng còn lại, cảnh báo khi sắp hết…
- Thực hành: hàm + điều kiện + biến — gom các thao tác lặp lại thành hàm gọi nhiều lần.

## 7. Lab: Build an Apply Discount Function (11/11 test)

- Viết một hàm tính giá sau khi giảm giá: nhận giá gốc và tỉ lệ giảm, trả về giá cuối.
- Kết quả: **11/11 test đạt**.

## 8. Workshop: Build a Caesar Cipher (Step 1–26)

- Mã hoá Caesar: dịch mỗi chữ cái trong bảng chữ cái đi một số bước (shift) — mật mã cổ điển, dùng để luyện thao tác chuỗi và hàm.
- Hàm `caesar()` có kiểm tra shift, hỗ trợ cả chữ hoa; có hai hàm `encrypt()` (mã hoá) và `decrypt()` (giải mã).
- Giải mã thành công chuỗi `'Pbhentr vf sbhaq va hayvxryl cynprf.'` thành `'Courage is found in unlikely places.'` (Dũng cảm được tìm thấy ở những nơi không ngờ tới).
- Thực hành: vòng lặp qua từng ký tự, điều kiện xử lý chữ hoa/thường, dùng hàm để tách mã hoá và giải mã thành hai việc riêng.

## 9. Lab: Build an RPG Character (19/19 test)

- Viết hàm `create_character()` tạo nhân vật game nhập vai: kiểm tra tên và các chỉ số đầu vào có hợp lệ không, rồi trả về bảng thông tin nhân vật hiển thị bằng ký tự `●`/`○`.
- Kết quả: **19/19 test đạt**.
- Thực hành: kiểm tra dữ liệu đầu vào (validation) + hàm trả về chuỗi định dạng đẹp — giống mẫu validator ở phần 1, nhưng gọn hơn nhờ dùng hàm.

## 10. Python Basics Review + Python Basics Quiz

- Ôn tập toàn bộ Python Basics: biến, kiểu dữ liệu, chuỗi, số học, boolean, điều kiện, hàm, phạm vi biến.
- Quiz cuối phần: **20/20 câu đúng** — hoàn thành Python Basics 144/144.

## Áp dụng cho công việc của Niên

- Hàm + scope: viết tool CLI (Niên Tool) và script xử lý video theo kiểu mỗi việc một hàm, gọi lại được — thay vì viết code dài một mạch.
- Validation (kiểm tra đầu vào): mẫu `create_character()` và các lab kiểm tra — dùng khi kiểm tra file video, tên file, số liệu trước khi chạy bước tiếp theo trong pipeline.
- Caesar Cipher: thao tác chuỗi ký tự — nền tảng để xử lý phụ đề, tên file, chuỗi văn bản trong các script tự động hoá.
- Bill Splitter / Movie Ticket / Apply Discount: tính toán chia tiền, giảm giá — mẫu cho các hàm tính toán trong app tính toán điện (chia tải, tính giá, chiết khấu).

## Tiến độ tổng

| Mục | Xong/Tổng |
|---|---|
| Python Basics | 144/144 ✅ |
| Toàn khóa Python v9 | 144/588 |

Mục tiếp theo: **Install Python (0/5 bước)** — cài đặt Python trên máy, rồi sang các phần sau của khóa Python v9.
