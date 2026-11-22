# freeCodeCamp Python v9 — Phần 7: Linear Data Structures + Build a Hash Table

- Tài khoản: CHÍNH (lehuunien95@gmail.com, fcc-07869125) — đã xác minh qua trang Profile đầu phiên, không dùng tài khoản phụ.
- Số đếm đọc trực tiếp từ trang khóa học: đầu phiên 439/588, cuối phiên **471/588**.
- Linear Data Structures: **31/32** (1 bước còn lại là Data Structures Quiz — bài thi 20 câu, cần ≥18 đúng để qua; BỎ QUA theo quy tắc: bước cần Hoai tự làm thì bỏ qua, không hỏi, không chờ).
- Phiên 7 chạy 22/11 ~18:52–20:04 (trình duyệt, gõ từng ký tự, không điền sẵn).

## 1. Linear Data Structures — lý thuyết (Working with Common Data Structures, 6/6)

- **Algorithm & Big O:** độ phức tạp thời gian/không gian; O(1) truy cập theo chỉ số, O(n) duyệt tuyến tính, O(log n) tìm kiếm nhị phân, O(n²) vòng lồng nhau. Chọn cấu trúc dữ liệu theo thao tác hay dùng nhất.
- **Problem-Solving:** đọc đề → ví dụ nhỏ → chia bài toán → chọn cấu trúc → code → kiểm thử biên.
- **Dynamic Arrays:** mảng động (list trong Python) — truy cập O(1), thêm cuối O(1) trung bình (amortized), chèn/xóa giữa O(n) vì phải dời phần tử.
- **Stacks & Queues:** Stack LIFO (push/pop ở một đầu), Queue FIFO (enqueue một đầu, dequeue đầu kia). Dùng list hoặc collections.deque.
- **Singly Linked Lists:** mỗi node giữ data + con trỏ next; chèn/xóa đầu O(1), truy cập theo vị trí O(n), không cần dời phần tử như mảng.
- **Maps / Hash Maps / Sets:** dict/set trong Python — tra cứu trung bình O(1) nhờ hàm băm; set chỉ giữ key duy nhất, không có thứ tự.

## 2. Workshop: Build a Linked List (24/24)

- Class `LinkedList` với class con `Node` bên trong (mỗi node: `data`, `next`).
- `__init__`: head = None.
- `is_empty()`: kiểm tra head có phải None không.
- `add(data)`: tạo Node mới, chèn vào đầu danh sách (new_node.next = head, head = new_node).
- `remove(data)`: duyệt tìm node, nối lại con trỏ next của node trước để bỏ qua node cần xóa; không lỗi khi danh sách rỗng hoặc không tìm thấy.
- Kết quả kiểm thử trên terminal: `True, False, 2, 1`.

## 3. Data Structures Review (100%)

- Tick xác nhận đã ôn lại toàn bộ cấu trúc dữ liệu trong phần rồi nộp.

## 4. Lab: Build a Hash Table (22/22 tests Passed — "Completed Certification Project")

- Class `HashTable`, `__init__` với `self.collection = {}`.
- `hash(key)`: tổng `ord()` từng ký tự — ví dụ `hash('golf') = 424`.
- `add(key, value)`: tính hash, lưu vào `collection[hash]`; **va chạm hash lưu dict lồng nhau** — ví dụ `'dear'`/`'read'` cùng 412, `'fcc'`/`'cfc'` cùng 300.
- `remove(key)`: chỉ xóa đúng key trong dict lồng nhau; không lỗi khi key không tồn tại.
- `lookup(key)`: trả về giá trị hoặc `None` — ví dụ `lookup('cfc') = None` khi chỉ có `'fcc'`.

## Tiếp theo (phiên 8)

- Recursion → Algorithms → Tower of Hanoi (project) → Graphs and Trees → Dynamic Programming → Python Review → bài thi chứng chỉ (thi thì BỎ QUA theo quy tắc).
