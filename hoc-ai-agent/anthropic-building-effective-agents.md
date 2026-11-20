# Bài đã học: Building Effective Agents — Anthropic

- Nguồn: Anthropic Engineering Blog, "Building Effective Agents" (đăng 19/12/2024, cập nhật 09/10/2026). Link: https://www.anthropic.com/engineering/building-effective-agents
- Ngày học: 20/11/2026. Đây là bài số 2 trong top 5 lộ trình AI Agent của Hoai.

## 1. Workflow khác Agent như thế nào

- **Workflow**: LLM + tool được điều phối theo đường đi cố định viết sẵn trong code.
- **Agent**: LLM tự quyết định quy trình và cách dùng tool, giữ quyền kiểm soát cách hoàn thành việc.
- Cả hai đều gọi chung là "agentic systems".

## 2. Nguyên tắc chọn độ phức tạp

- Tìm giải pháp ĐƠN GIẢN NHẤT trước, chỉ tăng phức tạp khi thật sự cần. Nhiều việc chỉ cần 1 lần gọi LLM + retrieval là đủ.
- Việc xác định rõ ràng, lặp lại → dùng workflow (dễ đoán, ổn định).
- Việc cần linh hoạt, quyết định theo dữ liệu ở quy mô lớn → dùng agent.
- Agent đánh đổi độ trễ + chi phí để lấy hiệu quả; lỗi có thể cộng dồn qua nhiều bước.

## 3. Về framework

- Nên bắt đầu bằng cách gọi trực tiếp LLM API — nhiều pattern chỉ cần vài dòng code.
- Framework che mất prompt/response thật nên khó debug, và dễ dụ mình thêm phức tạp không cần thiết.
- Nếu dùng framework thì phải hiểu code bên dưới nó làm gì.

## 4. Khối xây dựng cơ bản: Augmented LLM

- LLM + retrieval + tools + memory. Model tự sinh truy vấn tìm kiếm, tự chọn tool, tự quyết giữ lại thông tin gì.
- MCP (Model Context Protocol) là cách chuẩn để nối agent với hệ tool bên ngoài.

## 5. Năm mẫu workflow

1. **Prompt chaining**: chia việc thành chuỗi bước nối tiếp, mỗi lần gọi LLM xử lý đầu ra của bước trước; có thể gắn bước kiểm tra giữa chừng. Đổi độ trễ lấy độ chính xác. Ví dụ: viết bản nháp → kiểm tra dàn ý → viết bài hoàn chỉnh.
2. **Routing**: phân loại đầu vào rồi đưa tới nhánh chuyên biệt. Ví dụ: câu hỏi thường → model nhỏ rẻ; câu khó → model mạnh.
3. **Parallelization**: chạy song song rồi gộp kết quả. Hai dạng: sectioning (chia nhỏ việc độc lập) và voting (chạy nhiều lần lấy nhiều góc nhìn, vd review code tìm lỗi bảo mật).
4. **Orchestrator-workers**: một LLM trung tâm tự chia việc, giao cho các worker LLM, rồi tổng hợp. Dùng khi không đoán trước được các bước con (vd sửa code nhiều file).
5. **Evaluator-optimizer**: một LLM sinh đáp án, một LLM khác đánh giá và góp ý, lặp lại. Dùng khi có tiêu chí đánh giá rõ ràng và việc lặp lại cải thiện được kết quả (vd dịch văn học).

## 6. Agent

- Agent = LLM dùng tool theo vòng lặp, dựa vào phản hồi từ môi trường (kết quả tool, kết quả chạy code) để đánh giá tiến độ.
- Cần "ground truth" ở mỗi bước; dừng khi xong việc hoặc chạm giới hạn (vd số vòng lặp tối đa); có thể dừng xin ý kiến người ở checkpoint.
- Dùng cho bài toán mở, không đoán trước được số bước. Phải test kỹ trong môi trường sandbox + gắn guardrails vì chi phí cao và lỗi cộng dồn.

## 7. Ba nguyên tắc cốt lõi khi làm agent

1. **Đơn giản** trong thiết kế.
2. **Minh bạch**: cho thấy rõ các bước lập kế hoạch của agent.
3. **Chăm chút agent-computer interface (ACI)**: tài liệu tool + kiểm thử kỹ — đầu tư cho ACI như đầu tư cho giao diện người–máy (HCI).

## 8. Hai ứng dụng thực tế tốt nhất (Appendix 1)

- **Hỗ trợ khách hàng**: hội thoại + tra cứu dữ liệu + hành động (hoàn tiền, cập nhật ticket), đo được kết quả rõ ràng.
- **Coding agent**: code kiểm chứng được bằng test tự động; agent lặp lại dựa trên kết quả test; Anthropic dùng agent giải các issue GitHub thật trong benchmark SWE-bench Verified. Con người vẫn cần review cuối.

## 9. Prompt engineering cho tool (Appendix 2)

- Định nghĩa tool xứng đáng được chăm chút như prompt chính.
- Cho model đủ token để "suy nghĩ" trước khi hành động.
- Format gần với dạng dữ liệu tự nhiên trên internet; tránh overhead như bắt model đếm dòng khi viết diff, hay escape ký tự trong JSON.
- Đặt mình vào vị trí của model: mô tả tool có rõ ràng không, có ví dụ dùng, trường hợp biên, ranh giới với tool khác không?
- Poka-yoke (thiết kế để khó sai): ví dụ Anthropic đổi tool đường dẫn thành bắt buộc dùng đường dẫn tuyệt đối vì model hay sai với đường dẫn tương đối — sau đó model dùng đúng hoàn toàn.
- Anthropic cho biết khi làm agent SWE-bench, họ tốn nhiều thời gian tối ưu tool hơn là tối ưu prompt tổng.

## 10. Áp dụng cho công việc của Niên

- Khi viết tool/agent (Niên Tool, shop bot, cron kiểm tra...): ưu tiên workflow đơn giản (prompt chaining, routing) trước; chỉ làm agent tự trị khi bài toán thật sự mở và không đoán được số bước.
- Viết mô tả cho mỗi tool như viết docstring cho một lập trình viên mới vào nghề: rõ ràng, có ví dụ, nêu trường hợp biên.
- Test tool với nhiều input mẫu trước khi đưa vào chạy tự động; với agent thì chạy thử trong môi trường an toàn trước.
- Đo hiệu quả (evals) rồi mới thêm phức tạp — không thêm bước chỉ vì "cho giống agent".
