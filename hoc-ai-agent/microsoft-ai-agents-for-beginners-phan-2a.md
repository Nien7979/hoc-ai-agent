# Bài đã học: Microsoft AI Agents for Beginners — Phần 2a (Bài 06–08)

- Nguồn: Microsoft, "AI Agents for Beginners" (MIT), repo https://github.com/microsoft/ai-agents-for-beginners — đã fork về https://github.com/Nien7979/ai-agents-for-beginners và clone về máy.
- Ngày học: 20/11/2026 (đêm). Phần 2 của khóa, học tiếp sau phần 1 (bài 01–05).
- Phần này tóm tắt bài 06–08. Còn bài 09–18 học tiếp.

## Bài 06 — Building Trustworthy AI Agents (agent đáng tin)

### System Message Framework — 4 bước viết system prompt có thể mở rộng
1. **Meta system message**: viết một prompt mẫu để LLM tự sinh system prompt cho các agent ("Bạn là chuyên gia tạo trợ lý AI... hãy mô tả chi tiết vai trò, nhiệm vụ...").
2. **Basic prompt**: mô tả ngắn agent cần làm gì (vai trò + nhiệm vụ + trách nhiệm).
3. **Đưa basic prompt qua meta prompt**: LLM trả về system prompt đầy đủ, có cấu trúc (tên công ty, vai trò, mục tiêu, trách nhiệm từng việc, tone, hướng dẫn tương tác, ghi chú).
4. **Iterate**: system prompt hiếm khi đúng ngay lần đầu — chỉnh basic prompt, chạy lại, so sánh và đánh giá kết quả.

### 5 loại đe dọa và cách khóa
| Đe dọa | Cách khóa |
|---|---|
| **Task/Instruction manipulation** — kẻ tấn công đổi lệnh/mục tiêu của agent qua prompt | Validate + lọc input trước khi xử lý; giới hạn số lượt hội thoại |
| **Access to critical systems** — agent chạm hệ thống chứa dữ liệu nhạy cảm | Chỉ cấp quyền theo nhu cầu (need-only), xác thực + kiểm soát truy cập, giao tiếp mã hóa |
| **Resource/Service overloading** — lợi dụng agent gửi ồ ạt request gây sập hệ thống hoặc tốn tiền | Giới hạn số request agent được gọi tới mỗi service, giới hạn lượt hội thoại |
| **Knowledge base poisoning** — đầu độc dữ liệu/kho tri thức agent dùng | Kiểm tra dữ liệu định kỳ; chỉ người tin cậy mới được sửa dữ liệu |
| **Cascading errors** — lỗi ở một tool lan sang các hệ thống khác | Chạy agent trong môi trường giới hạn (vd Docker container); fallback + retry logic khi service lỗi |

### Human-in-the-loop
- Người dùng là một "agent" trong hệ thống: duyệt hoặc dừng tiến trình đang chạy.
- Code mẫu dùng MAF: agent xin phê duyệt trước khi chốt đáp án; kèm **risk tiering** (phân tầng rủi ro) và **audit logging** (ghi log để kiểm tra sau).
- Kết luận bài: agent đáng tin = thiết kế cẩn thận + biện pháp bảo mật mạnh + lặp lại liên tục; luôn chủ động về bảo mật, riêng tư, đạo đức.

## Bài 07 — Planning Design Pattern (lập kế hoạch)

- Việc thật thường quá phức tạp cho 1 bước → agent cần **mục tiêu tổng rõ ràng**, rồi **chia thành subtask** (vd làm lịch trình du lịch 3 ngày → đặt vé bay, đặt khách sạn, thuê xe, cá nhân hóa), mỗi subtask giao cho agent chuyên biệt, agent điều phối tổng hợp lại.
- **Structured output**: bắt LLM trả JSON theo schema (vd Pydantic: `TravelPlan { main_task, subtasks: [{task_details, assigned_agent}], is_greeting }`) để máy đọc được và route đúng agent — đặc biệt quan trọng trong multi-agent.
- **Planning agent + multi-agent orchestration**: planner (semantic router) nhận yêu cầu → sinh kế hoạch có cấu trúc → route từng subtask tới agent tương ứng (hoặc qua group chat manager nếu nhiều agent phối hợp) → tổng hợp kết quả cho người dùng.
- **Iterative planning**: kết quả subtask trước có thể đổi kế hoạch sau (vd phát hiện định dạng dữ liệu lạ khi đặt vé → đổi chiến lược trước khi đặt khách sạn); phản hồi của người dùng cũng trigger re-plan một phần.
- Tham khảo thêm: **Magentic-One** (Microsoft Research) — orchestrator tạo plan theo từng task, giao việc, theo dõi tiến độ và re-plan khi cần; đạt kết quả cao trên nhiều benchmark agent.

## Bài 08 — Multi-Agent Design Pattern (nhiều agent phối hợp)

### Khi nào dùng multi-agent
- **Khối lượng lớn**: chia nhỏ để xử lý song song, xong nhanh hơn.
- **Nhiệm vụ phức tạp**: mỗi agent chuyên một khía cạnh (vd xe tự lái: agent lái, agent phát hiện chướng ngại, agent giao tiếp với xe khác).
- **Chuyên môn đa dạng**: mỗi agent giỏi một mảng (vd y tế: chẩn đoán, phác đồ điều trị, theo dõi bệnh nhân).

### Lợi thế so với 1 agent
- **Specialization**: agent chuyên thì không bị rối khi việc phức tạp; 1 agent ôm đồm dễ làm việc không đúng sở trường.
- **Scalability**: thêm agent dễ hơn là nhồi thêm việc vào 1 agent.
- **Fault tolerance**: 1 agent lỗi, các agent khác vẫn chạy.
- Ví dụ: đặt chuyến đi — 1 agent phải ôm cả vé bay + khách sạn + xe (đơn khối, khó bảo trì); multi-agent thì agent vé, agent khách sạn, agent xe riêng biệt, module hóa. So sánh: cửa hàng gia đình (1 người làm hết) vs chuỗi franchise (mỗi người một mảng).

### 6 building blocks
1. **Agent Communication**: quyết định agent nào chia sẻ thông tin gì, bằng cách nào (vd agent vé phải báo ngày đi cho agent khách sạn).
2. **Coordination Mechanisms**: phối hợp hành động để thỏa preference + constraint của user (vd khách sạn gần sân bay, xe chỉ thuê được ở sân bay).
3. **Agent Architecture**: cấu trúc nội bộ để agent ra quyết định và học từ tương tác (vd model ML gợi ý vé theo sở thích cũ).
4. **Visibility**: phải thấy được các agent tương tác ra sao — log/monitoring, visualization, performance metrics.
5. **Multi-Agent Patterns**: chọn kiến trúc tập trung / phi tập trung / lai.
6. **Human in the loop**: dạy agent khi nào cần hỏi người (vd xác nhận trước khi đặt vé).

### Visibility — 3 công cụ
- **Logging & Monitoring**: log từng hành động (agent nào, làm gì, lúc nào, kết quả) để debug và tối ưu.
- **Visualization**: đồ thị luồng thông tin giữa các agent → thấy bottleneck.
- **Performance Metrics**: thời gian hoàn thành, số task/giờ, độ chính xác gợi ý.

### 3 pattern multi-agent cụ thể
- **Group chat**: mỗi agent là một "người" trong nhóm chat, trao đổi tin nhắn với nhau (hợp tác nhóm, CSKH, mạng xã hội). Có thể tập trung (qua server trung tâm) hoặc phi tập trung.
- **Hand-off**: agent bàn giao task cho nhau theo luật định sẵn (CSKH, quản lý task, tự động workflow).
- **Collaborative filtering**: nhiều agent chuyên môn khác nhau cùng gợi ý (vd chọn cổ phiếu: agent chuyên ngành + agent phân tích kỹ thuật + agent phân tích cơ bản).

### Ví dụ: quy trình hoàn tiền (refund)
- Agent chuyên cho refund: customer, seller, payment, resolution, compliance.
- Agent dùng chung cho cả công ty: shipping, feedback, escalation, notification, analytics, audit, reporting, knowledge, security, quality.
- Bài tập: thiết kế hệ multi-agent cho CSKH — liệt kê agent, vai trò, cách tương tác (đáp án trong `solution/solution.md`).
- Quiz: Q1 — đáp án A2 (refund cần fraud/payment/compliance phối hợp); Q2 — đáp án A1 (1 bộ hướng dẫn + tool là đủ thì dùng 1 agent).

## Áp dụng cho công việc của Niên và Hoai

- **Công ty thu nhỏ của Hoai vốn đã là một hệ multi-agent**: Niên = planner/orchestrator (nhận việc, chia task, tổng hợp), các phòng = specialized agents (Thảo kịch bản, Hùng sản xuất, Phong xu hướng...), Hoai = human-in-the-loop duyệt quyết định lớn. Bài 07–08 cho khung lý thuyết để vận hành nó chặt chẽ hơn: kế hoạch có cấu trúc, route rõ ràng, log đầy đủ.
- **Planning cho sản xuất video**: planner chia việc 1 video thành subtask (chọn trend → viết kịch bản → thu voice → dựng → kiểm tra → đăng), mỗi subtask giao đúng phòng; kết quả bước trước đổi được kế hoạch bước sau (vd voice lỗi → làm lại voice trước khi dựng).
- **Trust (bài 06) áp thẳng vào quy tắc nhà mình**: agent/cron chỉ được quyền tối thiểu (need-only), mọi thao tác ghi/xóa/đăng đều qua approval gate của Hoai hoặc Niên; ghi audit log cho mọi hành động tự động (đúng như log cron/watchdog đã làm); giới hạn số lượt gọi API để tránh tốn tiền (vd fosta tính theo balance, Flow tính theo credit).
- **System message framework**: khi tạo agent mới (vd agent kiểm tra video, agent đăng bài), viết system prompt theo 4 bước thay vì viết tay một lần — dễ mở rộng và cải tiến dần.
