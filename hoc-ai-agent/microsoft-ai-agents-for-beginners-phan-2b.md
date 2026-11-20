# Bài đã học: Microsoft AI Agents for Beginners — Phần 2b (Bài 09–13)

- Nguồn: Microsoft, "AI Agents for Beginners" (MIT), repo https://github.com/microsoft/ai-agents-for-beginners — đã fork về https://github.com/Nien7979/ai-agents-for-beginners và clone về máy.
- Ngày học: 20/11/2026 (đêm). Học tiếp sau phần 2a (bài 06–08).
- Phần này tóm tắt bài 09–13. Còn bài 14–18 học tiếp.

## Bài 09 — Metacognition in AI Agents (tự suy nghĩ về cách mình nghĩ)

- **Metacognition = "thinking about thinking"**: agent tự đánh giá và điều chỉnh hành động dựa trên tự nhận thức + kinh nghiệm. Lợi ích: tự phản chiếu (self-reflection), thích nghi, tự sửa lỗi, quản lý tài nguyên (thời gian, token).
- Ví dụ "metacognition thật": agent nói rõ lý do chọn ("Tôi ưu tiên vé rẻ vì... có thể tôi bỏ sót vé bay thẳng, để tôi kiểm tra lại") và **sửa cả chiến lược**, không chỉ sửa kết quả cuối.
- Thành phần agent: **persona** (tính cách), **tools** (khả năng), **skills** (kiến thức) — hợp thành một "expertise unit".
- **Planning**: xác định task hiện tại → chia bước → xác định tài nguyên → dùng kinh nghiệm quá khứ. Ví dụ travel agent: thu thập sở thích → tìm vé/khách sạn/điểm đến → gợi ý lịch trình → lấy feedback → điều chỉnh → xác nhận cuối → đặt → hỗ trợ tiếp.
- **Corrective RAG**: dùng RAG để sửa lỗi, gồm 3 phần — prompting technique (prompt dẫn việc retrieve), tool (thuật toán đánh giá độ liên quan), evaluation (đánh giá liên tục và điều chỉnh). Phân biệt với **pre-emptive context load**: nạp sẵn context (vd dict thông tin điểm đến) trước khi xử lý query — nhanh nhưng tốn context.
- **Bootstrap plan với goal trước khi lặp**: định mục tiêu rõ (vd tối đa hóa sự hài lòng trong ngân sách) rồi mới iterate cải tiến plan.
- **LLM re-ranking & scoring**: retrieve ứng viên → LLM chấm điểm + xếp lại theo độ liên quan/chất lượng.
- **RAG là prompting technique hay tool?** Technique = tự viết prompt cho từng query (kiểm soát cao, tốn công); Tool = hệ thống tự động retrieve + generate (nhanh, dễ tích hợp, hợp quy mô lớn).
- **Evaluating relevancy**: context awareness, accuracy, user intent, feedback loop. Kỹ thuật: relevance scoring (chấm điểm theo sở thích/ngân sách/địa điểm), filtering + ranking (top 10), NLP hiểu query, tích hợp feedback (liked +1, disliked −1).
- **Search with intent**: 3 loại intent — informational (tìm thông tin), navigational (tìm trang cụ thể), transactional (đặt/mua). Hiểu intent + context + cá nhân hóa mới ra kết quả đúng.
- **Code generation as a tool**: agent viết + chạy code để giải quyết việc (vd sinh SQL query theo schema của bảng, chạy rồi lấy kết quả — "SQL as RAG"). Cần hiểu schema (environment awareness) để sinh query đúng.
- **Ví dụ metacognition**: `HotelRecommendationAgent` chọn khách sạn rẻ nhất → nhận feedback "bad" (chất lượng < 7) → tự phản chiếu và **đổi chiến lược** sang "highest_quality" cho lần sau. Điểm mấu chốt: đánh giá quyết định cũ và điều chỉnh cách suy nghĩ, không chỉ đổi đáp án.

## Bài 10 — AI Agents in Production: Observability & Evaluation

- Agent từ prototype ra production cần **observability** (quan sát được) và **evaluation** (đánh giá) — biến "black box" thành "glass box".
- **Trace & Span** (Langfuse, Microsoft Foundry, OpenTelemetry): trace = toàn bộ một task từ đầu đến cuối; span = từng bước trong trace (gọi LLM, retrieve dữ liệu...). MAF tích hợp OpenTelemetry sẵn; có thể tạo span thủ công, gắn attribute (user_id, session_id, model_version).
- **Vì sao observability quan trọng**: debug + tìm root cause khi agent lỗi; quản lý latency & chi phí (theo dõi từng lần gọi LLM/API tính theo token); trust/safety/compliance (audit trail để phát hiện prompt injection, nội dung độc hại, rò rỉ PII); vòng cải tiến liên tục (dữ liệu production → cải tiến offline).
- **Metrics cần theo dõi**: latency, chi phí mỗi lần chạy, request errors (lỗi API/tool), user feedback tường minh (👍/👎, sao) và ngầm (hỏi lại, bấm retry), accuracy (định nghĩa "thành công" trước rồi đo), automated eval metrics (dùng LLM chấm output; thư viện RAGAS cho RAG, LLM Guard phát hiện ngôn ngữ độc hại/prompt injection).
- **Evaluation — offline vs online**:
  - Offline: test trên dataset có đáp án đúng (vd GSM8K), lặp lại được, đo accuracy rõ; đưa vào CI/CD để chống regression. Nhược điểm: dataset có thể lỗi thời so với thực tế.
  - Online: đánh giá trên traffic thật (tỉ lệ thành công, hài lòng, A/B test, shadow test) — bắt được model drift và query bất ngờ; khó lấy label tin cậy.
  - Vòng lặp chuẩn: evaluate offline → deploy → monitor online → thu thập case lỗi → bổ sung vào offline dataset → refine agent → lặp lại.
- **Vấn đề thường gặp**: agent làm không nhất quán (làm rõ prompt, chia subtask); vòng lặp vô hạn (đặt điều kiện dừng rõ, task suy luận phức tạp dùng model lớn hơn); tool call kém (test tool riêng ngoài agent, sửa tên/tham số/prompt của tool); multi-agent không ổn định (prompt từng agent riêng biệt, dùng routing/controller agent phân cấp).
- **Quản lý chi phí**: dùng model nhỏ (SLM) cho task đơn giản (phân loại intent, trích tham số), model lớn chỉ cho suy luận phức tạp; **router model** route request theo độ khó; **cache** các request giống nhau.

## Bài 11 — Agentic Protocols: MCP, A2A, NLWeb

### MCP (Model Context Protocol) — LLM ↔ tool/dữ liệu
- Chuẩn mở, kiến trúc client–server: **Host** (app LLM, vd VSCode) → **Client** (kết nối 1-1 với server) → **Server** (chương trình nhẹ, expose khả năng).
- 3 primitive của server: **Tools** (hành động gọi được, vd "search flights"), **Resources** (dữ liệu chỉ đọc: file, DB record, log), **Prompts** (template gợi ý cho workflow phức tạp).
- Lợi ích: dynamic tool discovery (agent tự hỏi server có tool gì, "integrate once" thay vì code cứng từng API), dùng được qua nhiều LLM khác nhau, chuẩn xác thực thống nhất.
- Ví dụ: trợ lý đặt vé — client hỏi MCP server của hãng bay có tool gì → gọi "search flights" → server gọi API nội bộ hãng → trả kết quả.

### A2A (Agent-to-Agent) — agent ↔ agent
- Cho các agent khác tổ chức/môi trường/tech stack cộng tác làm chung một task.
- Thành phần: **Agent Card** (tên, mô tả task, danh sách skill, endpoint URL, version, capabilities như streaming/push), **Agent Executor** (chuyển context chat cho agent xa, agent xa tự dùng LLM + tool của nó), **Artifact** (sản phẩm hoàn thành + mô tả + context, gửi xong đóng kết nối), **Event Queue** (xử lý cập nhật, giữ kết nối khi task lâu).
- Lợi ích: cộng tác xuyên vendor, mỗi agent tự chọn LLM riêng, xác thực tích hợp sẵn.
- Ví dụ: Travel Agent nhận "đặt trọn chuyến Honolulu" → qua A2A gọi Airline Agent, Hotel Agent, Car Rental Agent (mỗi bên một công ty) → tổng hợp trả lời.

### NLWeb (Natural Language Web) — website nói được ngôn ngữ tự nhiên
- Thành phần: NLWeb Application (engine xử lý câu hỏi), NLWeb Protocol (luật tương tác, trả JSON dùng Schema.org), MCP Server endpoint (mỗi NLWeb cũng là một MCP server, expose method `ask`), Embedding Models (chuyển nội dung web thành vector), Vector Database (Qdrant, Snowflake, Milvus, Azure AI Search, Elasticsearch...).
- Cách chạy: ingest catalog (Schema.org/RSS) → tạo embedding → lưu vector DB → user hỏi bằng ngôn ngữ tự nhiên → LLM + vector search → trả lời dựa trên dữ liệu thật của web (không bịa).
- Agent bên ngoài cũng query được qua MCP `ask(...)`.

## Bài 12 — Context Engineering for AI Agents

- **Context engineering** = đảm bảo agent có đúng thông tin để làm bước tiếp theo. Khác **prompt engineering** (một bộ hướng dẫn tĩnh) ở chỗ context engineering quản lý **tập thông tin động** theo thời gian, làm cho quy trình lặp lại được và tin cậy.
- **5 loại context**: Instructions (prompt, system message, few-shot, mô tả tool), Knowledge (fact, dữ liệu retrieve, memory dài hạn, RAG), Tools (định nghĩa function/API/MCP + kết quả trả về), Conversation History (hội thoại đang dài dần, chiếm context window), User Preferences (sở thích học được theo thời gian).
- **Planning**: (1) Define Clear Results — "thế giới sẽ ra sao khi agent xong việc?"; (2) Map the Context — agent cần thông tin gì, ở đâu; (3) Create Context Pipelines — lấy bằng cách nào (RAG, MCP server, tool).
- **Chiến lược thực hành**:
  1. **Agent Scratchpad**: ghi chú trong session, lưu ngoài context window (file/runtime object), cần thì lấy lại.
  2. **Memories**: lưu qua nhiều session (tóm tắt, preference, feedback cải tiến).
  3. **Compressing Context**: gần đầy window thì summarize/trim, giữ thông tin liên quan nhất.
  4. **Multi-Agent Systems**: mỗi agent có context window riêng — chia context cũng là context engineering.
  5. **Sandbox Environments**: chạy code/xử lý tài liệu lớn trong sandbox, chỉ đưa kết quả vào context.
  6. **Runtime State Objects**: container lưu kết quả từng subtask, context chỉ gắn với subtask đang làm.
- **Inspecting context** (debug): không cần log raw prompt — ghi record nhỏ: số chunk/tool/memory đã xét/đã chọn + luật lọc, token trước/sau compress, subtask nào chạy riêng + summary trả về, id/score/redaction của RAG/memory. Mục tiêu: đủ bằng chứng để biết chiến lược nào đã chạy và có đổi đúng model call tiếp theo không.
- **4 lỗi context thường gặp**:
  | Lỗi | Cách sửa |
  |---|---|
  | **Poisoning** — ảo giác/lỗi chui vào context rồi bị tham chiếu mãi, agent theo đuổi mục tiêu không thể | Validate trước khi đưa vào memory dài hạn; quarantine — phát hiện thì mở context thread mới |
  | **Distraction** — context quá lớn, model bám vào lịch sử cũ thay vì dùng kiến thức đã học | Summarize định kỳ, giữ phần liên quan, bỏ lịch sử thừa |
  | **Confusion** — quá nhiều tool khiến model gọi sai tool (model nhỏ đặc biệt dễ dính) | Tool loadout management: RAG trên mô tả tool, chỉ đưa tool liên quan nhất cho từng task; nghiên cứu cho thấy nên giữ dưới 30 tool |
  | **Clash** — thông tin mâu thuẫn trong context (vd lúc đầu economy, sau đổi business) | Context pruning (xóa/ghi đè chỉ dẫn cũ) + offloading (scratchpad riêng để đối chiếu trước khi quyết) |

## Bài 13 — Memory for AI Agents

- Không có memory, agent là stateless — mỗi lần tương tác bắt đầu lại từ đầu, trải nghiệm lặp lại và khó chịu. Memory giúp agent reflective, interactive, proactive/reactive, autonomous — đáng tin và có khả năng hơn.
- **Các loại memory**:
  - **Working memory**: giấy nháp trong một task — thông tin cần ngay cho bước tiếp theo (yêu cầu, đề xuất, quyết định, hành động).
  - **Short-term memory**: trong một session/conversation (MAF: `AgentSession` qua `agent.create_session()` — hết session hoặc restart app là mất).
  - **Long-term memory**: tồn tại qua nhiều session — sở thích user, lịch sử tương tác (vd "Ben thích trượt tuyết, thích cà phê view núi, tránh dốc khó vì chấn thương cũ").
  - **Persona memory**: giữ tính cách/vai trò nhất quán của agent (vd "chuyên gia lập kế hoạch trượt tuyết").
  - **Workflow/Episodic memory**: chuỗi bước của task phức tạp, gồm cả thành công và thất bại (vd lần đặt vé thất bại vì hết chỗ → lần sau thử vé khác).
  - **Entity memory**: trích và nhớ thực thể (người, địa điểm, sự vật, sự kiện) — vd nhớ "Le Chat Noir" để lần sau gợi ý đặt bàn lại.
  - **Structured RAG**: trích thông tin có cấu trúc (vd chi tiết chuyến bay từ email: điểm đến, ngày, giờ, hãng) thay vì chỉ match ngữ nghĩa — cho precision/recall cao, truy vấn chính xác kiểu "Tôi đã đặt chuyến nào đi Paris thứ Ba?".
- **Quản lý memory**: generate → store → retrieve → integrate → update → quên/xóa. Retrieve là khâu quan trọng nhất.
- **Công cụ chuyên biệt**:
  - **Mem0**: memory layer bền vững; pipeline 2 pha — extraction (LLM tóm tắt lịch sử chat, trích memory mới) rồi update (LLM quyết định thêm/sửa/xóa), lưu trong hybrid store (vector + graph + key-value); hỗ trợ graph memory cho quan hệ giữa các entity.
  - **Cognee**: semantic memory mã nguồn mở — biến dữ liệu có/không cấu trúc thành knowledge graph + embeddings; kiến trúc dual-store (vector similarity + graph relationships), hybrid retrieval (vector + graph + LLM reasoning); "living memory" lớn dần mà vẫn query được như một đồ thị thống nhất. Notebook demo: ingest nhiều nguồn, visualize graph, query theo nhiều chiến lược.
  - **Azure AI Search**: backend lưu/truy xuất memory, đặc biệt Structured RAG.
- **Self-improving agents — pattern "knowledge agent"**: một agent riêng quan sát cuộc hội thoại chính → xác định thông tin đáng lưu → trích + tóm tắt → lưu vào knowledge base (thường vector DB) → query sau sẽ retrieve thông tin đó và append vào prompt (như RAG).
- **Tối ưu**: dùng model rẻ/nhanh để sàng lọc xem thông tin có đáng lưu/lấy không, chỉ gọi model phức tạp khi cần (giảm latency); thông tin ít dùng chuyển sang "cold storage" để tiết kiệm chi phí.

## Áp dụng cho công việc của Niên và Hoai

- **MEMORY.md + daily notes của Niên chính là long-term memory**: sở thích, quyết định, lịch sử làm việc của Hoai được lưu bền vững qua nhiều session — đúng pattern bài 13. Daily log = episodic memory (ghi cả thành công lẫn thất bại để lần sau tránh lặp lại).
- **Compaction/context window**: mỗi lần hội thoại bị compact chính là bài 12 — phải summarize và giữ thông tin quan trọng nhất; ghi chú ngắn gọn, có cấu trúc giúp lần sau retrieve nhanh (đúng kiểu Structured RAG).
- **Observability (bài 10) áp thẳng vào hệ cron/watchdog**: log từng lần chạy, ghi rõ trace (cron nào, bước nào, kết quả), đo chi phí (token fosta, credit Flow) — và vòng lặp offline→online: case lỗi trong log (vd browser báo sai subtitle) được bổ sung thành bài học trong AGENTS.md để lần sau không lặp lại.
- **Metacognition (bài 09)**: tự phản chiếu sau mỗi việc — vd vụ báo Facebook "chưa kết nối" 09:48 rồi tự sửa 10:31 là self-correction đúng nghĩa: không chỉ sửa đáp án mà sửa cả cách kiểm tra (đợi merge lan truyền xong mới kết luận).
- **MCP/A2A (bài 11)**: khi các phòng/agent cần giao tiếp với nhau (Niên ↔ phòng, phòng ↔ phòng), dùng chuẩn chung thay vì mỗi bên một cách — đúng hướng của hệ side chat + worker hiện tại.
- **Context failures (bài 12)**: lỗi "poisoning" = một thông tin sai lọt vào MEMORY.md rồi bị dùng mãi (đã từng xảy ra: memory job làm mất fingerprint host key) → cần validate trước khi ghi memory dài hạn; lỗi "clash" = chỉ dẫn cũ/mới mâu thuẫn trong context → prune, chỉ giữ quyết định mới nhất của Hoai.
