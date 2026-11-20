# Bài đã học: Microsoft AI Agents for Beginners — Phần 1 (Bài 01–05)

- Nguồn: Microsoft, "AI Agents for Beginners" (MIT), repo https://github.com/microsoft/ai-agents-for-beginners — đã fork về https://github.com/Nien7979/ai-agents-for-beginners và clone về máy (`~/workspace/knowledge/ai-agents-for-beginners/`).
- Ngày học: 20/11/2026. Đây là bài số 3 trong top 5 lộ trình AI Agent của Hoai.
- Cấu trúc khóa: 18 bài (00–18), mỗi bài có README + video + code Python dùng **Microsoft Agent Framework (MAF)** + **Microsoft Foundry Agent Service V2**. Có bản dịch tiếng Việt (thư mục `translations/vi`, dịch máy qua Co-op Translator).
- **Lưu ý quan trọng khi thực hành:** code mẫu mặc định cần Azure subscription + `az login` (Foundry). Với Hoai thì KHÔNG đăng ký Azure trả phí — thực hành bằng client tương thích OpenAI với các API miễn phí đã có (xKiro, Xingchi relay, Nara...). Một số sample trong khóa đã hỗ trợ provider tương thích OpenAI (vd MiniMax). Học khái niệm từ README trước, chạy code bằng provider miễn phí sau.
- Phần này tóm tắt bài 01–05. Các bài 06–18 học tiếp ở phần 2.

## Bài 01 — Intro to AI Agents: Agent là gì

- **Định nghĩa:** AI Agent là hệ thống cho phép LLM **thực sự làm việc** — bằng cách cấp cho nó tools và kiến thức để tác động lên thế giới, không chỉ trả lời prompt.
- Ba thành phần cốt lõi của mọi agent: **Environment** (môi trường làm việc, vd nền tảng đặt vé), **Sensors** (cách đọc trạng thái môi trường, vd kiểm tra giá vé), **Actuators** (cách hành động, vd đặt phòng, gửi xác nhận).
- Thêm **Memory + Knowledge**: nhớ ngắn hạn (hội thoại hiện tại) và dài hạn (database khách hàng, vd thích ghế cửa sổ).

### 7 loại agent (ví dụ agent đặt vé du lịch)
| Loại | Làm gì |
|---|---|
| Simple Reflex | Theo luật cứng, không nhớ, không lập kế hoạch (thấy mail khiếu nại → chuyển cho CSKH) |
| Model-Based Reflex | Giữ mô hình nội bộ về thế giới, cập nhật theo thay đổi (theo dõi giá vé, cảnh báo tuyến đột nhiên đắt) |
| Goal-Based | Có mục tiêu, tự tìm cách đạt được từng bước (đặt trọn chuyến đi: bay + xe + khách sạn) |
| Utility-Based | Không chỉ tìm *một* cách giải mà tìm cách *tốt nhất* (cân chi phí vs tiện lợi) |
| Learning | Học từ phản hồi để giỏi dần (điều chỉnh gợi ý theo khảo sát sau chuyến đi) |
| Hierarchical | Agent cấp cao chia việc, giao cho agent cấp dưới (hủy chuyến → hủy bay + hủy khách sạn + hủy xe) |
| Multi-Agent Systems | Nhiều agent độc lập hợp tác hoặc cạnh tranh (agent khách sạn, agent bay, agent giải trí...) |

### Khi nào nên dùng agent — và khi nào không
- Nên dùng: **vấn đề mở** (không lập trình trước được các bước), **quy trình nhiều bước** dùng tool qua nhiều lượt, **cải thiện theo thời gian** nhờ phản hồi.
- Không phải việc gì cũng cần agent — nhiều việc chỉ cần 1 lần gọi LLM là đủ (nối với bài Anthropic: đơn giản trước).

### Ba khái niệm nền
- **Agentic Patterns**: chiến lược prompt + điều phối tái sử dụng được (vì không thể viết tay từng prompt cho agent chạy nhiều bước).
- **Agentic Frameworks**: template, tool, hạ tầng dựng sẵn — nối tool, quan sát/debug agent, phối hợp nhiều agent.

## Bài 02 — Frameworks: MAF vs Foundry Agent Service

### Framework giúp gì
- **Modular Components**: thành phần dựng sẵn (AI connector, memory connector, function calling, prompt template) → ráp nhanh prototype, không làm lại từ đầu.
- **Collaborative Tools**: thiết kế agent theo vai trò chuyên biệt, test luồng phối hợp.
- **Learn in Real-Time**: vòng phản hồi — agent học từ tương tác và điều chỉnh hành vi.

### Microsoft Agent Framework (MAF) — SDK để build
- API Python gọn qua `FoundryChatClient`: tạo agent bằng tên + instructions + tools.
- Tool định nghĩa bằng decorator `@tool` trên hàm Python thường (vd `book_flight(date, location)`); framework tự sinh schema, tự xử lý gọi đi gọi lại với model.
- Xác thực không cần key: `AzureCliCredential` / `DefaultAzureCredential`.
- Multi-agent: tạo nhiều agent chuyên biệt (planner → executor...), chạy nối tiếp nhau.
- **Dùng khi:** build agent production-ready, tool use, workflow nhiều bước, muốn bắt đầu nhanh.

### Microsoft Foundry Agent Service — nền tảng triển khai
- Agent chạy như một "smart microservice" trên Foundry: trả lời (RAG), hành động, tự động workflow.
- Khái niệm **Thread**: lưu toàn bộ hội thoại (tin nhắn, ngữ cảnh, trạng thái) phía server.
- Built-in tools: **Knowledge tools** (Grounding với Bing Search, File Search, Azure AI Search) và **Action tools** (Function Calling, Code Interpreter, OpenAPI tools, Azure Functions) — gom thành **toolset**.
- Hỗ trợ nhiều model (OpenAI, Llama 3, Mistral, Cohere...), bảo mật enterprise.
- **Dùng khi:** cần triển khai enterprise, tích hợp sẵn Search/code execution, mở rộng quy mô.
- **Quy tắc chọn:** bắt đầu bằng MAF để build và lặp nhanh logic agent; khi cần deploy/scale thì đưa lên Agent Service.

## Bài 03 — Agentic Design Principles (thiết kế lấy con người làm trung tâm)

Không phải kiến trúc bắt buộc — là điểm khởi đầu để thiết kế trải nghiệm agent tốt. Agent nên: mở rộng năng lực con người, lấp khoảng trống kiến thức, hỗ trợ cộng tác theo cách mỗi người thích, giúp ta trở thành phiên bản tốt hơn của chính mình.

### Agent (Space) — không gian
- **Connecting, not collapsing**: kết nối người với người, sự kiện, kiến thức hành động được — không thay thế hay hạ thấp con người.
- **Dễ tiếp cận nhưng đôi khi vô hình**: agent chạy nền, chỉ "nudge" (nhắc nhẹ) khi phù hợp; chuyển mượt giữa foreground/background, chủ động/bị động theo nhu cầu người dùng; quá trình nền phải minh bạch và người dùng kiểm soát được.

### Agent (Time) — thời gian
- **Past**: phản ánh lịch sử (trạng thái + ngữ cảnh) để trả kết quả phù hợp hơn.
- **Now**: "nudge" hơn là "notify" — đơn giản hóa luồng, tạo gợi ý đúng lúc, đúng ngữ cảnh xã hội/văn hóa, đúng ý định người dùng.
- **Future**: thích ứng với thiết bị/nền tảng/phương thức, hành vi và nhu cầu accessibility của người dùng; tiến hóa qua tương tác liên tục.

### Agent (Core) — lõi
- **Embrace uncertainty but establish trust**: chấp nhận agent có lúc không chắc chắn; niềm tin và minh bạch là nền tảng; con người luôn kiểm soát được agent bật/tắt và thấy rõ trạng thái của nó.

### Ba guideline khi triển khai
1. **Transparency**: cho người dùng biết có AI tham gia, nó hoạt động thế nào (kể cả hành động đã làm), và cách góp ý/sửa hệ thống.
2. **Control**: cho phép tùy chỉnh, đặt preference, cá nhân hóa — kể cả quyền "quên" (xóa dữ liệu/hội thoại).
3. **Consistency**: trải nghiệm nhất quán, đa phương thức, trên mọi thiết bị; dùng UI quen thuộc (icon micro, kẹp giấy...), giảm tải nhận thức (trả lời ngắn gọn, hỗ trợ trực quan, "Learn More").

## Bài 04 — Tool Use Design Pattern

- **Tool** = đoạn code agent thực thi để đạt mục tiêu, được gọi theo **function call do model sinh ra**. Có thể là hàm đơn giản (máy tính) hoặc gọi API bên ngoài (giá cổ phiếu, thời tiết).
- Use cases: tra cứu thông tin động, chạy code/giải toán, tự động workflow, hỗ trợ khách hàng (CRM/ticket), sinh và sửa nội dung.

### 6 building blocks
1. **Function/Tool Schemas**: mô tả chi tiết tool (tên, mục đích, tham số, đầu ra) để LLM hiểu và gọi đúng.
2. **Function Execution Logic**: quyết định khi nào gọi tool nào (planner, routing, luồng điều kiện).
3. **Message Handling System**: quản lý hội thoại giữa user ↔ LLM ↔ tool call ↔ tool output.
4. **Tool Integration Framework**: hạ tầng nối agent với tool (hàm đơn giản hay service phức tạp).
5. **Error Handling & Validation**: xử lý lỗi thực thi, kiểm tra tham số, phản hồi bất ngờ.
6. **State Management**: theo dõi ngữ cảnh, lịch sử tool đã gọi, dữ liệu bền vững qua nhiều lượt.

### Function calling — 3 bước (ví dụ `get_current_time`)
1. Khởi tạo LLM **hỗ trợ function calling** (không phải model nào cũng hỗ trợ).
2. Gửi cho model **schema JSON** mô tả function (tên, mô tả, tham số). Model trả về **tool call** (tên function + arguments) — CHƯA phải câu trả lời cuối.
3. Code thực thi function, gửi kết quả trả lại model, model gọi API lần 2 để ra câu trả lời cuối.
- Với MAF, chỉ cần decorator `@tool` trên hàm Python — framework tự sinh schema và xử lý vòng gọi lại.
- Tool không nhất thiết là hàm trong code mình: **MCP server** cũng publish tool + schema (vd `MCPStreamableHTTPTool` nối tới server web search miễn phí của Keenable — chỉ cần credentials của model, query của model được gửi tới server đó).

### Trust: SQL do LLM sinh ra
- Nguy cơ: SQL injection hoặc hành động phá hoại (DROP database...).
- Cách khóa: cấp cho app **quyền chỉ đọc (SELECT)** trên database; tốt nhất là tách dữ liệu sang database/warehouse chỉ đọc, schema thân thiện, chạy trong môi trường an toàn.

## Bài 05 — Agentic RAG

- **RAG tĩnh**: retrieve rồi đọc — đường đi định sẵn. **Agentic RAG**: LLM **tự lập kế hoạch các bước tiếp theo** trong khi lấy thông tin từ nguồn ngoài — vòng lặp `LLM call → tool use → LLM call → ...`, đánh giá kết quả, viết lại query, đổi phương pháp retrieval, gọi thêm tool cho tới khi đủ tốt.
- Phong cách **maker-checker**: lặp lại để tăng độ đúng, xử lý query hỏng (vd NL2SQL), cho kết quả chất lượng cao.
- Điểm phân biệt cốt lõi: hệ thống **"own its reasoning process"** — tự quyết định trình tự bước đi dựa trên chất lượng thông tin tìm được, không chạy theo kịch bản người viết sẵn. Ví dụ chiến lược ra mắt sản phẩm: tự tra báo cáo thị trường (Bing) → tìm dữ liệu đối thủ (AI Search) → đối chiếu doanh số nội bộ (SQL) → tổng hợp → tự đánh giá chỗ thiếu → tra thêm vòng nữa.
- **Vòng lặp + memory**: Initial call → Tool invocation → Assessment & refinement → Repeat until satisfied; state/memory giúp nhớ các lần thử trước, tránh lặp lại vòng vô ích.
- **Self-correction**: thử lại/viết lại query, dùng diagnostic tool để debug suy luận, **fallback xin ý kiến người** khi việc quan trọng hoặc thất bại lặp lại.
- **Boundaries of agency**: KHÔNG phải AGI — chỉ tự chủ trong domain, với tool/data/policy do dev cấp; không tự chế tool mới, không vượt guardrails.
- **Use cases mạnh**: môi trường correctness-first (tuân thủ, pháp lý, nghiên cứu), tương tác DB phức tạp (query hay lỗi, cần tự sửa), workflow dài (thông tin mới xuất hiện thì đổi chiến lược).
- **Governance**: explainable reasoning (audit trail: đã query gì, tra nguồn nào, suy luận ra sao), kiểm soát bias trong retrieval, human oversight cho quyết định hệ trọng.
- Paper tham khảo trong bài: Self-Refine, Reflexion, CRITIC, Agentic RAG survey (arXiv:2501.09136).

## Áp dụng cho công việc của Niên và Hoai

- **Mẫu agent cho Niên Tool / shop bot / cron kiểm tra**: dùng pattern **tool-use + agentic RAG** — agent tra cứu tài liệu/nguồn rồi mới trả lời, tự kiểm tra kết quả trước khi báo Hoai (đúng bài học "verify artifact, đừng tin report").
- **Viết tool schema như viết docstring cho dev mới**: rõ ràng, có ví dụ, nêu trường hợp biên (nối bài Anthropic về ACI + bài Karpathy về function calling).
- **Khóa tool chạm dữ liệu quan trọng ở chế độ chỉ đọc**; mọi thao tác ghi/xóa/đăng bài đều cần người duyệt (human-in-the-loop) — đúng quy tắc "big decisions cần Hoai".
- **Audit trail**: log lại tool calls + nguồn đã tra để debug được khi agent sai (giống như log cron/watchdog đã làm).
- **Thiết kế theo 3 nguyên tắc bài 03**: minh bạch (Hoai luôn biết Niên đang làm gì), kiểm soát được (Hoai bật/tắt được), nhất quán (cùng một cách báo cáo).
- **Thực hành**: chạy notebook bằng client tương thích OpenAI + API miễn phí (xKiro/Xingchi) thay vì Azure; khi nào cần deploy thật thì mới tính tới Foundry.
