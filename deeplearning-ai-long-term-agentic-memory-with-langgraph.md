# Long-Term Agentic Memory With LangGraph — Ghi chú học tập

- **Khóa học:** Long-Term Agentic Memory With LangGraph — DeepLearning.AI × LangChain
- **Giảng viên:** Harrison Chase (Co-Founder & CEO LangChain)
- **Thời lượng:** 7 bài, 1 giờ 24 phút, trình độ Intermediate
- **Học phí:** miễn phí 100% (giai đoạn beta, nút "Enroll for Free", không cần thẻ)
- **Tài khoản học:** lhnien95@gmail.com (đăng nhập bằng Google, 23/11/2026)
- **Ngày học:** 23/11/2026
- **Cách học (ghi trung thực):** Bài 1 và bài 7 xem hết video. Bài 2–6 học qua toàn bộ transcript chính thức trên nền tảng (gồm cả phần walkthrough code trong notebook). Tiến độ hiển thị trên nền tảng: 22% (2/8 mục) vì chỉ 2 bài được phát video tới cuối. Quiz (Graded, 10 phút) và Appendix chưa làm — không thuộc 7 bài chính.

## Tổng quan

Ứng dụng agentic tồn tại lâu dài nên cần **trí nhớ dài hạn**. Chatbot chỉ lưu lịch sử hội thoại trong context; agent làm việc theo thời gian (ví dụ agent quản lý lịch, trợ lý email) cần nhớ qua nhiều lần gọi. Ba câu hỏi cốt lõi khi thiết kế trí nhớ:

1. **Lưu gì?** (what to store)
2. **Truy xuất gì khi dùng?** (what to retrieve)
3. **Cập nhật khi nào?** (when to update)

**LangMem** là thư viện mới của LangChain cho trí nhớ dài hạn: vector database có thể tìm kiếm, chia sẻ, lưu trữ bền vững, cập nhật ngay lập tức bởi agent hoặc ở nền background bởi helper agent. Cả khóa xây một **email assistant** minh họa bằng LangMem.

## Ba loại trí nhớ

| Loại | Nghĩa (theo con người) | Ánh xạ sang agent |
|---|---|---|
| **Semantic** (ngữ nghĩa) | Sự thật đã học (ví dụ: sinh nhật quan trọng) | Sự thật về user / người / địa điểm / vật |
| **Episodic** (tình tiết) | Trải nghiệm, ký ức (ví dụ: chuyến đi Disneyland) | Các hành động trước đây của agent dưới dạng few-shot examples — literal trajectories (đầu vào + đầu ra thật) |
| **Procedural** (thủ tục) | Hướng dẫn, kỹ năng vận động (ví dụ: cách đạp xe) | System prompt — hướng dẫn cách agent hành xử |

## Hai cơ chế cập nhật trí nhớ

| Cơ chế | Cách làm | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Hot path** | Chính agent đang trả lời tự cập nhật trí nhớ ngay lúc đó | Một agent duy nhất, đơn giản để duy trì; cập nhật tức thì | Agent phức tạp hơn; tăng độ trễ trả lời |
| **Background** (quy trình riêng) | Tách việc cập nhật khỏi agent chính; helper agent cập nhật sau | Agent chính đơn giản, phản hồi nhanh | Tổng thể có 2 agent nên phức tạp hơn; cập nhật không tức thì, chỉ khi được kích hoạt |

**Khung quyết định chọn loại trí nhớ:**
- Cần hướng dẫn tốt hơn → **procedural** (sửa system prompt)
- Cần few-shot examples hướng dẫn hành động → **episodic**
- Chỉ cần sự thật về người / địa điểm / vật → **semantic**

## Bài 1 — Introduction (2 phút)

- Trợ lý cá nhân AI càng học càng làm tốt hơn theo thời gian — đó là lý do cần trí nhớ dài hạn.
- 3 loại trí nhớ: semantic, episodic, procedural (xem bảng trên).
- LangMem: vector DB tìm kiếm được, chia sẻ được, lưu bền vững, cập nhật tức thì hoặc ở background.
- Cả khóa xây một email assistant minh họa.

## Bài 2 — Introduction to Agent Memory (8 phút)

- Khảo sát: trợ lý cá nhân / năng suất là ứng dụng phù hợp thứ hai với agent; trí nhớ là cốt lõi, tương tự trợ lý con người.
- Email agent mẫu: phân loại (triage) email thành **ignore / respond / notify**, kiểm tra lịch trống, viết email.
- Ba loại trí nhớ ánh xạ sang agent (xem bảng trên); hai cơ chế cập nhật hot path vs background (xem bảng trên).
- Khung quyết định (xem trên).
- Lộ trình khóa học: bài 3 xây email agent cơ bản → bài 4 thêm semantic → bài 5 thêm episodic → bài 6 thêm procedural.

## Bài 3 — Baseline Email Assistant (16 phút)

Chưa có trí nhớ nào. Nền tảng để các bài sau gắn trí nhớ vào.

**Profile người dùng:** tên, background, facts — có thể tùy chỉnh. **Prompt chia nhỏ thành module:** `triage_rules` (ignore/notify/respond) và `agent_instructions`, để sau này từng phần có thể được tạo ra từ trí nhớ thay vì hardcode.

**Code quan trọng:**

- Schema `Email` (from, to, subject, body).
- Pydantic `Router`: `reasoning: str` và `classification: Literal["ignore", "respond", "notify"]`, dùng `llm.with_structured_output(Router)` với GPT-4o-mini.
- `triage_system_prompt.format(...)`, `triage_user_prompt.format(author, to, subject, email_thread)`; `llm_router.invoke([SystemMessage, HumanMessage])`.
- Ba tool mock bằng `@tool` của LangChain: `write_email(to, subject, content)`, `schedule_meeting(attendees, subject, duration, preferred_day)`, `check_calendar_availability(day)` — thực tế thay bằng API Gmail/Outlook.
- Response agent: `create_react_agent(model="openai:gpt-4o", tools=[...], prompt=prompt_function)`, trong đó `prompt_function(state)` chèn system message lên đầu danh sách messages.
- Luồng chính: `StateGraph(EmailState)`; node triage trả về `Command(goto, update)` — respond thì goto `"response_agent"` và thêm HumanMessage `"Respond to the email: {email}"`, ignore/notify thì goto END; `add_edge(START, "triage")`; compile và vẽ bằng `get_graph(xray=True).draw_mermaid_png()`.

**Ví dụ kiểm tra:** email bán hàng spam → Ignore; email của Alice → Respond, agent gọi `write_email`.

## Bài 4 — Email Assistant with Semantic Memory (12 phút)

Semantic = sự thật về user; agent tự quản lý bằng tool trong **hot path** (chính agent đang trả lời tự quyết định lưu/tìm trí nhớ).

**Code quan trọng:**

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore(index={"embed": "openai:text-embedding-3-small"})
```

- Từ LangMem: `create_manage_memory_tool(namespace=("email_assistant", "{langgraph_user_id}", "collection"))` tạo tool `manage_memory` (tham số content, action create/update/delete, mặc định create) và `create_search_memory_tool(...)` tạo tool `search_memory` (query, limit, offset, filter; mô tả "Search your long-term memories for information relevant to your current context").
- Namespace có `{langgraph_user_id}` dạng template để **tách trí nhớ theo từng user**.
- Agent: `tools = [write_email, schedule_meeting, check_calendar_availability, manage_memory_tool, search_memory_tool]`; `create_react_agent(..., store=store)`; gọi với config `{"configurable": {"langgraph_user_id": "lance"}}`.

**Ví dụ:** nói "Jim is my friend" → agent gọi `manage_memory` lưu "Jim is John Doe's friend"; hỏi "who is jim?" → `search_memory(query="Jim")` rồi trả lời dựa trên trí nhớ.

**Kiểm tra store:** `store.list_namespaces()`, `store.search(namespace)`, `store.search(namespace, query="jim")` trả điểm tương đồng cosine (~0,5) nhờ embedding; không có query thì score là None.

**Ví dụ xuyên suốt:** email Alice hỏi về missing API endpoints → agent trả lời bằng `write_email` rồi lưu "Follow up needed. Alice Smith inquired about missing API endpoints."; email tiếp theo "Any update on my previous ask?" → triage respond, agent `search_memory(query="Alice Smith previous request")` rồi viết thư nhắc lại yêu cầu về API endpoints.

## Bài 5 — Email Assistant with Semantic + Episodic Memory (9 phút)

Episodic = kinh nghiệm quá khứ dưới dạng **few-shot examples chèn vào prompt ở bước triage**; cập nhật ở background (ở đây minh họa bằng cách ghi trực tiếp vào store).

**Code quan trọng:**

- Dữ liệu ví dụ `{"email": email, "label": "respond"}`; `store.put(("email_assistant", "lance", "examples"), str(uuid.uuid4()), data)` — namespace `…/examples` khác với semantic `…/collection`.
- Hàm `format_few_shot_examples(results)` định dạng "Email Subject / From / To" kèm "Triage Result = {label}".
- Trong `triage_router(state, config, store)`: namespace = `("email_assistant", config["configurable"]["langgraph_user_id"], "examples")`; `store.search(namespace, query=str(email_data), limit=1)` lấy ví dụ giống nhất; `triage_system_prompt` nay được định nghĩa ngay trong code (không import) và có mục few-shot examples với hướng dẫn "pay close attention to the examples".

**Ví dụ:** email của Tom Jones "Want to buy documentation" mặc định được phân loại respond; sau khi thêm một ví dụ tương tự với label ignore cho user "harrison", cùng email (thậm chí sửa nhẹ tên thành Jim Jones) → ignore; đổi sang user ID khác → lại respond, vì examples được tách theo user.

**Điểm đáng nhớ:** episodic ở đây không phải tóm tắt kinh nghiệm bằng lời, mà là **trajectory thật** (email đầu vào + nhãn đầu ra) dùng làm few-shot.

## Bài 6 — Email Assistant with Semantic + Episodic + Procedural Memory (15 phút)

Procedural = chính các **system prompt** (của agent chính và của triage). Trước đây `prompt_instructions` (triage_rules + agent_instructions) được hardcode; nay đưa vào store, agent truy xuất lúc chạy, và một agent khác có thể cập nhật chúng ở background mà không cần sửa code.

**Truy xuất prompt từ store:**

- Trong `triage_router(state, config, store)`, namespace = `(langgraph_user_id,)` (tuple một phần tử); `store.get(namespace, "triage_ignore")` — nếu None thì `store.put(namespace, "triage_ignore", {"prompt": giá trị mặc định})` rồi dùng mặc định, nếu có thì dùng `result.value["prompt"]`; tương tự cho `"triage_notify"`, `"triage_respond"`.
- Trong `create_prompt(state, config, store)`: cùng mẫu với key `"agent_instructions"`.

**Cập nhật prompt ở background bằng LangMem optimizer:**

```python
from langmem import create_multi_prompt_optimizer
optimizer = create_multi_prompt_optimizer("anthropic:claude-3-5-sonnet-latest", kind="prompt_memory")
```

- Claude 3.5 Sonnet được chọn vì tốt nhất cho tối ưu prompt; `kind="prompt_memory"` là dạng đơn giản nhất; optimizer tự nó là một graph.
- Đầu vào: `conversations = [(response["messages"], "Always sign your emails `John Doe`")]` — cặp (trajectory, feedback); `prompts` là danh sách mỗi mục gồm name, prompt (giá trị hiện tại lấy từ store), `update_instructions` (ví dụ "Keep the instructions short and to the point."), `when_to_update` (ví dụ main_agent: khi có feedback về cách agent viết email/lên lịch; triage_ignore: khi có feedback về email nào nên ignore).
- Gọi `updated = optimizer.invoke({"trajectories": conversations, "prompts": prompts})`; nếu `updated_prompt["prompt"]` khác cũ thì `store.put(namespace, key tương ứng theo name, {"prompt": ...})` — notebook chỉ triển khai đầy đủ trường hợp main_agent, ba trường hợp triage để lại làm bài tập; có kiểm tra ValueError nếu name không khớp.

**Ví dụ 1:** feedback "Always sign your emails `John Doe`" → prompt của agent chính được cập nhật thêm "When sending emails always sign them as John Doe.", chạy lại agent ký tên John Doe.
**Ví dụ 2:** feedback "Ignore any emails from Alice Jones" → chỉ `triage_ignore` được cập nhật ("ignore all emails from Alice Jones"), chạy lại cùng email của Alice → Ignore.

**Điểm đáng nhớ:** procedural memory chính là "hướng dẫn sống" — thay đổi hành vi agent bằng cách sửa prompt đã lưu, **không cần deploy lại code**; nhưng cần `when_to_update` rõ ràng để mỗi feedback chỉ sửa đúng prompt liên quan.

## Bài 7 — Conclusion (1 phút)

Đã biết phân loại trí nhớ thành semantic, episodic, procedural và thực hiện các thao tác trí nhớ ở hot path hoặc ở background, nơi thông tin được hợp nhất sau mà không ảnh hưởng độ trễ. Harrison hẹn gặp lại ở các dự án tương lai của người học.

## Áp dụng vào việc của anh Hoai

1. **Văn Phòng AI / trợ lý email:** email assistant trong khóa này là mẫu trực tiếp cho một agent xử lý Gmail — triage (ignore/respond/notify) + viết thư + kiểm tra lịch + nhớ sự thật về người liên hệ (semantic) và học từ các email đã xử lý trước đây (episodic).
2. **Trí nhớ theo từng user:** namespace `{langgraph_user_id}` — áp dụng được cho Văn Phòng AI khi nhiều người dùng chung một agent.
3. **Procedural memory:** thay vì sửa code mỗi lần muốn đổi cách agent hành xử, đưa system prompt vào store và cho optimizer cập nhật từ feedback — phù hợp với cách các phòng ban góp ý, chỉnh hành vi của Javis và các trưởng phòng AI.
4. **Hot path vs background:** với agent trả lời nhanh (chat), cập nhật trí nhớ nên để background để không tăng độ trễ; chỉ dùng hot path khi cần lưu ngay sự thật quan trọng trong lúc trả lời.
5. **Niên Score / phân loại:** episodic dưới dạng literal trajectories (đầu vào + nhãn) giống cách Niên Score học từ ví dụ — có thể dùng cùng ý tưởng để cải thiện phân loại khóa học/deal theo thời gian.

## Phần chưa làm

- Quiz (Graded, 10 phút) và Appendix — Tips and Helps (10 phút): không thuộc 7 bài chính; nếu muốn hoàn tất 100% trên nền tảng thì làm Quiz và phát hết video các bài 2–6 để nền tảng đánh dấu hoàn thành.
- Notebook thực hành của khóa (chạy code LangGraph/LangMem thật) chưa chạy — có thể làm tiếp khi cần áp dụng vào Văn Phòng AI.
