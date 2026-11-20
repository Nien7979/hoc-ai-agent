# Bài đã học: "Lets Build An Agent from Scratch" — Kam Lasater (AI Engineer)

- Nguồn: video talk tại hội nghị AI Engineer (kênh Kam Lasater, AI Engineer), https://www.youtube.com/watch?v=xzXdLRUyjUg
- Slides + code: https://kamlasater.com/talks/agents-2025 (bản có transcript + tóm tắt: https://ai.engineer/talks/xzXdLRUyjUg)
- Ngày học: 20/11/2026 (đêm). Học theo transcript đầy đủ (96 dòng, lấy qua youtubetotranscript.com vì yt-dlp bị YouTube chặn bot từ IP máy ảo).
- Đây là bài số 4 trong top 5 lộ trình học AI Agent. Thời lượng video chưa xác minh được.

## 1. Định nghĩa agent — đơn giản hơn ta tưởng

- **Agent = LLM + memory + planning + tools + một vòng lặp while.** Đó là toàn bộ định nghĩa.
- Phân tích toán học một chút: memory thực ra là cả đọc và ghi; vòng lặp while thực ra là một điều kiện (condition) + lặp lại. Sắp xếp lại 4 thành phần đó là ra kế hoạch xây agent từng bước.
- 3 mục tiêu của talk: (1) trải nghiệm phiên bản đơn giản nhất của một agent; (2) thoải mái chạy và PHÁ code đi (chạy + phá là phần học quan trọng nhất); (3) mang về trực giác về cách agent/framework hoạt động bên dưới.
- Thông điệp chính: framework chỉ là những building block này đóng gói lại. Hiểu được chúng thì dùng framework nào cũng hiểu.

## 2. Step 0 — Gọi LLM (điểm xuất phát)

- Chat completion chuẩn của OpenAI (hello world trong docs): gửi prompt, nhận câu trả lời. Chưa có gì là agent cả.
- Đây là nền để mọi bước sau chồng lên.

## 3. Step 1 — Điều kiện: LLM-as-judge

- Sau khi LLM trả lời, gọi thêm MỘT lần LLM nữa đóng vai "strict critic" (nhà phê bình khó tính): "hãy xác định xem câu trả lời này có phải là câu trả lời đầy đủ cho câu hỏi không".
- Ép output thành JSON strict: `{done: boolean}` — đúng/sai, không được nói vòng vo.
- Ví dụ demo: "What is the average wing speed of a swallow?" (tốc độ bay trung bình của chim én) — judge cho thumbs up, câu trả lời được trả về cho user.
- Tới đây mọi thứ vẫn rất deterministic, rất cơ học: hỏi → trả lời → chấm điểm → trả về.

## 4. Step 2 — Tools: LLM được chạm vào thế giới thật

- Thêm tool tìm kiếm Google qua SerpAPI (dịch vụ bọc Google Search API thành JSON dễ dùng).
- **Tool config viết tay bằng JSON** để nhìn rõ cấu trúc: tên function, mô tả, `strict` (bám chặt schema), parameters (`query`, `location` — cái nào bắt buộc).
- Điểm quan trọng: **OpenAI SDK không tự chạy tool**. LLM chỉ trả về "hãy gọi tool X với tham số Y"; code phía client phải tự đọc `tool_calls`, gọi hàm local, rồi đẩy cả lời gọi tool lẫn kết quả trả về vào lại conversation, rồi gọi LLM tiếp (hàm `completeWithTools` — đệ quy).
- Demo: mua áo hoodie lót lông gần Times Square, New York. Lần chạy đầu **thất bại thật ngay trên sân khấu**: tham số `location` bị ràng buộc validation — chỉ nhận một tập chuỗi cố định, không phải chuỗi tùy ý. (Bài học: tool cần validate tham số, và lỗi validation là chuyện thường gặp, không phải lỗi của agent.)
- Sau khi sửa, judge lại chấm thumbs up: LLM quyết định cần search → search trả về text → LLM đánh giá text đó có trả lời được câu hỏi mua áo ở đâu không.

## 5. Step 3 — Refactor + gọi nhiều tool song song

- Dọn code: đưa `completeWithTools` sang file utils, khỏi file main.
- **Parallel tool calls**: LLM có thể yêu cầu gọi nhiều tool cùng lúc (mảng các tool call) — chạy bằng `Promise.all`, truyền arguments của từng tool call vào đúng hàm.
- Mỗi tool call có `tool_call_id` để LLM truy vết: "tôi đã yêu cầu gọi tool này, đây là ID, và đây là kết quả trả về". Đôi khi LLM yêu cầu gọi cùng một tool nhiều lần với tham số khác nhau.
- **Mô hình tư duy: tool là phép biến đổi text.** Đầu vào có thể nhiều tham số, nhưng cốt lõi thường là string → string: nhận text, trả text.
- Tool `browse_web` tương tự `search_google`: nhận URL, dùng Cheerio + Turndown tải trang, trả về markdown của trang đó.
- Tới đây vẫn còn cảm giác deterministic — chỉ là sắp xếp lại các khối đã có.

## 6. Step 4 — Planning: bước ngoặt, agent bắt đầu "cảm giác" như agent

- Thay đổi lớn nhất: **thêm một to-do list**. To-do list giải quyết cả read + write (memory) và planning cùng lúc.
- Vòng lặp while được LLM tự lo: cứ gọi tool → nhận kết quả → gọi tiếp → làm việc dần tới đáp án.
- **Nguy cơ thật: LLM có thể lặp mãi không hội tụ** (gọi tool hoài, không bao giờ xong) — nhất là khi context bị cắt mà không chạm ngưỡng lỗi nào. Guardrail thực tế: giới hạn số vòng lặp tối đa rồi cắt ("you're drunk, go home").
- To-do list tools: `add_todos` (thêm mảng to-do), `mark_done` (đánh dấu xong, gỡ khỏi danh sách), `check_todos` (xem còn gì), `check_done` (xem đã xong hết chưa — trả về "no tasks have been marked done" nếu chưa có gì xong).
- **System prompt lái planning**: "Bạn là trợ lý của một giám đốc bận rộn. Giọng thân thiện nhưng trực tiếp. Luôn lập kế hoạch bằng to-do list trước khi làm việc. Đánh dấu to-do sau khi xong. Tóm tắt hành động đã làm bằng cách kiểm tra to-do list, rồi tạo báo cáo. Luôn nhờ assistant kiểm tra goal done. Nếu xong thì gửi báo cáo cho user; nếu assistant có feedback thì thêm vào to-do list." (Có thêm ngày hôm nay vào prompt vì LLM đôi khi không biết đang là ngày nào.)
- **LLM-as-judge lúc này trở thành một tool** (`check_goal_done`): điều kiện (condition) là một tool, vòng lặp do LLM tự xử lý — và hàm main chỉ còn là một lần gọi `completeWithTools` duy nhất.

## 7. Demo: agent tự học + lên kế hoạch buổi tối

- Prompt: "I want to learn about building agents without a framework" → agent tự lập plan (search thông tin → tóm tắt điểm chính → nhờ assistant kiểm tra xem giải thích đã đủ chưa), search Google, browse blog, browse nhiều trang web, đánh dấu từng to-do đã xong, judge thumbs up, trả về bản tóm tắt các bước xây agent không framework.
- Demo thứ hai: lên kế hoạch date với vợ tối thứ Bảy (hoạt động + ăn tối 18h–23h) — agent search hoạt động địa phương, tìm nhà hàng, browse web, đánh dấu xong, trả về danh sách gợi ý.
- Tác giả thừa nhận: agent chưa hoàn hảo, nhưng đủ để thấy từng thành phần (LLM call → condition → tool → planning + read/write) cộng lại tạo ra thứ thú vị.

## 8. Hướng mở rộng

- Thêm **vector database**: đưa các trang web đã browse vào Chroma DB (in-memory) để tạo hệ thống kiểu RAG.
- Từ đó agent không chỉ search rồi quên, mà còn lưu kiến thức lại để dùng cho lần sau.

## Áp dụng cho công việc của Niên và Hoai

- **To-do list + vòng lặp + judge = đúng cách Niên đang làm việc.** Todo list của Niên (todo.write) chính là planning tool trong bài này; vòng lặp cron/watchdog (kiểm tra → hành động → kiểm tra lại tới khi xong) chính là `completeWithTools`; và quy tắc "verify trước khi báo" chính là LLM-as-judge với JSON strict.
- **Judge phải là một lần gọi riêng, nghiêm khắc, ép JSON.** Không được để cùng một lần sinh text vừa trả lời vừa tự chấm điểm — tách ra như bài này làm: một call trả lời, một call chấm.
- **Tool là phép biến đổi text (string → string).** Mỗi phòng ban trong công ty thu nhỏ cũng là một "tool" như vậy: nhận text (nhiệm vụ), trả text (kịch bản, video, báo cáo). Niên là vòng lặp điều phối các tool đó. Hiểu được khung này thì thiết kế thêm phòng/bot mới chỉ là thêm tool + sửa system prompt.
- **Guardrail "you're drunk, go home" = giới hạn số lần lặp.** Đúng như cron backup dừng sau N lần timeout, watchdog nén báo cáo trùng lặp — agent nào cũng cần giới hạn vòng lặp, không để nó chạy mãi.
- **Lỗi validation trên sân khấu = bài học thật.** Tool phải validate tham số trước khi gọi (như Niên Tool validate args), và khi tool lỗi thì báo rõ ràng chứ không đổ cho agent.
- **Bài tập thực hành đầu tay cho Hoai (khi cần):** viết một agent nhỏ bằng Python thuần theo đúng cấu trúc 4 bước này, chạy bằng API miễn phí đã có (xKiro/Xingchi) thay vì OpenAI trả phí — ví dụ agent kiểm tra deal miễn phí: search → judge xem có phải 0đ thật không → ghi vào to-do → báo cáo. Không cần framework, chỉ cần 1 file Python + vòng lặp while + 3 tool (search, judge, ghi file).
