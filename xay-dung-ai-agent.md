# XÂY DỰNG AI AGENT — Danh sách tài nguyên đã xác minh

> Nguồn: ảnh chụp màn hình Hoai gửi tối 20/11/2026 (danh sách gốc của Sairam Sundaresan — bài đăng viral "After 100+ hours of trial, error, and breakdowns — These 8 videos are what you need to understand AI agents", cộng thêm các repo/guide/sách).
> Ngày xác minh: 20/11/2026. Tất cả link lnkd.in đều được giải ra URL đích thật; link nào trả 404 thì ghi rõ và xác định đích theo tên.
> Tổng cộng: **34 mục trong ảnh → 32 tài nguyên riêng biệt** (2 cặp trùng lặp, xem mục 3). **Tất cả 34 mục đều đã xác định được đích thật** (tên, tác giả/nguồn, miễn phí hay trả phí, tóm tắt).

## 1. Tổng quan

- Danh sách gồm 5 nhóm: 8 video YouTube, 12 kho GitHub, 5 hướng dẫn (guide), 7 sách, 2 bài nghiên cứu (ảnh bị cắt ở cuối — mục nghiên cứu có thể còn tiếp).
- **Hầu hết tài nguyên là MIỄN PHÍ**: toàn bộ video, toàn bộ repo GitHub, toàn bộ guide, 2 bài báo khoa học, và 1 sách có bản PDF miễn phí (Understanding Deep Learning).
- **Trả phí: 6 cuốn sách** (5 O'Reilly + 1 Manning) — nhưng 5/6 cuốn trong số đó có repo/bản code miễn phí đi kèm, nên vẫn học được phần thực hành mà không cần mua sách.
- Danh sách này khớp rất tốt với lộ trình AI Engineer của Cole (Python → ML → DL → CV → NLP → LLM/RAG → RL → XAI): phần ML/DL có ML-For-Beginners + Understanding Deep Learning, phần LLM/RAG/Agent có ai-agents-for-beginners + llm-course + các guide của Anthropic/Google/OpenAI.
- Lưu ý: Hoai đã học khóa GLM-5.3-Flash (tự build + train model 25M tham số từ đầu trên CPU), nên phần "LLM from scratch" trong danh sách này phần lớn là ôn lại — không cần đầu tư nhiều thời gian vào đó nữa.

## 2. Bảng từng mục đã xác minh

### 2.1. VIDEO (YouTube — tất cả miễn phí)

| # | Tên thật | Kênh / Tác giả | Thời lượng | Link đích thật | Tóm tắt | Phù hợp người mới |
|---|---|---|---|---|---|---|
| V1 | Intro to Large Language Models | Andrej Karpathy | 59 phút 23 giây (11/2023) | https://www.youtube.com/watch?v=zjkBMFhNj_g | Giới thiệu LLM từ đầu: tokenization, transformer, pretraining/fine-tuning, RLHF — giải thích trực quan, không cần toán nặng. Video nền tảng kinh điển, nên xem đầu tiên. | ⭐ Rất phù hợp — điểm bắt đầu số 1 |
| V2 | LLMs from Scratch (Stanford CS229 — "Building Large Language Models") | Stanford CS229 | chưa xác minh được | https://www.youtube.com/watch?v=9vM4p9NN0Ts | Bài giảng Stanford 2023: pretraining, post-training, tokenization, đánh giá, scaling laws, hệ thống. Học thuật, sâu hơn V1. | Khá — xem sau V1 |
| V3 | Agentic AI: A Progression of Language Model Usage | Stanford Online (Stanford Webinar, 05/02/2025) | chưa xác minh được | https://www.youtube.com/watch?v=kJLiOGle3Lw | Tổng quan tiến trình từ LLM → agent: tool use, planning, multi-agent. Góc nhìn học thuật, khung tư duy tổng thể về agentic AI. | Khá — xem để lấy khung tư duy |
| V4 | Building and evaluating AI agents that matter | Sayash Kapoor (Princeton), AI Engineer Summit 2025 | ~20 phút | https://www.youtube.com/watch?v=d5EltXhbcfA | Vì sao nhiều agent demo thất bại ngoài thực tế; cách đánh giá agent cho đúng (evals), chọn bài toán đáng làm. Ngắn, thực tế. | ⭐ Rất phù hợp — tư duy đánh giá |
| V5 | How We Build Effective Agents | Barry Zhang, Anthropic, AI Engineer Summit 2025 (New York) | chưa xác minh được | https://www.youtube.com/watch?v=D7_ipDqhtwk | Bản talk của bài viết nổi tiếng "Building Effective Agents" của Anthropic. 3 nguyên tắc: đừng agent hóa mọi thứ, giữ đơn giản, nghĩ như agent. | ⭐ Rất phù hợp — nguyên tắc cốt lõi |
| V6 | Building Agents with Model Context Protocol — Full Workshop with Mahesh Murag of Anthropic | AI Engineer (kênh), ghi hình 01/03/2025 (AI Engineer Summit 2025) | 1 giờ 44 phút | https://www.youtube.com/watch?v=kQmXtrmQ5Zg | Workshop đầy đủ về MCP: cách agent kết nối tool/dữ liệu qua MCP, code mẫu thực tế. Dài, thực hành. | Trung bình — học sau khi nắm agent cơ bản |
| V7 | Lets Build An Agent from Scratch | Kam Lasater, AI Engineer (kênh) | chưa xác minh được | https://www.youtube.com/watch?v=xzXdLRUyjUg | Xây một agent nhỏ từng bước bằng Python thuần, không dùng framework. Ngắn gọn, code trực tiếp, rất hợp để làm agent đầu tay. | ⭐ Rất phù hợp — thực hành đầu tiên |
| V8 | PhiloAgents (playlist, 6 video) | The Neural Maze (Miguel Otero Pedrido) + Paul Iusztin (Decoding ML, tác giả LLM Engineer's Handbook) | chưa xác minh được (tổng playlist) | https://www.youtube.com/playlist?list=PLacQJwuclt_sV-tfZmpT1Ov6jldHl30NR | Khóa học theo dự án: xây engine mô phỏng game với các triết gia AI (Plato, Aristotle, Turing) có memory, RAG, LLMOps. Stack: LangGraph, Groq, MongoDB, FastAPI, Docker, WebSockets, Opik. 6 tập: mô phỏng làng triết gia, LangGraph NPC, memory, realtime API, LLMOps observability + 1 video full course. | Nâng cao — làm sau khi xong agent cơ bản |

> ⚠️ Lưu ý về ID video: ảnh chụp màn hình dễ nhầm ký tự I/l/1/0/O. Các ID trong bảng trên là ID đã sửa đúng (đã đối chiếu qua tìm kiếm). Ví dụ: V2 là `9vM4p9NN0Ts` (số 0, không phải chữ O), V4 là `d5EltXhbcfA` (chữ L thường), V8 là `PLacQJwuclt_sV-tfZmpT1Ov6jldHl30NR`.

### 2.2. KHO GITHUB (code miễn phí — xem lưu ý API ở mục 5)

| # | Tên thật | Stars (20/11/2026) | Giấy phép | Link đích thật | Tóm tắt | Phù hợp người mới |
|---|---|---|---|---|---|---|
| G1 | GenAI_Agents — NirDiamant | 25.076 ⭐, 4.200 fork | Other (NOASSERTION, có file LICENSE) | https://github.com/nirdiamant/GenAI_Agents | 64 tutorial từ bot cơ bản tới hệ đa-agent (LangChain, LangGraph, MCP, RAG). Kho thực hành agent lớn, cập nhật thường xuyên. Sách đi kèm "AI Agents Made Simple" là TRẢ PHÍ (Amazon). | ⭐ Rất phù hợp — kho bài tập agent |
| G2 | ai-agents-for-beginners — Microsoft | 80.587 ⭐, 26.340 fork | MIT | https://github.com/microsoft/ai-agents-for-beginners | 18 bài học có cả chữ + video + code Python (Microsoft Agent Framework + Foundry; code mẫu cũng hỗ trợ provider tương thích OpenAI). Có bản dịch TIẾNG VIỆT. Mẫu Foundry cần tài khoản Azure. | ⭐ Rất phù hợp — khóa học agent có cấu trúc nhất trong danh sách |
| G3 | Prompt Engineering Guide — DAIR.AI | ~77–79,5k ⭐ (con số dao động theo nguồn, chưa xác minh được chính xác) | MIT (repo) | Site: https://www.promptingguide.ai — Repo: https://github.com/dair-ai/Prompt-Engineering-Guide (link rút gọn lnkd.in/gJJGbXQr trả 404, đích xác định theo tên) | Guide prompt engineering đầy đủ nhất hiện nay: 18 kỹ thuật, notebook, bài báo. Guide/notebook miễn phí; DAIR.AI Academy Pro là TRẢ PHÍ ($49/tháng hoặc $350/năm) — không cần mua. | ⭐ Rất phù hợp — học prompt trước khi làm agent |
| G4 | Hands-On Large Language Models — HandsOnLLM (repo code của sách O'Reilly, Jay Alammar & Maarten Grootendorst) | 30.220 ⭐, 6.847 fork | Apache-2.0 | https://github.com/HandsOnLLM/Hands-On-Large-Language-Models (link rút gọn lnkd.in/dxaVF86w) | Code theo từng chương sách: token, embedding, transformer, phân loại, RAG, multimodal, fine-tuning. Chạy được trên Google Colab miễn phí (GPU T4). **Sách in là TRẢ PHÍ; code repo miễn phí.** | ⭐ Rất phù hợp — học LLM bằng code + hình minh họa |
| G5 | AI Agents for Beginners — Microsoft | = G2 (trùng lặp, xem mục 3) | MIT | https://github.com/microsoft/ai-agents-for-beginners | — | — |
| G6 | GenAI Agents — NirDiamant | = G1 (trùng lặp, xem mục 3) | — | https://github.com/NirDiamant/GenAI_Agents (link rút gọn lnkd.in/dEt72MEy) | — | — |
| G7 | Made With ML — Goku Mohandas | 50.099 ⭐, 7.824 fork | MIT | Bài học: https://madewithml.com — Code: https://github.com/GokuMohandas/Made-With-ML (link rút gọn lnkd.in/d2dMACMf trả 404, đích xác định theo tên) | Khóa MLOps end-to-end: từ thiết kế, train, tuning, tracking (MLflow), serving, CI/CD tới continual learning — làm 1 dự án ML production hoàn chỉnh. Bài học 100% miễn phí; có khóa cohort 4 tuần tương tác TRẢ PHÍ (không bắt buộc). | Trung bình — học sau khi nắm ML cơ bản; rất hợp lộ trình Cole |
| G8 | Hands-On AI Engineering — Sumanth077 | 4.120 ⭐, 987 fork | MIT | https://github.com/Sumanth077/Hands-On-AI-Engineering | Tuyển tập ~60+ dự án thực hành: AI agent (đa-agent nghiên cứu, tài chính, RAG tự phản chiếu…), OCR, audio, multimodal, RAG, fine-tuning. Mỗi dự án có code + hướng dẫn đầy đủ. Nhiều dự án dùng API trả phí (OpenAI/Anthropic…) nhưng có lựa chọn local/Ollama. | Khá — kho ý tưởng + code mẫu để bắt chước |
| G9 | Awesome Generative AI Guide — aishwaryanr | ~29,8–30,1k ⭐ (dao động theo nguồn) | MIT | https://github.com/aishwaryanr/awesome-generative-ai-guide (link rút gọn lnkd.in/dJ8qxp3q trả 404, đích xác định theo tên) | Hub tổng hợp: 3 lộ trình (Dùng AI / Xây AI / Hiểu AI), 90+ khóa học miễn phí, 60 câu phỏng vấn, danh sách paper hàng tháng. Dùng làm bản đồ tra cứu, không phải khóa học tuyến tính. | ⭐ Phù hợp — tra cứu, tìm khóa học |
| G10 | Designing Machine Learning Systems — Chip Huyen (tài liệu kèm sách O'Reilly 2022) | 5.475 ⭐, 1.124 fork | chưa xác minh được | https://github.com/chiphuyen/dmls-book (link rút gọn lnkd.in/dEx8sQJK) | Tóm tắt từng chương + tài nguyên + danh sách công cụ MLOps + ôn tập ML cơ bản của sách "Designing Machine Learning Systems". Repo miễn phí; **sách in là TRẢ PHÍ**. Có bản dịch tiếng Việt. | Khá — đọc sau khi làm quen ML; thiên về hệ thống production |
| G11 | ML-For-Beginners — Microsoft | 92.345 ⭐, 22.953 fork | MIT | https://github.com/microsoft/ML-For-Beginners (link rút gọn lnkd.in/dBj3BAEY) | Giáo trình ML cổ điển 12 tuần, 26 bài, 52 quiz: regression, classification, clustering, NLP, time series, reinforcement — dùng scikit-learn, có bản dịch TIẾNG VIỆT, có video walkthrough. | ⭐ Rất phù hợp — nền tảng ML cho lộ trình Cole |
| G12 | llm-course — Maxime Labonne | 85.167 ⭐, 9.903 fork | Apache-2.0 | https://github.com/mlabonne/llm-course | 3 phần: LLM Fundamentals, LLM Scientist, LLM Engineer + notebook Colab (fine-tuning, quantization, RAG, agent). Lộ trình LLM thực hành đầy đủ nhất trong danh sách, miễn phí. Sách đi kèm "LLM Engineer's Handbook" là TRẢ PHÍ (xem S3). | ⭐ Rất phù hợp — học LLM → agent sau Python |

### 2.3. HƯỚNG DẪN (Guides — tất cả miễn phí)

| # | Tên thật | Tác giả / Nguồn | Link đích thật | Tóm tắt | Phù hợp người mới |
|---|---|---|---|---|---|
| H1 | Agents (Google Agent Whitepaper) | Julia Wiesinger, Patrick Marlow, Vladimir Vuskovic — Google, 09/2024 | https://www.kaggle.com/whitepaper-agents (link rút gọn lnkd.in/gFvClbSN trả 404, đích xác định theo tên) | Whitepaper định nghĩa agent: model + tools + orchestration; cognitive architectures; ví dụ LangChain; chạy trên Vertex AI. Khung khái niệm chuẩn của Google, PDF miễn phí. | ⭐ Rất phù hợp — đọc để nắm khung |
| H2 | Agent Companion (Google) | Google | https://www.kaggle.com/whitepaper-agent-companion (link rút gọn lnkd.in/gfmCrgAH) | Tài liệu đồng hành của whitepaper H1: ví dụ thực tế, code mẫu triển khai agent theo khung Google. Miễn phí. | Khá — đọc kèm H1 |
| H3 | Building Effective Agents | Anthropic (Engineering blog) | https://www.anthropic.com/engineering/building-effective-agents (link rút gọn lnkd.in/gRWKANS4) | Bài viết nền tảng về agent: workflows vs agents, khi nào dùng agent, patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer), MCP, evals. Miễn phí, ngắn gọn. | ⭐ Rất phù hợp — đọc ngay sau V1 |
| H4 | Claude Code Best Agentic Coding Practices | Anthropic (Engineering blog) | https://www.anthropic.com/engineering/claude-code-best-practices (link rút gọn lnkd.in/gs99zyCf) | Best practices dùng Claude Code làm agent lập trình: CLAUDE.md, workflow, subagent, hooks. Miễn phí. Hữu ích khi Hoai dùng Claude Code/FCC/Codex để code. | Khá — đọc khi bắt đầu dùng agent để code |
| H5 | A Practical Guide to Building Agents | OpenAI, 2025 (PDF 34 trang) | Trang web: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/ — PDF: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf (link rút gọn lnkd.in/guRYx5FK trả 404, đích xác định theo tên) | Hướng dẫn thực tế: chọn model/tools/instructions, chọn use case, mẫu orchestration, guardrails, evals. Miễn phí. Góc nhìn của OpenAI, bổ sung cho H3 của Anthropic. | ⭐ Rất phù hợp — đọc kèm H3 |

### 2.4. SÁCH (Books — 6/7 trả phí, 1 miễn phí PDF)

| # | Tên thật | Tác giả / NXB | Trả phí? | Link đích thật | Tóm tắt | Ghi chú cho Hoai |
|---|---|---|---|---|---|---|
| S1 | Understanding Deep Learning | Simon J.D. Prince, MIT Press, 2023 (544 trang) | **MIỄN PHÍ bản PDF** (CC BY-NC-ND, open-access do MIT Press phát hành); bản in ~$100 trả phí | https://udlbook.github.io/udlbook/ (trang web không tải được nội dung trích xuất, đã xác minh qua tìm kiếm: PDF chính thức tại github.com/udlbook/udlbook) | Sách deep learning toàn diện: ML nền tảng, mạng nơ-ron, training, CNN, sequence model, attention, transformer, generative, diffusion. Có notebook Python + bài tập. | ⭐ Sách DL miễn phí tốt nhất trong danh sách — dùng cho giai đoạn DL của lộ trình Cole |
| S2 | Build a Large Language Model from Scratch | Sebastian Raschka, Manning | **TRẢ PHÍ** (sách Manning); code repo miễn phí | https://www.manning.com/books/build-a-large-language-model-from-scratch (link rút gọn lnkd.in/g2YGbnWS) | Tự code LLM từ đầu bằng PyTorch: tokenizer, transformer, pretraining, fine-tuning. Code tại github.com/rasbt/LLMs-from-scratch (miễn phí). | Hoai ĐÃ làm tương tự với khóa GLM-5.3-Flash (25M tham số, train trên CPU) → **không cần mua**, chỉ tham khảo code repo miễn phí nếu cần |
| S3 | The LLM Engineer's Handbook | Paul Iusztin & Maxime Labonne, O'Reilly (Packt) | **TRẢ PHÍ** (O'Reilly) | https://www.oreilly.com/library/view/llm-engineers-handbook/9781836200079/ (link rút gọn lnkd.in/gWUT2EXe) | Sách đi kèm repo G12 (llm-course): LLM engineering thực chiến — fine-tuning, RAG, agent, deploy. | Repo G12 miễn phí đã đủ; **không cần mua sách** ở giai đoạn này |
| S4 | AI Agents: The Definitive Guide | Nicole Koenstein, O'Reilly | **TRẢ PHÍ** (O'Reilly) | https://www.oreilly.com/library/view/ai-agents-the/0642572247775/ (link rút gọn lnkd.in/dJ9wFNMD) | Tổng quan toàn diện về AI agent: khái niệm, kiến trúc, ứng dụng, triển khai. | Chưa cần — các guide miễn phí H1/H3/H5 đã bao quát phần cốt lõi |
| S5 | Building Applications with AI Agents: Designing and Implementing Multiagent Systems | Michael Albada, O'Reilly, 10/2025 (352 trang, ISBN 978-1-098-17650-1) | **TRẢ PHÍ** (O'Reilly; khoảng $56–80 theo Amazon) | https://learning.oreilly.com/library/view/building-applications-with/9781098176495/ (link rút gọn lnkd.in/d5s8srk5 trả 404, đích xác định theo tên) | Thiết kế hệ agent từ đơn tới đa-agent: tool design, orchestration, memory/RAG/GraphRAG, học từ kinh nghiệm, UX cho agent, validation, monitoring production. Tác giả từng làm ML ở Uber, ServiceNow, Microsoft. Lưu ý: code ví dụ dùng langchain 0.1.20/langgraph 0.0.28 (cũ, trước 1.0) — đọc lấy khái niệm, tự port code. | Nâng cao, giai đoạn sau; **không mua vội** |
| S6 | AI Agents with MCP | Kyle Stratis, O'Reilly | **TRẢ PHÍ** (O'Reilly; giá chưa xác minh được) | https://www.oreilly.com/library/view/ai-agents-with/9798341639546/ (link rút gọn lnkd.in/dRR2bETZ trả 404, đích xác định qua kết quả tìm kiếm) | Sách về xây agent với MCP (Model Context Protocol). Chi tiết chưa xác minh được đầy đủ. | Học MCP miễn phí qua V6 + guide H3 trước; **không mua vội** |
| S7 | AI Engineering: Building Applications with Foundation Models | Chip Huyen, O'Reilly, 12/2024 | **TRẢ PHÍ** (O'Reilly; cần đăng ký O'Reilly hoặc mua sách) | https://www.oreilly.com/library/view/ai-engineering/9781098166298/ | Quy trình xây ứng dụng bằng foundation model: AI stack mới, eval (kể cả AI-as-a-judge), prompt engineering, RAG, fine-tuning, agent, dataset engineering, triển khai (latency/cost). Bổ sung cho "Designing ML Systems" của cùng tác giả. | Sách gối đầu giường của lộ trình AI Engineer, nhưng **không mua vội** — repo + guide miễn phí đã đủ cho giai đoạn đầu |

### 2.5. BÀI NGHIÊN CỨU (Research — tất cả miễn phí, arXiv)

| # | Tên thật | Tác giả | Link đích thật | Tóm tắt | Phù hợp người mới |
|---|---|---|---|---|---|
| R1 | ReAct: Synergizing Reasoning and Acting in Language Models | Shunyu Yao (Princeton) và cộng sự (Google), ICLR 2023, arXiv:2210.03629 | https://arxiv.org/abs/2210.03629 (link rút gọn lnkd.in/dRBH37Rg trả 404, đích xác định theo tên) | Bài báo nền tảng của agent: cho LLM xen kẽ "suy nghĩ" (Thought) → "hành động" (Action) → "quan sát" (Observation), giảm ảo giác, xử lý được nhiệm vụ tương tác. Mọi framework agent hiện nay đều vay mượn vòng lặp này. | Đọc sau khi hiểu agent cơ bản — chỉ cần đọc phần ý tưởng + ví dụ, không cần đọc hết toán |
| R2 | Generative Agents: Interactive Simulacra of Human Behavior | Joon Sung Park (Stanford) và cộng sự (Google), 04/2023, arXiv:2304.03442 | https://arxiv.org/abs/2304.03442 (link rút gọn lnkd.in/gsDCUsWm) | Mô phỏng một làng nhỏ với 25 agent AI có memory, tự lên kế hoạch, tương tác xã hội, tổ chức tiệc Valentine — chứng minh agent có thể có hành vi xã hội đáng tin. Đây cũng là ý tưởng gốc của playlist PhiloAgents (V8). | Đọc để lấy cảm hứng — không cần đọc kỹ thuật sâu |

## 3. Trùng lặp trong danh sách

| Cặp trùng | Chi tiết |
|---|---|
| GitHub §2 = GitHub §5 | Cùng là `microsoft/ai-agents-for-beginners` (80.587 ⭐, MIT). Chỉ học 1 lần. |
| GitHub §1 = GitHub §6 | Cùng là `nirdiamant/GenAI_Agents` (25.076 ⭐). §6 là link rút gọn lnkd.in/dEt72MEy trỏ về cùng repo. Chỉ học 1 lần. |
| (Không trùng nhưng liên quan) | Sách S3 "LLM Engineer's Handbook" là sách đi kèm repo G12 `mlabonne/llm-course`; playlist V8 PhiloAgents do Paul Iusztin (đồng tác giả S3) giảng dạy; sách S7 "AI Engineering" là phần tiếp theo của sách trong repo G10 "Designing ML Systems" (cùng tác giả Chip Huyen). |

→ **Số tài nguyên riêng biệt thực tế: 32** (34 mục − 2 cặp trùng).

## 4. Thứ tự học đề xuất cho Hoai (NÊN HỌC TRƯỚC)

Căn cứ: Hoai là kỹ sư điện/PLC, bằng CNTT, đã xong freeCodeCamp JS + React, đang học Relational Databases, tiếp theo là Python v9, đã tự build + train model 25M tham số (GLM-5.3-Flash) trên CPU, theo lộ trình Cole Python → ML → DL → CV → NLP → LLM/RAG → RL → XAI. Ưu tiên: miễn phí, thực hành, nền tảng trước rồi tới agent.

### Giai đoạn 0 — Đang làm (không đổi)
- **freeCodeCamp Python v9** (tiếp theo sau Relational Databases) — Python là điều kiện tiên quyết cho mọi repo trong danh sách này.

### Giai đoạn 1 — Nền tảng LLM + Agent (bắt đầu ngay, song song với Python)
1. **V1 — Karpathy "Intro to Large Language Models" (59 phút)** — Hiểu LLM hoạt động thế nào. Xem 1 buổi là xong.
2. **H3 — Anthropic "Building Effective Agents" (bài viết, ~30 phút đọc)** — Nguyên tắc agent cốt lõi. Đọc ngay sau V1.
3. **G3 — Prompt Engineering Guide (DAIR.AI)** — Học prompt cho chắc trước khi làm agent; tra cứu dần, không cần đọc hết một lượt.
4. **V7 — "Build an Agent from Scratch" (Kam Lasater)** — Agent đầu tay bằng Python thuần, không framework. Làm theo code trong 1 buổi.

### Giai đoạn 2 — Khóa học agent có cấu trúc
5. **G2 — Microsoft "AI Agents for Beginners" (18 bài, có tiếng Việt)** — Học theo từng bài, làm bài tập code. Đây là khóa chính của giai đoạn này.
6. **H1 + H5 — Google Agent Whitepaper + OpenAI Practical Guide** — Đọc để nắm khung thiết kế agent (model + tools + orchestration, guardrails, evals). Đọc song song với G2.

### Giai đoạn 3 — LLM sâu + nền tảng ML/DL (theo lộ trình Cole)
7. **G12 — mlabonne/llm-course (miễn phí)** — LLM Fundamentals → fine-tuning, quantization, RAG, agent. Làm trên Colab miễn phí.
8. **G11 — Microsoft ML-For-Beginners (12 tuần, 26 bài, có tiếng Việt)** — Nền tảng ML cổ điển (scikit-learn) cho lộ trình Cole. Học sau Python.
9. **S1 — Understanding Deep Learning (PDF miễn phí)** — Sách DL để đọc dần ở giai đoạn Deep Learning; không cần đọc hết, đọc theo chương cần dùng.

### Giai đoạn 4 — Dự án tổng hợp (sau khi xong giai đoạn 2–3)
10. **V8 — PhiloAgents playlist (6 video)** — Dự án capstone: xây game mô phỏng triết gia AI với LangGraph + MCP + memory + LLMOps. Làm khi đã quen LangGraph.
11. **G1 — GenAI_Agents (64 tutorial)** — Kho bài tập để luyện thêm từng kỹ thuật agent cụ thể (RAG, multi-agent, MCP…).
12. **G8 — Hands-On AI Engineering (dự án mẫu)** — Lấy ý tưởng dự án thực tế (agent nghiên cứu, RAG, OCR…) để làm sản phẩm của riêng mình.
13. **R1 + R2 — ReAct + Generative Agents (bài báo)** — Đọc khi muốn hiểu sâu vì sao agent hoạt động như vậy; R2 là cảm hứng trực tiếp cho V8.

### Tóm tắt top 5 nên học trước
1. **V1 Karpathy LLM intro** (video, 59 phút, miễn phí)
2. **H3 Anthropic "Building Effective Agents"** (bài viết, miễn phí)
3. **G2 Microsoft AI Agents for Beginners** (18 bài, có tiếng Việt, miễn phí)
4. **V7 Build an Agent from Scratch** (video thực hành, miễn phí)
5. **G12 mlabonne/llm-course** (miễn phí, Colab)

## 5. Mục cần lưu ý

### ⚠️ TRẢ PHÍ — không mua (theo quy tắc: không đăng ký gói trả phí)
- **6 cuốn sách trả phí**: S2 Raschka (Manning), S3 LLM Engineer's Handbook, S4 AI Agents: The Definitive Guide, S5 Albada, S6 AI Agents with MCP, S7 AI Engineering (đều O'Reilly trừ S2). **Không cần mua cuốn nào ở giai đoạn này** vì: 5/6 cuốn có repo code miễn phí đi kèm (G4, G12…), và các guide miễn phí H1/H3/H5 đã bao quát phần cốt lõi của S4/S7.
- **DAIR.AI Academy Pro** ($49/tháng hoặc $350/năm) — guide G3 miễn phí, bỏ qua phần Pro.
- **Made With ML cohort 4 tuần** — bài học miễn phí, bỏ qua cohort trả phí.
- **Sách "AI Agents Made Simple"** (đi kèm repo G1) — trả phí trên Amazon, không cần.

### ⚠️ Chi phí ẩn khi chạy code trong repo
- Nhiều repo (G1, G8, một phần G2) dùng API OpenAI/Anthropic/Google trả phí trong code mẫu. Khi chạy, **ưu tiên model miễn phí/local**: Ollama trên máy, hoặc các API miễn phí Hoai đã có (Xingchi relay, xKiro, Nara, LLM7, Groq…). Đừng nạp tiền API chỉ để chạy tutorial.
- Mẫu Foundry trong G2 cần tài khoản Azure — bỏ qua phần đó, dùng code mẫu tương thích OpenAI thay thế.

### ⚠️ Link rút gọn lnkd.in bị lỗi (8 link trả 404)
- Các link sau **không mở được trực tiếp** (nhiều khả năng do lỗi ký tự hoa/thường khi chụp màn hình): G3 (gJJGbXQr), G7 (d2dMACMf), G9 (dJ8qxp3q), H1 (gFvClbSN), H5 (guRYx5FK), S5 (d5s8srk5), S6 (dRR2bETZ), R1 (dRBH37Rg).
- Tất cả 8 đích đều đã được xác định theo tên qua tìm kiếm và ghi link đích thật trong bảng trên — dùng link đích thật, không dùng link rút gọn.

### ⚠️ Danh sách bị cắt ở cuối ảnh
- Mục "BÀI NGHIÊN CỨU" chỉ thấy 2 mục (ReAct, Generative Agents) vì ảnh chụp bị cắt. **Danh sách gốc có thể còn tiếp** (các bài báo khác như Chain-of-Thought, Tree of Thoughts, AutoGPT… thường xuất hiện trong các danh sách tương tự). Nếu Hoai có ảnh đầy đủ hơn thì gửi thêm để bổ sung.

### ⚠️ Số liệu chưa xác minh được
- Thời lượng video: V2, V3, V5, V7 và tổng thời lượng playlist V8 — chưa xác minh được.
- Số star chính xác: G3 Prompt Engineering Guide (~77–79,5k, dao động theo nguồn), G9 Awesome Generative AI Guide (~29,8–30,1k, dao động theo nguồn).
- Giấy phép repo G10 dmls-book — chưa xác minh được.
- Giá sách S6 AI Agents with MCP — chưa xác minh được.
- Nội dung chi tiết sách S6 — chỉ xác định được tên/tác giả/NXB qua kết quả tìm kiếm, chưa đọc được mục lục đầy đủ.
- Số liệu star/fork trong ghi chú này là số đọc được ngày 20/11/2026, sẽ thay đổi theo thời gian.

### ✅ Điểm tốt của danh sách này
- 100% video + guide + repo + bài báo đều miễn phí; chỉ sách là trả phí mà vẫn có đường vòng miễn phí.
- Có 2 tài nguyên tiếng Việt (G2, G11) — hiếm trong các danh sách tiếng Anh.
- Bao phủ đúng lộ trình Cole: ML (G11) → DL (S1) → LLM (G12, G4) → RAG/Agent (G2, G1, H3, H5) → MLOps/production (G7, G10, S7).
