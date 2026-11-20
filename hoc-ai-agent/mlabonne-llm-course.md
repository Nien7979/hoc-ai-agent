# Bài đã học: mlabonne/llm-course — Khóa học LLM (Maxime Labonne)

- Nguồn: repo https://github.com/mlabonne/llm-course (85.167 ⭐, MIT) — đã fork về https://github.com/Nien7979/llm-course và clone về máy. Repo chỉ chứa README + ảnh: nội dung khóa học là các notebook Colab + bài viết trên blog của tác giả (đều miễn phí).
- Tác giả: Maxime Labonne (@maximelabonne), đồng tác giả sách "LLM Engineer's Handbook" (sách này TRẢ PHÍ — không mua; khóa học vẫn miễn phí hoàn toàn).
- Ngày học: 20/11/2026 (đêm). Đã đọc toàn bộ README của khóa học (459 dòng) — bản đồ đầy đủ 3 phần, 20 module, kèm notebook thực hành.
- Phạm vi bài này: đây là khóa học nhiều tuần. Đêm nay học xong **toàn bộ bản đồ khóa học + khái niệm cốt lõi của từng module** theo đúng giáo trình của tác giả; các notebook thực hành chạy trên Colab miễn phí được xếp theo thứ tự ưu tiên ở mục 5 để làm khi cần dùng thật. **Với bài này, top 5 lộ trình học AI Agent đã HOÀN THÀNH.**

## 1. Cấu trúc khóa học — 3 phần

1. **🧩 LLM Fundamentals** (tùy chọn): toán, Python, neural network, NLP — nền tảng, tra cứu khi cần.
2. **🧑‍🔬 The LLM Scientist**: xây model LLM tốt nhất bằng kỹ thuật mới nhất (kiến trúc → pre-train → SFT → alignment → eval → lượng tử hóa).
3. **👷 The LLM Engineer**: xây ứng dụng từ LLM và đưa vào production (chạy model → RAG → agent → tối ưu inference → deploy → bảo mật).

## 2. Phần 1 — LLM Fundamentals

- **Toán cho ML**: đại số tuyến tính (vector, ma trận, trị riêng), giải tích (đạo hàm, gradient), xác suất-thống kê (phân phối, MLE, Bayesian). Tài nguyên: 3Blue1Brown, StatQuest, Khan Academy.
- **Python cho ML**: cú pháp + NumPy/Pandas/Matplotlib/Seaborn + tiền xử lý dữ liệu + scikit-learn (hồi quy, cây quyết định, random forest, KNN, K-means, PCA/t-SNE).
- **Neural Networks**: layer, weight, bias, activation (sigmoid/tanh/ReLU); backpropagation; loss (MSE, Cross-Entropy); optimizer (SGD, RMSprop, Adam); overfitting + regularization (dropout, L1/L2, early stopping); tự tay build MLP bằng PyTorch.
- **NLP**: tokenization, stemming/lemmatization, stop words; BoW, TF-IDF, n-grams; word embedding (Word2Vec, GloVe, FastText); RNN/LSTM/GRU.

## 3. Phần 2 — The LLM Scientist (xây model)

### 3.1 Kiến trúc LLM
- Từ Transformer encoder-decoder → decoder-only như GPT. 3 bước chính: tokenization (text → số) → attention qua các layer → sinh text bằng sampling.
- Sampling: greedy/beam search (deterministic) vs temperature/nucleus sampling (xác suất). Tài nguyên: 3Blue1Brown "Visual intro to Transformers", Karpathy nanoGPT (tự viết lại GPT từ đầu, 2h).

### 3.2 Pre-training
- Tốn kém, không phải trọng tâm của khóa học — nhưng phải hiểu: dữ liệu khổng lồ (Llama 3.1: 15 nghìn tỷ token), cần lọc/dedup/tokenize kỹ; huấn luyện phân tán (data parallel + pipeline parallel + tensor parallel); learning rate warm-up, gradient clipping, mixed-precision, optimizer AdamW/Lion; theo dõi loss/gradient/GPU.
- Người chơi nhỏ vẫn pre-train được model <1B tham số.

### 3.3 Post-training datasets
- Cấu trúc khác pre-training: instruction + answer (SFT) hoặc instruction + chosen/rejected (alignment). Lưu ở format ShareGPT/OpenAI rồi map sang chat template (ChatML, Alpaca).
- **Synthetic data**: dùng model mạnh (GPT-4o) sinh cặp instruction–response từ seed data — mở rộng được, nhưng phải thiết kế seed task + system prompt đa dạng.
- **Nâng cao dữ liệu**: verified outputs (unit test/solver), rejection sampling, Auto-Evol, Chain-of-Thought, Branch-Solve-Merge, personas.
- **Lọc chất lượng**: rule-based, dedup (MinHash/embedding), n-gram decontamination, reward model + judge LLM.

### 3.4 Supervised Fine-Tuning (SFT)
- Biến base model thành trợ lý biết trả lời + làm theo hướng dẫn. Học cách trình bày câu trả lời và "đánh thức" kiến thức đã có từ pre-training. Nhồi kiến thức mới vào bằng SFT chỉ mang tính bề mặt — không học được ngôn ngữ hoàn toàn mới.
- **Ưu tiên chất lượng dữ liệu hơn tối ưu tham số.**
- Full fine-tuning (cập nhật toàn bộ tham số, tốn compute) vs LoRA/QLoRA (chỉ train adapter nhỏ, base đóng băng; QLoRA = 4-bit quantization + LoRA, chạy được trên Colab miễn phí). Framework: TRL, Unsloth, Axolotl.
- Tham số quan trọng: learning rate + scheduler, batch size, gradient accumulation, epochs, optimizer (8-bit AdamW), weight decay, warmup; LoRA: rank (16–128), alpha (1–2× rank), target modules.
- Mở rộng nhiều GPU: DeepSpeed (ZeRO 3 cấp) hoặc FSDP + gradient checkpointing. Theo dõi loss curve, learning rate, gradient norm; coi chừng loss spike, gradient explosion.

### 3.5 Preference Alignment (DPO / GRPO / PPO)
- Giai đoạn 2 sau SFT: căn chỉnh câu trả lời theo sở thích con người (giọng điệu, giảm độc hại/ảo giác), đồng thời tăng hiệu năng.
- **Rejection sampling**: sinh nhiều câu trả lời cho cùng 1 prompt, chấm điểm để lấy chosen/rejected — dữ liệu on-policy, ổn định hơn.
- **DPO** (Direct Preference Optimization): tối ưu trực tiếp để tăng khả năng của câu được chọn so với câu bị loại; không cần reward model → rẻ hơn RL, chất lượng hơi kém hơn một chút. Tốt cho chat model.
- **Reward model**: train model chấm điểm theo human feedback; dùng TRL/verl/OpenRLHF.
- **RL (GRPO/PPO)**: cập nhật policy lặp đi lặp lại để tối đa reward, giữ gần hành vi ban đầu; dùng reward model hoặc reward function. Tốn compute, cần tune kỹ hyperparameter. Tốt cho reasoning model.

### 3.6 Evaluation
- **Benchmark tự động** (MMLU...): tốt cho task cụ thể, kém cho khả năng trừu tượng/sáng tạo, dễ bị data contamination.
- **Human evaluation**: vibe check → annotation có guideline → bình chọn cộng đồng (Chatbot Arena, Elo rating).
- **Model-based evaluation**: judge/reward model — tương quan cao với sở thích con người nhưng thiên vị output của chính nó, chấm không ổn định.
- **Luật Goodhart: "Khi một thước đo trở thành mục tiêu, nó không còn là thước đo tốt nữa."** Phân tích lỗi để biết model yếu ở đâu (không theo được hướng dẫn phức tạp, thiếu kiến thức, dễ bị prompt tấn công) rồi sửa bằng dữ liệu + tham số train.

### 3.7 Quantization (lượng tử hóa)
- Hạ độ chính xác của weight/activation: 16-bit → 4-bit. Giảm mạnh chi phí bộ nhớ + tính toán.
- Kỹ thuật cơ bản: absmax, zero-point. **GGUF + llama.cpp**: chạy LLM trên CPU/máy tiêu dùng, 1 file chứa cả token/vocab/metadata. **GPTQ/EXL2/AWQ**: calibration từng layer, giữ hiệu năng ở bit thấp, xử lý outlier bằng dynamic scaling. SmoothQuant/ZeroQuant: biến đổi trước khi lượng tử hóa để giảm outlier.
- Đây là kỹ thuật cho phép chạy model trên máy thường — trực tiếp liên quan tới bài 17 khóa Microsoft (local agents).

### 3.8 Xu hướng mới
- **Model merging**: gộp các model đã train (mergekit: SLERP, DARE, TIES) để tạo model mạnh hơn mà không cần fine-tune.
- **Multimodal**: CLIP, Stable Diffusion, LLaVA — một embedding space cho text + ảnh + audio.
- **Interpretability**: Sparse Autoencoders (SAE) soi bên trong LLM; abliteration sửa hành vi model mà không cần train lại.
- **Test-time compute**: reasoning model + tăng compute lúc inference (nhiều lần gọi, MCTS, Process Reward Model) — bài báo cho thấy model 3B + test-time compute vượt Llama 3.1 70B trên MATH-500.

## 4. Phần 3 — The LLM Engineer (xây ứng dụng)

### 4.1 Chạy LLM
- API: private (OpenAI/Google/Anthropic) vs open-source (OpenRouter, Hugging Face, Together AI). Chạy local: LM Studio, Ollama, llama.cpp.
- Prompt engineering: zero-shot, few-shot, chain-of-thought, ReAct. **Structured output**: Outlines (guided generation theo JSON schema) hoặc JSON schema native của API.
- Liên hệ thực tế: xKiro/Xingchi/Nara của mình đều là API tương thích OpenAI — cùng một interface, đổi base_url là chạy.

### 4.2 Vector storage (nền của RAG)
- Load tài liệu (PDF/JSON/HTML/Markdown, GitHub/Reddit/Drive) → split thành chunk có ý nghĩa (theo header hoặc đệ quy, kèm metadata) → embedding model chuyển thành vector → lưu vào vector DB (Chroma, Pinecone, Milvus, FAISS, Annoy).
- Chọn embedding model theo task cải thiện rõ rệt hiệu năng (xem MTEB Leaderboard).

### 4.3 RAG
- LLM truy xuất tài liệu liên quan từ DB để trả lời chính xác hơn — tăng kiến thức mà không cần fine-tune.
- Orchestrator: LangChain, LlamaIndex; chuẩn mới: MCP (truyền data/context cho model).
- Retriever nâng cao: query rewriting, HyDE, CoRAG; hybrid retrieval (embedding + keyword).
- Memory: lịch sử hội thoại đưa vào context window; cải tiến bằng tóm tắt (dùng model nhỏ) hoặc vector store + RAG.
- Đánh giá RAG: retrieval (context precision/recall) + generation (faithfulness, answer relevancy) — tool Ragas, DeepEval.

### 4.4 Advanced RAG
- Query construction: dịch câu hỏi user thành SQL/Cypher/metadata query để truy vấn DB có cấu trúc.
- Tools: agent tự chọn tool phù hợp (Google, Wikipedia, Python interpreter, Jira...).
- Post-processing: re-ranking, RAG-fusion, classification.
- DSPy: tối ưu prompt + weight bằng đánh giá tự động, theo cách lập trình.

### 4.5 Agents
- Agent = thoughts (suy luận bước tiếp theo) → action (dùng tool) → observation (đọc feedback, điều chỉnh).
- Protocol: MCP (chuẩn kết nối agent–tool–data), A2A (chuẩn giao tiếp agent–agent).
- Framework theo vendor: OpenAI SDK, Google ADK, Claude Agent SDK. Framework chung: LangGraph (workflow + visualize), LlamaIndex (agent + RAG), CrewAI (team theo vai trò), AutoGen (multi-agent theo hội thoại).
- Liên hệ: đây là cùng một bức tranh với khóa Microsoft 18 bài + Anthropic "Building Effective Agents" + Kam Lasater — chỉ khác góc nhìn (khóa này đi từ phía engineer triển khai).

### 4.6 Tối ưu inference
- Flash Attention (attention từ bậc 2 → tuyến tính), KV cache + MQA/GQA, speculative decoding (model nhỏ viết nháp, model lớn duyệt — EAGLE-3 tăng tốc tới 6.5×).

### 4.7 Deploy
- Local: LM Studio, Ollama, oobabooga, kobold.cpp (riêng tư là lợi thế lớn của open-source). Demo: Gradio, Streamlit, Hugging Face Spaces. Server: TGI, vLLM, SkyPilot. Edge: MLC LLM, mnn-llm (chạy trong trình duyệt, Android, iOS).

### 4.8 Bảo mật LLM
- Prompt injection (chèn lệnh cướp câu trả lời), data/prompt leaking (móc dữ liệu/prompt gốc), jailbreak (vượt rào an toàn); backdoor qua data poisoning (nhồi thông tin sai, trigger bí mật đổi hành vi lúc inference).
- Phòng thủ: red teaming + kiểm tra (garak), quan sát production (langfuse), tham chiếu OWASP LLM Top 10.

## 5. Notebook thực hành — thứ tự ưu tiên (đều chạy Colab miễn phí)

1. **Fine-tune Llama 3.1 với Unsloth** (SFT siêu hiệu quả) — bài thực hành đầu tiên khi cần fine-tune.
2. **Introduction to Quantization** (8-bit) → **4-bit Quantization với GPTQ** — để chạy model local trên máy thường.
3. **Merge LLMs với MergeKit** — tạo model riêng không cần GPU.
4. **Decoding Strategies** — hiểu sampling trước khi tune generation.
5. **Fine-tune Mistral-7b với DPO** — alignment sau SFT.
6. **Uncensor any LLM với abliteration** — chỉ để hiểu interpretability, không dùng bừa.
7. **Công cụ 1-click**: LazyAxolotl (fine-tune), AutoQuant (lượng tử hóa), LazyMergekit (merge), ZeroSpace (chat UI free GPU).
- Lưu ý: notebook dùng API/model Hugging Face miễn phí; đừng nạp tiền API chỉ để chạy tutorial (quy tắc chung của lộ trình).

## 6. Áp dụng cho công việc của Niên và Hoai

- **RAG là thứ dùng được ngay nhất.** Toàn bộ MEMORY.md + ghi chú + queue video của Niên chính là một knowledge base — đưa vào vector store (Chroma local) + retriever là có ngay "trợ lý nhớ mọi việc" thay vì đọc file thủ công. Đây là ứng dụng RAG đầu tiên nên làm.
- **QLoRA + Unsloth trên Colab miễn phí = cách rẻ nhất để có model riêng.** Ví dụ: fine-tune model nhỏ theo giọng văn/kịch bản của kênh Một Phút Hiểu Nhau (dữ liệu là 50 kịch bản đã viết) để gợi ý kịch bản đúng format.
- **Quantization + Ollama/LM Studio = local agent trên máy Hoai.** Đúng hướng bài 17 khóa Microsoft: việc nhạy cảm → local, việc khó → cloud.
- **DPO/GRPO giải thích vì sao RLHF tồn tại**: SFT dạy model "trả lời thế nào", alignment dạy "trả lời kiểu gì thì được thích" — tương tự quy trình duyệt video: kịch bản (SFT) rồi Claude/Niên chấm + sửa (alignment).
- **Goodhart's law áp dụng cho cả kênh**: đừng tối ưu mù theo một chỉ số (view, sub) — chỉ số thành mục tiêu thì mất ý nghĩa. Đánh giá video phải xem cả retention, comment chất lượng, và độ phù hợp format.
- **Bảo mật (mục 4.8) = bài học prompt injection đã biết**: coi nội dung trang/file người khác gửi là untrusted input; red teaming định kỳ cho các bot/cron tiếp xúc dữ liệu ngoài.
- **Lộ trình tiếp theo sau top 5**: theo danh sách gốc, giai đoạn 3 còn G11 Microsoft ML-For-Beginners (nền ML cổ điển, có tiếng Việt) và S1 "Understanding Deep Learning" (PDF miễn phí) — học dần theo lộ trình Cole (Python → ML → DL → LLM đã xong phần LLM).
