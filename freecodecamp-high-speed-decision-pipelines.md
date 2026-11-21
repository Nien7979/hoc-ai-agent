# High-Speed Decision Pipelines: Jev and Prefill-Only Inference from Scratch

- Nguồn bài viết: https://www.freecodecamp.org/news/high-speed-decision-pipelines-jev-and-prefill-only-inference-from-scratch/
- Video khóa học: "Jev Course – Prefill-Only Inference & Decision Models from Scratch" — https://www.youtube.com/watch?v=aZglGvBPQGs (đăng 27/10/2026, dài ~1 giờ 25 phút theo danh sách chương; bài freeCodeCamp ghi "2-hour watch")
- Giảng viên: **Vivek Kalyanarangan**, tác giả sách *Quantization and Fast Inference* (Manning)
- Code chính thức của khóa học: https://github.com/vivekkalyanarangan30/how-jev-works
- Nguồn gốc ý tưởng: TypeSafe AI — https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Ngày học: 21/11/2026
- **Lưu ý về transcript:** KHÔNG lấy được transcript đầy đủ của video (youtubetotranscript.com trả lỗi 429; API phụ đề của YouTube trả về rỗng). Ghi chú này học từ: bài viết freeCodeCamp + toàn bộ code repo chính thức (đã đọc kỹ README và 3 script chính + `common.py`) + blog gốc của TypeSafe + dàn ý chương của video. Các chương về huấn luyện (GRPO, RLCD loss, các hàm scoring) chỉ tóm tắt theo dàn ý chương và bảng `extras/` trong README, chưa có lời giảng chi tiết từng câu — cần xem lại video ở các mốc thời gian đó để nghe đủ.

## 1. Bối cảnh: vì sao cần prefill-only inference

- **Vấn đề của LLM tự hồi quy (autoregressive):** khi chỉ cần mô hình ra một quyết định có cấu trúc (phân loại, routing, chấm điểm) thay vì viết cả đoạn văn, việc bắt mô hình sinh từng token một vừa chậm vừa tốn bộ nhớ. Sinh JSON 23 token = 23 lần forward pass, mà cấu trúc chỉ đúng khi "may mắn".
- **Jev là gì:** mô hình đầu tiên thuộc lớp **System One Models** của **TypeSafe AI** (sáng lập bởi Diogo Almeida, cựu nghiên cứu viên OpenAI, người góp phần xây dựng phương pháp đằng sau ChatGPT), ra mắt 15/9/2026. Tên "Jev" đặt theo nhà kinh tế William Stanley Jevons; "System One" lấy cảm hứng từ Tư duy nhanh (System 1) trong *Thinking, Fast and Slow* của Daniel Kahneman.
- **Định nghĩa ngắn gọn:** Jev là một "function call có trí tuệ frontier" — **state không cấu trúc vào, quyết định có kiểu (typed) kèm xác suất ra**. Bỏ hẳn việc sinh chuỗi ký tự để đổi lấy tốc độ và độ tin cậy.
- **Vì bỏ sinh chữ mà được "siêu năng lực":**
  - **Không hallucinate, không lỗi kiểu (type errors):** đáp án luôn nằm trong tập lựa chọn định trước, cấu trúc được đảm bảo bằng toán học, không cần parse + validate chuỗi.
  - **Tốc độ:** TypeSafe công bố 70–500ms mỗi lần gọi, nhanh hơn LLM frontier 40–200 lần cho cùng mức trí tuệ trên các truy vấn dạng System One (LLM mất 3–329 giây end-to-end).
  - **Chi phí:** token đầu vào $0.042 / triệu token; **token đầu ra MIỄN PHÍ** (rẻ đến mức không đáng đo) vì không có vòng lặp decode.
  - **Xác suất đã hiệu chuẩn (calibrated):** mọi đáp án đi kèm độ tin cậy trung thực — 0.7 nghĩa là đúng 0.7.
- **Dùng để làm gì:** AI-Powered Workflows / "smart if-statements" — phân loại, routing, chấm điểm, trích xuất, rẽ nhánh trong code thường; map-reduce trên dữ liệu lớn; ứng dụng real-time (100ms); chấm điểm/kiểm tra/guardrail cho prompt và output của LLM khác.

## 2. Prefill vs. Decode — hai giai đoạn suy luận của LLM

- **Prefill:** xử lý toàn bộ prompt trong **một** lần forward pass, tính ra key-value (KV) cache cho mọi token trong prompt. Đây là giai đoạn "đọc và hiểu" đầu vào.
- **Decode:** sinh từng token đáp án, mỗi token cần **một** forward pass mới (dùng lại KV cache để không tính lại prompt). Đây là giai đoạn "viết" — chậm, tuần tự, và là nút thắt cổ chai.
- **Prefill-only inference:** chỉ làm giai đoạn prefill, rồi **đọc thẳng quyết định từ logits tại vị trí quyết định** (decision position — điểm mà mô hình sắp viết đáp án), hoàn toàn không bước vào vòng lặp decode.
- **Hệ quả:** độ trễ gần như chỉ bằng một lần forward pass trên prompt; thêm câu hỏi chỉ tốn thêm prefill, không tốn thêm độ trễ decode; token đầu ra bằng 0 nên không mất chi phí sinh chữ.

## 3. Ba giao diện (interfaces) của Jev: Choice, Score, Boolean

- **Choice (lựa chọn):** tối đa 255 lựa chọn; đáp án là phân phối xác suất trên các lựa chọn (softmax trên logits của các token đại diện cho lựa chọn).
- **Score (chấm điểm):** 2–10 mức có thứ tự; đáp án là **trung bình có trọng số theo xác suất** của các chỉ số mức (probability-weighted mean of level indices).
- **Boolean / Noul:** một xác suất duy nhất (ví dụ "có/không"), không kèm confidence riêng.
- **Chạy song song:** nhiều câu hỏi chạy cùng lúc trên một state chung; câu hỏi thêm chỉ tốn prefill, không tốn độ trễ.
- **Điểm cần nhớ:** cả ba giao diện đều trả về **số có kiểu + xác suất**, không trả về chuỗi — phần mềm dùng trực tiếp được, không cần parse.

## 4. Baseline: Vanilla decoding — cách làm thông thường và vì sao nó tệ

- **Cách làm (script `1_vanilla_decoding.py`):** nhắc mô hình (prompt) rồi để nó viết JSON từng token một cho đến khi ra JSON hợp lệ. Ví dụ trong repo: schema có 3 trường (`priority`, `decision`, `escalate`) — mô hình viết 23 token, tức 23 lần forward pass.
- **"Độ tin cậy" giả:** script còn tính điểm bằng cách nhân xác suất của từng token đã viết ra giá trị — con số trông giống confidence nhưng **không phải xác suất quyết định**, vì 4 lý do:
  1. Nó chuẩn hóa trên **toàn bộ từ vựng**, không phải trên các giá trị được phép.
  2. Nó nhân xác suất mà **không chuẩn hóa độ dài**, nên giá trị dài trông ít khả năng hơn giá trị ngắn.
  3. Nó phụ thuộc vào **các trường đã viết trước đó**, nên thứ tự trường làm thay đổi kết quả.
  4. Các lựa chọn thay thế **chưa bao giờ được chấm điểm** — mô hình chưa từng so sánh chúng với nhau.
- **Kết luận:** mọi bước sau trong khóa học đều nhằm **xóa bỏ vòng lặp decode** và sửa cả 4 lỗi trên.

## 5. Kiến trúc encoder-only — cách Laya (Jev mở) làm

- **Ý tưởng (script `2_pure_encoding.py`):** một sequence **cho mỗi câu hỏi**, state đặt ở cuối, một token `[MASK]` cắm trước mỗi lựa chọn:
  ```
  [CLS] instructions [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
                           ^^^^^^        ^^^^^^
  ```
- **Cách chấm điểm:** attention **hai chiều** — mọi token thấy mọi token khác. Đọc hidden state tại vị trí các token marker, tính dot product với vector câu hỏi (mean pooling trên span của câu hỏi), rồi softmax. Ví dụ trong repo dùng ModernBERT-base (`answerdotai/ModernBERT-base`).
- **Ưu điểm:** rất ít "plumbing" — không cần causal mask, không cần segment id, không cần position giả. State được đọc **khi đã thấy câu hỏi**, các lựa chọn được so sánh trực tiếp với nhau.
- **Nhược điểm chí mạng:** toàn bộ state phải encode lại cho **mỗi** câu hỏi, và **không cache được gì**, vì đổi câu hỏi là đổi mọi vector state.
- **Hai lưu ý quan trọng:**
  - `[MASK]` ở đây là một **token placeholder** (mượn từ pretraining kiểu BERT, dùng làm chỗ để đọc ra), **không phải attention mask** — khác hẳn "mask" trong file 3.
  - Encoder MLM **không có next-token head** trỏ vào vị trí đáp án nên không mượn được trick đọc logits như decoder; repo dùng dot product không tham số nên "số chỉ đúng về hình dạng". Laya thật dùng một MLP đã huấn luyện để chấm từng marker và không cần vector câu hỏi.

## 6. Kiến trúc decoder-only / prefill-only — cách Jev làm

- **Ý tưởng (script `3_pure_decoding.py`):** một sequence duy nhất cho tất cả:
  ```
  [ the state ][ question 0 + its options ][ question 1 + its options ]
    seen by all     sees state + itself        sees state + itself
                    never question 1           never question 0
  ```
- **Custom causal masking:** một attention mask tự viết để state được mọi câu hỏi thấy, mỗi câu hỏi chỉ thấy state + chính nó, **không bao giờ thấy câu hỏi khác**. Nhờ đó nhiều câu hỏi được "đóng gói" trong một lần pass mà không làm nhiễu lẫn nhau.
- **Decision position:** mỗi câu hỏi kết thúc tại một "vị trí quyết định" — điểm mà mô hình sắp viết đáp án. Lấy **vocabulary logits tại vị trí đó**, chỉ giữ lại các hàng ứng với lựa chọn được phép, rồi softmax trên các hàng đó.
- **Vì sao không cần huấn luyện:** chính **output head của mô hình** làm nhiệm vụ chấm điểm, nên không có gì phải train thêm, và không thể sinh ra ngoài tập lựa chọn.
- **KV-cache tái sử dụng:** state chỉ đọc **một lần**, cache của nó dùng lại được cho mọi câu hỏi, vì không có gì trong state phụ thuộc vào phần đứng sau nó.
- **Ví dụ kết quả trong repo** (ticket khách hàng bị tính tiền 2 lần):
  ```
  Is the customer asking for money back?   {'no': 0.052, 'yes': 0.948}
  Which team owns this?                    {'billing': 0.975, 'technical': 0.025}
  ```
- **Giới hạn first-token (multi-token collision):** chỉ đọc **token đầu tiên** của mỗi lựa chọn, nên hai lựa chọn bắt đầu giống nhau — ví dụ `FRAUD` và `FRAUD_REVIEW` — không phân biệt được. Hàm `pack()` trong repo **từ chối** trường hợp này thay vì trả về kết quả hòa. Cách xử lý đầy đủ là chấm **toàn bộ token** của lựa chọn (xem mục 8 về trie).

## 7. Custom causal masking — viết mask attention cấp thấp

- **Hàm `attention_mask()` trong `common.py`:** xây ma trận `visible` (ai được thấy ai) rồi biến thành mask bằng cách điền giá trị rất âm (`torch.finfo(dtype).min`) vào các vị trí không được thấy:
  - `causal`: token chỉ thấy các token đứng trước hoặc bằng nó (ma trận tam giác dưới).
  - `key_is_state`: mọi query đều thấy các token thuộc state (segment 0).
  - `same_question`: query chỉ thấy các key cùng segment câu hỏi với nó.
  - Công thức decoder thuần: `visible = (key_is_state | same_question) & causal`.
- **Hàm `question_mask()` (dùng với KV-cache):** câu hỏi thấy toàn bộ state đã cache + chính nó theo causal, không thấy câu hỏi khác. Mask có chiều rộng `state_length + width` vì phần state đã nằm trong cache.
- **`position_ids` — chi tiết dễ sai:** token trong state giữ vị trí gốc; mỗi câu hỏi **reset position về cuối state** (`state_length + i - segments.index(seg)`) để vị trí không bị lệch khi nhiều câu hỏi nối tiếp nhau trong một sequence.
- **Biến thể hybrid / prefix-LM (`extras/hybrid_encoding.py`):** state dùng attention hai chiều, câu hỏi vẫn causal — được cả hai lợi ích (state đọc có ngữ cảnh đầy đủ + cache vẫn hợp lệ), nhưng "không miễn phí" (có đánh đổi, xem trong file extras).

## 8. KV-cache reuse và chấm điểm bằng trie

- **Prefill một lần, dùng nhiều lần:** hàm `prefill()` đẩy state qua mô hình một lần và giữ lại `past_key_values`. Các câu hỏi sau dùng `copy.deepcopy(cache)` để chạy tiếp từ cache đó thay vì tính lại state. Khi nhiều quyết định dùng chung một system context lặp lại, phần tính toán lặp lại gần như bằng 0.
- **Nhân bản cache cho nhiều nhánh:** `repeat_cache()` dùng `batch_repeat_interleave` để nhân cache thành nhiều bản sao cho các nhánh tính toán song song.
- **Trie scoring (`build_trie` / `tree_scores` / `decide` trong `common.py`):** để chấm đúng cả lựa chọn nhiều token, repo gom mọi lựa chọn của mọi trường thành một **cây trie theo token chung** (các lựa chọn cùng tiền tố chia sẻ nút). Chỉ cần **2 forward pass** cho toàn bộ schema: một pass prefill context để lấy cache, một pass chạy qua các nút của trie; điểm của mỗi lựa chọn = trung bình log-prob của các token trên đường đi của nó. Hàm `decide()` trả về đúng "bề mặt Jev": mọi câu hỏi được trả lời từ một prefill, dưới dạng lựa chọn có kiểu kèm xác suất.
- **Chuẩn hóa lựa chọn:** `compile_field()` tách phần tiền tố chung của các lựa chọn (ví dụ `FRAUD` trong `FRAUD`/`FRAUD_REVIEW`) ra khỏi phần cần chấm, và báo `collision` khi hai lựa chọn có cùng token đầu tiên.

## 9. Huấn luyện: distillation, GRPO và RLCD

- **Teacher-student distillation:** dùng mô hình lớn (teacher) làm chuẩn để huấn luyện mô hình quyết định nhỏ hơn; mục tiêu là giữ độ chính xác trong khi giảm độ trễ.
- **RL với GRPO (Group Relative Policy Optimization):** dùng các quỹ đạo bị nhiễu (perturbed trajectories) để tạo nhóm mẫu so sánh, tính advantage theo nhóm thay vì cần hàm value riêng — dùng để tối ưu cả độ trễ và độ chính xác của quyết định.
- **RLCD — Reinforcement Learning for Calibrated Decisions:** phương pháp huấn luyện riêng của TypeSafe. Ý tưởng: huấn luyện bằng **proper scoring rules** (quy tắc chấm điểm "trung thực") để xác suất mô hình đưa ra là xác suất thật — 0.7 nghĩa là đúng 0.7.
  - **Log loss:** strictly proper scoring rule — chỉ đạt tối thiểu khi xác suất báo cáo là xác suất trung thực. Trong production, người ta thay output head bằng một readout nhỏ đã huấn luyện và fit nó theo log loss (xem `extras/train_the_head.py`).
  - **Spherical score** và **Ranked Probability Score (RPS):** các proper scoring rules khác được dùng để đánh giá/chấm điểm phân phối xác suất, đặc biệt với đáp án có thứ tự (như Score).
  - **Hàm loss RLCD cuối cùng:** kết hợp các scoring rules trên thành một mục tiêu huấn luyện duy nhất (chương 1:21:38 trong video).
  - **Vòng lặp RLCD thực của Laya** (`extras/rlcd_training.py`): reward theo proper scoring rule + GRPO advantage + fit nhiệt độ (temperature fit) để hiệu chuẩn.
- **Công thức confidence (chưa được TypeSafe công bố):** một bản tái tạo độc lập cho rằng với Choice, confidence = `(n·p_max − 1)/(n − 1)`. Đây là ước đoán có ghi nhãn, không phải tài liệu chính thức.
- **Điểm khác biệt cấu trúc cần nhớ:** repo này chấm mỗi lựa chọn **độc lập**, nên log-odds giữa hai lựa chọn **bất biến** khi thêm lựa chọn thứ ba. Jev thật thì các lựa chọn **tương tác** — một phép đo tái tạo thấy một lựa chọn không liên quan làm log-odds dịch từ +0.38 xuống +0.11.

## 10. Encoder vs. Decoder — so sánh và đánh đổi

- **Thứ hạng compute đảo chiều theo số câu hỏi trên mỗi tài liệu:** encoder đọc lại state cho mỗi câu hỏi; decoder đọc state một lần rồi cache. Điểm hòa vốn nằm gần **tỉ lệ tham số giữa hai mô hình**.
- **Encoder đọc state khi đã thấy câu hỏi; decoder thuần phải encode state "mù câu hỏi"** — vì đó là điều kiện tiên quyết để cache được. Prefix-LM có cả hai, nhưng "không ai ship" (xem `extras/hybrid_encoding.py` để biết vì sao không miễn phí).
- **Bidirectional attention lẽ ra giúp encoder phân loại tốt hơn trên mỗi tham số** — nhưng chưa được kiểm chứng công bằng; một phép so sánh công bằng cần cùng dữ liệu, cùng mục tiêu, cùng ngân sách tham số và cùng readout trên cả hai.
- **Cẩn thận khi so sánh số liệu trong repo:** file 2 chạy **base encoder**, file 3 chạy **instruction-tuned decoder**, nên mọi chênh lệch chất lượng zero-shot đều là do instruction tuning, **không phải kiến trúc**.
- **Trần context khác nhau:** Laya giới hạn 8k token, Qwen3 native 32k.
- **Domain fit quan trọng ngang pretraining:** checkpoint tiếng Anh của Laya chỉ đạt 0.362 trên chính benchmark typed-decisions của họ, nhưng lên 0.766 sau khi fine-tune trên đúng domain đó. ModernBERT-base chưa từng được huấn luyện để trả lời câu hỏi — hỏi qua chính MLM head của nó, nó vẫn trả lời `technical` ở mức 0.955 cho một ticket bị tính tiền 2 lần.

## 11. Dàn ý code PyTorch từ con số 0 (theo repo chính thức)

- **Chạy thử:**
  ```bash
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  .venv/bin/python 1_vanilla_decoding.py
  .venv/bin/python 2_pure_encoding.py
  .venv/bin/python 3_pure_decoding.py
  ```
- **Mô hình mặc định:** decoder `Qwen/Qwen3-0.6B` (cần full softmax attention), encoder `answerdotai/ModernBERT-base`. Đổi qua biến môi trường `MODEL_ID` / `ENCODER_ID`. `TINY=1` đổi sang mô hình ngẫu nhiên 2-layer — chạy ngay không cần tải, số vô nghĩa nhưng đúng hình dạng. `./verify.sh` chạy tất cả.
- **Các file chính:**
  - `1_vanilla_decoding.py` — baseline: sinh JSON từng token, 23 token = 23 forward pass; chứng minh p(path) không phải xác suất quyết định.
  - `2_pure_encoding.py` — encoder-only: `[CLS] instructions [SEP] [MASK] opt [MASK] opt [SEP] state [SEP]`; mean pooling câu hỏi + dot product tại vị trí marker.
  - `3_pure_decoding.py` — decoder-only: prefill state một lần → `pack()` câu hỏi + lựa chọn → `question_mask()` → `position_ids` reset → đọc logits tại decision position → softmax trên các token lựa chọn.
  - `common.py` — dùng chung: `load()`, `prefill()`, `pack()`, `attention_mask()`, `question_mask()`, `encode()`, `compile_field()`, `build_trie()`, `tree_scores()`, `decide()`, `repeat_cache()`.
  - `labelled_examples.py` — ví dụ có nhãn dùng trong demo (ticket #8812, thẻ 4412).
- **Dàn ý pipeline prefill-only (dạng giả mã):**
  ```
  cache, state_len = prefill(model, state)            # 1. đọc state 1 lần, giữ KV cache
  ids, segments, decisions, candidates = pack(state, questions)  # 2. đóng gói câu hỏi + lựa chọn
  mask = question_mask(segments, state_len)          # 3. mask: câu hỏi thấy state + chính nó
  positions = reset_position_ids(segments, state_len) # 4. reset vị trí mỗi câu hỏi
  logits = model(ids, mask, positions, past_key_values=cache)    # 5. MỘT forward pass
  probs = softmax(logits[decision_pos][option_token_ids])        # 6. đọc xác suất tại decision position
  ```
- **Các file nâng cao trong `extras/`:** `rlcd_training.py` (vòng lặp RLCD của Laya), `train_the_head.py` (readout đã huấn luyện), `score_dont_generate.py` (trick chấm điểm không cần huấn luyện), `candidate_tokens.py`, `first_token_collisions.py`, `hybrid_encoding.py`, `trie_ancestor_mask.py`, `choice_score_noul.py`, `score_depends_on_wording.py`, `calibration_and_ece.py` (reliability bins, ECE, fit nhiệt độ), `rlcd_objective.py`, `pointwise_vs_listwise.py`, `vs_laya.py`.

## 12. Dàn ý chương video (để xem lại theo mốc thời gian)

- 0:00:00 Introduction & Course Overview
- 0:01:12 What is Jev & LLM Prefill vs. Decode
- 0:03:26 Jev Interfaces: Choice, Score, and Boolean
- 0:05:30 Prerequisites & Hardware Requirements
- 0:07:10 The 3-Phase Roadmap
- 0:08:00 Baseline: Vanilla LLM Token-by-Token Decoding
- 0:13:00 Stepping Through KV Cache & Autoregressive Loops
- 0:21:24 Encoder-Only Architecture for Fast Decisions
- 0:25:26 Implementing Encoder-Based Jev in PyTorch
- 0:32:00 Question Spans, Mask Tokens & Mean Pooling
- 0:36:30 Dot Product Interactions, Linear Heads & Fine-Tuning Approaches
- 0:40:42 Decoder-Only Architecture & Context Caching
- 0:46:22 Custom Causal Attention Masking for Multi-Question Packing
- 0:52:16 Implementing Decoder-Based Jev with KV Cache Re-use
- 1:03:52 Extracting Decision Point Logits & Token Probabilities
- 1:09:12 Handling Multi-Token Collision Issues
- 1:11:15 Training Workflows & Teacher-Student Distillation
- 1:12:35 Reinforcement Learning with GRPO & Perturbed Trajectories
- 1:16:54 Log Loss, Spherical Scores & Ranked Probability Score (RPS)
- 1:21:38 Constructing the Final RLCD Loss Function
- 1:23:17 Encoder vs. Decoder: Trade-offs & Architecture Comparison

## 13. Điểm cần nhớ

- **Prefill-only = bỏ vòng lặp decode:** đọc quyết định từ logits tại decision position trong lần prefill duy nhất; không sinh token nào.
- **Ba giao diện Jev:** Choice (≤255 lựa chọn), Score (2–10 mức, trung bình có trọng số), Boolean/Noul (1 xác suất).
- **Encoder:** đơn giản, đọc state khi thấy câu hỏi, nhưng encode lại toàn bộ state cho mỗi câu hỏi, không cache được.
- **Decoder:** một sequence cho tất cả, custom causal mask tách các câu hỏi, state đọc một lần + KV-cache tái sử dụng.
- **Chỉ đọc first token của lựa chọn → va chạm multi-token** (`FRAUD` vs `FRAUD_REVIEW`); giải pháp đầy đủ là trie scoring với trung bình log-prob.
- **p(path) nhân xác suất token KHÔNG phải xác suất quyết định** — sai vì chuẩn hóa trên toàn vocab, không chuẩn hóa độ dài, phụ thuộc thứ tự trường, và chưa từng chấm các lựa chọn khác.
- **RLCD:** huấn luyện bằng proper scoring rules để xác suất trung thực; log loss là strictly proper; thay readout đã huấn luyện + fit nhiệt độ để hiệu chuẩn.
- **So sánh encoder/decoder phải công bằng:** cùng dữ liệu, mục tiêu, ngân sách tham số, readout — và nhớ file 2 dùng base encoder còn file 3 dùng instruction-tuned decoder.
- **Điểm hòa vốn compute** giữa encoder và decoder nằm gần tỉ lệ tham số của hai mô hình và phụ thuộc số câu hỏi trên mỗi tài liệu.
- **Domain fit quan trọng ngang pretraining:** Laya 0.362 zero-shot → 0.766 sau fine-tune đúng domain.
- **Áp dụng thực tế:** routing/phân loại/chấm điểm/guardrail trong code; khi nào cần sinh văn bản dài hoặc suy luận nhiều bước thì vẫn cần decode — prefill-only không thay thế LLM sinh chữ.

## 14. Tài liệu tham khảo

- Bài freeCodeCamp: https://www.freecodecamp.org/news/high-speed-decision-pipelines-jev-and-prefill-only-inference-from-scratch/
- Video YouTube: https://www.youtube.com/watch?v=aZglGvBPQGs
- Code repo chính thức: https://github.com/vivekkalyanarangan30/how-jev-works
- TypeSafe — Introducing System One Models & Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe API reference: https://docs.typesafe.ai/api
- Hume — Jev's Architecture Unmasked: https://archerhume.com/posts/jevs-architecture-unmasked
- Bản tái tạo mở: Laya (Jev mở, encoder), LLM2Jev (https://github.com/Yinsongxu/LLM2Jev), NanoJev, open-jev, kev
- Bài báo JevCascade (phân bổ compute cho prefill-only): https://arxiv.org/pdf/2610.07716
- Sách của giảng viên: Vivek Kalyanarangan, *Quantization and Fast Inference* (Manning)
