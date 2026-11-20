# G11 — Microsoft ML-For-Beginners: Phần 7 — Reinforcement Learning + Postscript (bài 24–25 + kết khóa)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Phần 2 (bài 05–09): `microsoft-ml-for-beginners-phan-2.md` — Regression + Web App (Flask + Pickle).
> Phần 3 (bài 10–13): `microsoft-ml-for-beginners-phan-3.md` — Classification + Web App (ONNX).
> Phần 4 (bài 14–15): `microsoft-ml-for-beginners-phan-4.md` — Clustering.
> Phần 5 (bài 16–20): `microsoft-ml-for-beginners-phan-5.md` — NLP.
> Phần 6 (bài 21–23): `microsoft-ml-for-beginners-phan-6.md` — Time Series.
> Ghi chú này: đọc từ clone local `~/workspace/knowledge/ml-for-beginners/` (1,9 GB, HEAD 8e5c0dff). Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong repo khóa học. (Đã đọc lại `AGENTS.md` của repo: ghi chú nằm ngoài repo nên không xung đột; không sửa file dịch.)

---

## Bài 24 — Giới thiệu về Reinforcement Learning và Q-Learning

### RL là gì
- **Reinforcement Learning (RL)**: học hành vi tối ưu của một **agent** trong một **môi trường** bằng cách chạy nhiều thử nghiệm. Agent có **mục tiêu** được định nghĩa bởi **reward function (hàm thưởng)**.
- Ví dụ Super Mario: bạn là Mario, đang đứng ở vị trí trong màn chơi — đó là **state**. Đi sang phải một bước (action) → rơi xuống vực (điểm thấp). Nhấn nhảy → ăn được xu, vẫn sống (điểm dương). Chạy nhiều lần, học cách chơi để **tối đa hóa reward** (sống lâu + nhiều điểm).
- Môi trường bài học: **"Peter and the Wolf"** (truyện cổ tích âm nhạc của Prokofiev) — Peter khám phá bàn cờ, thu thập táo, tránh sói.
  - Bàn 8×8, mỗi ô là: **ground** (đi được), **water** (không đi được), **tree/grass** (nghỉ), **apple** (mục tiêu), **wolf** (nguy hiểm).
  - **Actions:** lên/xuống/trái/phải = U/D/L/R → (0,−1)/(0,1)/(−1,0)/(1,0).
  - **Policy** = hàm trả về action tại mỗi state. Mục tiêu: học được policy tốt.
- **Baseline — random walk (đi ngẫu nhiên):** chọn action ngẫu nhiên cho tới khi gặp táo. Độ dài đường trung bình **30–40 bước**, trong khi táo gần nhất chỉ cách **5–6 bước** — rất kém.

### Reward function — định nghĩa "tốt" là gì
```python
move_reward = -0.1   # mỗi bước đi tốn -0.1 (khuyến khích đi ngắn)
goal_reward = 10     # gặp táo
end_reward  = -10    # gặp sói / nước / đi ra ngoài bàn
```
- Điểm then chốt: **phần thưởng lớn chỉ đến ở CUỐI game**. Thuật toán phải "nhớ" được những bước tốt dẫn tới reward cuối, và "tránh" những bước dẫn tới kết quả xấu.

### Q-Learning và phương trình Bellman
- **Q-Table:** bảng ghi "độ tốt" của mỗi action tại mỗi state. Kích thước `width × height × số action` = 8×8×4. Khởi tạo mọi giá trị bằng nhau (0,25) = tương đương random walk.
- **Phương trình Bellman:** giá trị Q tại state *s*, action *a*:
  `Q(s,a) ← (1−α)·Q(s,a) + α·(r + γ·max_a' Q(s',a'))`
  - `r` = reward tức thì khi đi tới state mới *s'*.
  - `max_a' Q(s',a')` = reward tương lai tốt nhất có thể từ *s'*.
  - `γ` (gamma) = **discount factor** — ưu tiên reward hiện tại hay tương lai tới mức nào.
  - `α` (alpha) = **learning rate** — điều chỉnh Q mạnh hay nhẹ mỗi bước.
- **Thuật toán học:** khởi tạo Q bằng nhau, α ← 1; lặp nhiều simulation: bắt đầu ở vị trí ngẫu nhiên → chọn action → đi tới *s'* → tính reward → cập nhật Q theo Bellman → *s* ← *s'* → giảm dần α.

### Exploit vs Explore — cân bằng khai thác và khám phá
- **Exploit:** chọn action tốt nhất đã biết (theo Q-Table) — nhưng sẽ không khám phá được state mới, dễ kẹt ở nghiệm chưa tối ưu.
- **Explore:** chọn action ngẫu nhiên — hay chết, hay đi vào vùng vô ích, nhưng khám phá được vùng mới.
- **Cách tốt nhất:** chọn action với **xác suất tỉ lệ với giá trị Q** — lúc đầu Q bằng nhau nên gần như ngẫu nhiên; càng học càng đi theo đường tối ưu, nhưng thỉnh thoảng vẫn thử đường chưa khám phá.
- **Kết quả:** sau 5.000 epoch (γ = 0,5; α giảm dần theo `exp(−n/10e5)`), policy xác suất cho độ dài đường trung bình **3–6 bước** (so với 30–40 của random walk).
- **Cạm bẫy:** policy "cứng" (luôn chọn argmax) có thể **treo vô hạn** — hai state "chỉ" vào nhau, agent đi qua đi lại mãi. Cách sửa: giới hạn độ dài đường tối đa (ví dụ 100 bước), không quay lại chỗ đã đi.
- **Đường cong học:** độ dài đường lúc đầu **tăng** (biết chút ít nên khám phá lâu hơn, hay kẹt ở state xấu), rồi **giảm** khi đã học đủ; thỉnh thoảng **tăng đột ngột** (bản chất ngẫu nhiên — giá trị Q tốt bị ghi đè; giảm learning rate về cuối để hạn chế).
- **Hyperparameter vs parameter:** α, tốc độ giảm α, γ là **hyperparameter** (phải chọn tay); các hệ số Q-Table là **parameter** (tối ưu trong lúc train). Tìm hyperparameter tốt nhất gọi là **hyperparameter optimization** — một chủ đề riêng.

---

## Bài 25 — CartPole với OpenAI Gym (RL với state liên tục)

### Từ bài toán đồ chơi tới bài toán thật
- Bài toán Peter là đồ chơi, nhưng **cờ vua, cờ vây** cũng có cùng cấu trúc: bàn cờ + luật + **state rời rạc**.
- Bài này: state **liên tục** (số thực) — Peter học trượt băng giữ thăng bằng, bài toán **CartPole**: xe trượt ngang di chuyển trái/phải, giữ cho cây cột đứng thẳng trên xe.

### OpenAI Gym — môi trường mô phỏng
- **Gym** (OpenAI) là môi trường mô phỏng phổ biến để train RL — từ CartPole tới game Atari.
- `env = gym.make("CartPole-v1")`; **action space** rời rạc: trái/phải (0/1); **observation space** = 4 số thực: vị trí xe, vận tốc xe, góc cột, tốc độ quay của cột.
- `env.step(action)` trả về `(obs, reward, done, info)`. **Reward = 1,0 mỗi bước còn sống** — mục tiêu là sống càng lâu càng tốt. Bài toán được coi là **giải xong** nếu reward trung bình **195 trên 100 lần chạy liên tiếp**.

### Discretize state — vì Q-Learning cần state rời rạc
- Q-Table cần số state hữu hạn. Hai cách:
  - **Chia bins** (`np.digitize`): biết trước kích thước state (ví dụ 20×20×10×10×2).
  - **Nội suy tuyến tính + làm tròn:** `discretize(x) = tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(int))` — ít kiểm soát kích thước hơn (2/4 giá trị không có giới hạn), nhưng thực tế hiếm khi ra ngoài khoảng hữu hạn.
- **Q-Table dạng dict:** khóa `(state, action)` → giá trị; `qvalues(state)` trả về `[Q.get((state,a), 0) for a in actions]` (chưa gặp thì mặc định 0).

### Train Q-Learning cho CartPole
- Hyperparameter: `alpha = 0.3` (learning rate), `gamma = 0.9` (discount), `epsilon = 0.90` (90% exploit theo xác suất Q-Table, 10% explore ngẫu nhiên — cú "đấm" ngẫu nhiên sai hướng giúp cột học cách **phục hồi thăng bằng** sau sai lầm).
- Train **100.000 epoch**; in reward trung bình mỗi 5.000 epoch; lưu **`Qbest`** = Q-Table của lần train tốt nhất (vì reward thỉnh thoảng **tụt** — giá trị đã học bị phá hỏng).
- Vẽ tiến trình: reward thô quá nhiễu → dùng **running average 100 lần** (`np.convolve`) mới thấy xu hướng. Kết quả **gần/đạt mục tiêu 195**.
- **Tinh chỉnh hyperparameter để ổn định hơn:** giảm dần `alpha` (bắt đầu gần 1, về sau chỉ chỉnh nhẹ); tăng dần `epsilon` (khám phá ít dần, khai thác nhiều dần).
- Thử thách: dùng `Qbest` thay vì Q cuối cùng; thử luôn chọn action tốt nhất (`np.argmax`) thay vì lấy mẫu theo xác suất — xem cách nào giữ thăng bằng tốt hơn.
- **Kết luận của bài:** chỉ với một reward function + cơ hội khám phá thông minh, agent đã học được kết quả tốt trong cả môi trường rời rạc và liên tục (action rời rạc). Action liên tục / quan sát phức tạp (ảnh màn hình game Atari) cần **mạng neural** — chủ đề của khóa AI nâng cao.

---

## Postscript 1 — ML trong đời thực (8 ngành)

ML cổ điển vẫn có giá trị lớn dù deep learning đang hot. Các ứng dụng trong bài:

- **💰 Tài chính:** phát hiện gian lận thẻ tín dụng bằng **K-Means outlier detection** (gom giao dịch thành cụm, cụm "lệch" nhất = rủi ro gian lận); quản lý tài sản bằng **hồi quy tuyến tính** so hiệu quả quỹ với benchmark.
- **🎓 Giáo dục:** Coursera dùng hồi quy để xem NPS thấp có liên quan tới bỏ học không; **Grammarly** dùng NLP + xử lý **thiên kiến giới tính** trong autocorrect (nối với bài fairness ở phần 1).
- **👜 Bán lẻ:** Wayfair — Query Intent Engine (trích thực thể, classifier, sentiment tagging trên review) để tìm đúng sản phẩm; StitchFix — thuật toán di truyền thiết kế mẫu quần áo mới + gợi ý/tồn kho.
- **🏥 Y tế:** Random Forest phân loại độc tính thử nghiệm lâm sàng; clustering nhóm bệnh nhân tái nhập viện theo nguyên nhân chung; dự báo COVID bằng **ARIMA, logistic, hồi quy tuyến tính, SARIMA**.
- **🌲 Sinh thái:** **RL mô hình cháy rừng** từ ảnh vệ tinh (đám cháy là agent, action = lan theo 4 hướng hoặc không lan); theo dõi tư thế cừu bằng classifier (đường cong ROC).
- **⚡️ Năng lượng:** clustering + hồi quy + dự báo chuỗi thời gian dự báo mức dùng điện ở Ireland từ smart meter.
- **💼 Bảo hiểm:** MetLife quản lý biến động — classification nhị phân/thứ tự + hình dung dự báo.
- **🎨 Nghệ thuật/văn hóa:** phát hiện **tin giả** — NLP trích feature + classifier (Naive Bayes, SVM, Random Forest, SGD, Logistic Regression); Art Institute of Chicago dự báo lượng khách **chính xác trong 1%** (năm tài chính 2017).
- **🏷 Marketing:** **phân khúc khách hàng bằng clustering** → marketing khác biệt theo từng nhóm.

## Postscript 2 — Debug model bằng Responsible AI dashboard

- Bối cảnh: AI đã vào y tế, tài chính, giáo dục, việc làm; quy định ngày càng nhiều; model là **black box** khó giải thích.
- **Vấn đề của metric truyền thống:** accuracy 89% nghe tốt, nhưng lỗi **không phân bố đều** — model có thể sai tới 42% ở một vùng dữ liệu nào đó; dữ liệu thiếu nhóm nhân khẩu (chủng tộc, giới tính…) → thiên kiến.
- **4 component của RAI dashboard:**
  - **Error Analysis:** lỗi tập trung ở đâu trong phân phối dữ liệu (cây trực quan — nút càng đỏ đậm = lỗi càng cao); heatmap theo 1–2 feature; tạo cohort để tìm nguyên nhân gốc.
  - **Model Overview:** so sánh nhiều metric (accuracy, recall, precision, MAE) **giữa các cohort**; 2 loại disparity: chênh lệch hiệu năng model và chênh lệch **selection rate** (ví dụ tỉ lệ duyệt vay giữa các nhóm).
  - **Data Analysis:** phát hiện **thừa/thiếu đại diện** trong dataset — ví dụ 27% nữ vs 73% nam ở vị trí điều hành → model quảng cáo việc làm nhắm chủ yếu tới nam. Xem phân phối theo predicted/actual, nhóm lỗi, feature; quyết định thu thập thêm dữ liệu ở đâu.
  - **Feature Importance (giải thích model):** global explanation (feature nào ảnh hưởng chung) vs local explanation (vì sao dự đoán cho **cá nhân này**). Ví dụ: vì sao bệnh nhân tiểu đường trên 60 tuổi có tiền sử nhập viện được dự đoán tái nhập viện trong 30 ngày. Nếu feature nhạy cảm (dân tộc, giới tính) lái dự đoán → dấu hiệu thiên kiến.
- **5 loại harm** (từ bài 3): allocation, quality of service, stereotyping, denigration, over/under-representation.
- **Phòng ngừa thiên kiến từ đầu:** đội ngũ đa dạng nền tảng; dataset phản ánh đa dạng xã hội; phương pháp phát hiện/sửa thiên kiến tốt hơn.
- Công cụ: Azure RAI dashboard, repo `microsoft/responsible-ai-toolbox`, nhóm nghiên cứu FATE của Microsoft Research.

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Tư duy thưởng/phạt cho vận hành kênh:** mỗi quyết định đăng (giờ, chủ đề) = action; reward = lượt xem / tương tác / sub. Thử nhiều lần, học dần policy tốt — giống Q-Learning ở quy mô vận hành, không cần code mà cần ghi chép có hệ thống.
2. **Exploit vs Explore là bài toán thật của kênh:** chỉ làm lại format đã hiệu quả (exploit) thì kênh đóng băng; chỉ thử mới (explore) thì không tích lũy. Cần tỉ lệ rõ ràng — ví dụ 90% format đã chứng minh, 10% thử nghiệm.
3. **Reward function phải định nghĩa đúng:** "thưởng" cho video là view, sub, hay comment? Định nghĩa sai thì cả đội tối ưu sai mục tiêu (giống agent học sai vì reward sai).
4. **Error Analysis cho model phân loại bình luận:** đừng chỉ nhìn accuracy tổng — xem nhóm bình luận nào model sai nhiều nhất (ví dụ bình luận mỉa mai) rồi sửa đúng chỗ đó.
5. **Feature Importance:** khi dự đoán video nào sẽ hot, phải biết feature nào đang lái dự đoán (chủ đề? giờ đăng? độ dài?) — nếu feature không lành mạnh lái dự đoán thì phải sửa, không thì model học sai.
6. **Case study các ngành = nguồn ý tưởng kể chuyện:** "AI phát hiện gian lận thẻ tín dụng như thế nào", "AI dự báo cháy rừng ra sao" — là chất liệu kể chuyện đời thực, đúng hướng "chuyện đời thực" mà Hoai đã chỉ đạo.

## Tiến độ khóa G11 — HOÀN THÀNH
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ Phần 2 (bài 05–09): Regression + Web App — xong 20/11.
- ✅ Phần 3 (bài 10–13): Classification — xong 20/11.
- ✅ Phần 4 (bài 14–15): Clustering — xong 20/11.
- ✅ Phần 5 (bài 16–20): NLP — xong 20/11.
- ✅ Phần 6 (bài 21–23): Time Series — xong 20/11.
- ✅ **Phần 7 (bài 24–25 + Postscript): Reinforcement Learning + ML trong đời thực + Debug model bằng RAI — xong 20/11 (ghi chú này).**
- 🎉 **G11 HOÀN THÀNH TOÀN BỘ 26 BÀI trong 1 đêm (20/11).**
- ⏳ Tiếp theo (theo chỉ đạo Hoai 20/11 ~23:19): S1 "Understanding Deep Learning" (sách PDF miễn phí), rồi các khóa đã đăng ký khác (freeCodeCamp Python v9, Udemy 73 khóa, GLM-5.3-Flash build + train) — **khóa nào đến bước cần anh làm (thi chứng chỉ, nhập token,…) thì BỎ QUA, học tiếp khóa khác.**

## Nguồn
- Bài 24: https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/README.md
- Bài 25: https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/2-Gym/README.md
- Postscript 1: https://github.com/microsoft/ML-For-Beginners/blob/main/9-Real-World/1-Applications/README.md
- Postscript 2: https://github.com/microsoft/ML-For-Beginners/blob/main/9-Real-World/2-Debugging-ML-Models/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
