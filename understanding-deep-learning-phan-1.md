# S1 — Understanding Deep Learning (Simon J.D. Prince): Phần 1 — Chương 1–4

> Sách: **Understanding Deep Learning** — Simon J.D. Prince, MIT Press 2023, 544 trang, **PDF MIỄN PHÍ** (CC BY-NC-ND, open-access). Bản đang đọc: release v5.0.3 (08/02/2026), 541 trang, tải từ https://github.com/udlbook/udlbook/releases (file `UnderstandingDeepLearning_02_09_26_C.pdf`, 22 MB, lưu tại `~/workspace/knowledge/understanding-deep-learning/`).
> Sách có 21 chương + 68 notebook Python thực hành (https://udlbook.github.io/udlbook/). Mục tiêu của sách: hiểu **nguyên lý** đằng sau deep learning — không quá lý thuyết (không có chứng minh), cũng không quá thực hành (gần như không có code).
> Cấu trúc sách: chương 2–9 = pipeline học có giám sát; 10–13 = các kiến trúc (CNN, residual, transformer, GNN); 14–18 = học không giám sát / mô hình sinh; 19 = RL; 20 = vì sao DL hiệu quả; 21 = đạo đức.
> Ghi chú này: phần 1 (chương 1–4). Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong tài liệu gốc.

---

## Chương 1 — Introduction

### AI, ML, Deep Learning là gì
- **AI** = xây hệ thống mô phỏng hành vi thông minh (logic, tìm kiếm, suy luận xác suất…). **Machine learning** = nhánh của AI: học ra quyết định bằng cách **fit mô hình toán học vào dữ liệu quan sát**. **Deep neural network** = một loại model ML; **deep learning** = quá trình fit các model đó.
- ML chia thô thành 3 lĩnh vực: **supervised, unsupervised, reinforcement learning** — cả 3 hiện nay đều dựa vào deep learning.

### Supervised learning — học có giám sát
- Model định nghĩa **ánh xạ từ input → output**. Ví dụ:
  - **Regression** (hồi quy): giá nhà từ diện tích + số phòng (số liên tục); **multivariate regression**: điểm đóng băng + điểm sôi của phân tử.
  - **Binary classification**: review nhà hàng tích cực/tiêu cực (output = xác suất thuộc mỗi lớp).
  - **Multiclass classification**: thể loại nhạc từ file audio; vật thể trong ảnh (output = vector N xác suất).
- **Input đa dạng:** tabular (vector cố định, đổi thứ tự không đổi kết quả); text (độ dài thay đổi, **thứ tự quan trọng** — "my wife ate the chicken" ≠ "the chicken ate my wife"); audio (clip 10s = 441.000 số nguyên ở 44,1 kHz); ảnh (pixel RGB, có cấu trúc không gian); phân tử (số nguyên tử thay đổi + hình học).
- **Model = một họ các phương trình**; **tham số φ** chọn ra thành viên cụ thể của họ; **training** = tìm tham số mô tả tốt dữ liệu train. Các cặp input/output có nhãn đóng vai trò "thầy" → gọi là supervised learning.
- Deep net xử lý được input rất lớn, độ dài thay đổi, có cấu trúc nội bộ; output được số thực, nhiều số, hoặc xác suất các lớp.

### Structured outputs — output có cấu trúc
- Semantic segmentation (gán nhãn từng pixel: bò/nền), ước lượng độ sâu từng pixel, phiên âm audio, dịch Anh→Pháp, **sinh ảnh từ caption**.
- Khó ở 2 điểm: (1) output **mơ hồ** — nhiều bản dịch/ảnh đều đúng cho cùng một input; (2) output phải tuân theo **"ngữ pháp"** — không phải chuỗi từ nào cũng thành câu hợp lệ, không phải tập RGB nào cũng thành ảnh thật.
- May mắn: "ngữ pháp" này học được **không cần nhãn** — bằng cách học thống kê của kho văn bản/ảnh lớn → nối sang unsupervised learning.

### Unsupervised learning — học không giám sát
- Không có nhãn output → mục tiêu là **mô tả/hiểu cấu trúc dữ liệu**, không phải học ánh xạ input→output.
- Sách tập trung vào **generative model**: học cách **sinh ra ví dụ mới** không phân biệt được với dữ liệu train. Có loại mô tả tường minh phân phối xác suất (rồi lấy mẫu từ đó), có loại chỉ học cơ chế sinh.
- **Conditional generation:** inpainting (xóa dây cáp khỏi ảnh, model vẽ lại phần thiếu), text completion. LLM hiện đại mạnh tới mức trông như "hiểu" — nhưng thực chất chỉ biết **thống kê của ngôn ngữ**, không hiểu ý nghĩa câu trả lời.
- **Latent variables (biến ẩn):** dữ liệu thực có chiều thấp hơn số biến quan sát gợi ý — khuôn mặt người ≈ 42 cơ mặt; câu tiếng Anh hợp lệ chỉ là tập con rất nhỏ của mọi chuỗi từ ngẫu nhiên. Deep learning mô tả ánh xạ giữa **biến ẩn (chiều thấp, phân phối đơn giản)** và dữ liệu. Lấy mẫu biến ẩn → sinh ví dụ mới.
- **Interpolation:** tìm biến ẩn của 2 ảnh thật, nội suy giữa chúng rồi map ngược ra ảnh → ảnh trung gian vừa hợp lý vừa trộn đặc điểm cả hai.
- Nối sang supervised: caption→ảnh thì map **biến ẩn của text ↔ biến ẩn của ảnh** thay vì map trực tiếp: cần ít cặp text/ảnh hơn, ảnh sinh ra hợp lý hơn, và có thể sinh **nhiều ảnh** cho cùng một caption (thêm ngẫu nhiên).

### Reinforcement learning — học tăng cường
- **Agent** sống trong thế giới, mỗi bước chọn **action** làm đổi **state** (không nhất thiết tất định), nhận **reward**; mục tiêu: chọn action để reward trung bình cao.
- **Temporal credit assignment problem:** reward đến **muộn** sau action — khó biết action nào thực sự gây ra reward. Ví dụ: robot humanoid học đi — phải làm rất nhiều action mới tới được checkpoint.
- **Exploration vs exploitation:** robot đã biết bò lê bằng một chân (exploit — chậm nhưng có reward) hay thử đứng dậy đi (explore — có thể nhanh hơn nhiều)? Cờ vua: dùng nước khai cuộc đã biết hay thử nước mới?
- **Policy network:** dùng deep net map **state → action** (trạng thái bàn cờ → nước đi; cảm biến robot → chuyển động khớp).

### Ethics — 5 mối lo (chương 1.4)
1. **Bias and fairness (thiên kiến):** train dự đoán lương từ dữ liệu lịch sử → tái tạo thiên kiến lịch sử (dự đoán nữ lương thấp hơn nam); AI siêu phân giải làm người da màu trông "trắng hơn"; AI sinh ảnh chỉ vẽ đàn ông khi được yêu cầu vẽ luật sư.
2. **Explainability (giải thích được):** hệ thống quyết định nhưng không biết **dựa trên thông tin gì** — lĩnh vực explainable AI, local explanation (giải thích từng quyết định riêng lẻ).
3. **Weaponizing AI (vũ khí hóa):** mọi công nghệ lớn đều bị dùng cho chiến tranh; AI đã và đang được triển khai quân sự.
4. **Concentrating power (tập trung quyền lực):** các công ty lớn đầu tư AI để thu lợi nhuận khổng lồ; tự động hóa việc làm ảnh hưởng nặng nhất tới lao động lương thấp, ít kỹ năng.
5. **Existential risk (rủi ro tồn vong):** hệ thống có năng lực vượt con người → có thể không kiểm soát được, thậm chí không hiểu được động cơ của nó.
- Ngoài ra: giám sát, tin giả, xâm phạm riêng tư, gian lận, thao túng thị trường, **năng lượng train AI góp phần biến đổi khí hậu**. Tác giả kêu gọi: nhà khoa học phải suy nghĩ về trách nhiệm của mình, soi xét giá trị của tổ chức mình làm việc — "ethics-washing" (làm màu đạo đức) hay thật sự có cơ chế dừng dự án đáng ngờ.

---

## Chương 2 — Supervised learning

### Khung hình thức
- Model: **y = f[x, φ]** — hàm toán học; **inference** = tính dự đoán y từ input x. Tham số φ xác định quan hệ cụ thể.
- Training set: I cặp **{xᵢ, yᵢ}**. **Loss L[φ]** = số vô hướng đo độ lệch giữa dự đoán và output thật. Training = tìm **φ̂ = argmin_φ L[φ]**.
- Sau train, đánh giá trên **test data riêng** → xem model **generalize** (khái quát hóa) tới ví dụ chưa thấy tốt không.

### Ví dụ: hồi quy tuyến tính 1D
- Model: **y = φ₀ + φ₁·x** (φ₀ = intercept, φ₁ = slope). Họ hàm = mọi đường thẳng; tham số chọn ra một đường.
- **Least-squares loss:** L[φ] = Σᵢ (f[xᵢ, φ] − yᵢ)² = Σᵢ (φ₀ + φ₁xᵢ − yᵢ)². Bình phương làm hướng lệch không còn quan trọng (lý do lý thuyết ở chương 5).
- Loss là một **mặt** trên không gian (φ₀, φ₁); tham số tốt nhất nằm ở **đáy** của mặt đó.
- **Training = gradient descent:** khởi tạo tham số ngẫu nhiên → đo gradient (hướng dốc nhất) → bước xuống dốc → lặp lại tới khi gradient phẳng. Hồi quy tuyến tính có nghiệm dạng đóng (closed-form), nhưng model phức tạp thì không — nên phải dùng phương pháp lặp.
- **Underfitting:** model quá đơn giản (đường thẳng không bắt được quan hệ thật). **Overfitting:** model quá linh hoạt, học cả đặc thù ngẫu nhiên của dữ liệu train → dự đoán lạ trên dữ liệu mới.

### Ghi chú cuối chương
- **Loss vs cost:** loss = số hạng cho từng điểm dữ liệu; cost = tổng lượng cần tối thiểu hóa (có thể chứa thêm số hạng không gắn với điểm nào). Objective function = hàm bất kỳ cần tối đa/tối thiểu hóa.
- **Generative vs discriminative:** generative model x = g[y, φ] phải **đảo ngược** để suy luận (y = g⁻¹[x, φ]) — khó. Discriminative model (y = f[x, φ]) thống trị ML hiện đại: model linh hoạt + dữ liệu lớn thắng việc cài đặt kiến thức tiên nghiệm.
- Lộ trình tiếp theo: chương 3–4 = model; chương 5 = loss; chương 6–7 = training; chương 8 = đo hiệu năng; chương 9 = regularization.

---

## Chương 3 — Shallow neural networks (mạng nông)

### Ví dụ mạng 1 input, 1 output, 3 hidden unit
```
y = φ₀ + φ₁·a[θ₁₀+θ₁₁x] + φ₂·a[θ₂₀+θ₂₁x] + φ₃·a[θ₃₀+θ₃₁x]     (10 tham số)
a[z] = ReLU[z] = max(0, z)   (0 nếu z<0, z nếu z≥0)
```
- 3 bước: (1) tính 3 hàm tuyến tính của input; (2) cho qua **activation function**; (3) cộng tuyến tính có trọng số + offset.
- **Hidden units** h₁, h₂, h₃ = kết quả sau activation. Mỗi hidden unit đóng góp một **"joint" (khớp)** — điểm mà đường thẳng của nó cắt 0 và bị ReLU cắt. 3 hidden unit → tối đa **4 vùng tuyến tính**; mỗi vùng = một **activation pattern** khác nhau (unit nào active/inactive). Độ dốc của một vùng = tổng θ•₁·φ• của các unit đang active trong vùng đó.

### Universal approximation theorem (định lý xấp xỉ vạn năng)
- Với **D hidden unit**, output là hàm tuyến tính từng khúc với tối đa **D+1 vùng**. Thêm unit → thêm vùng → xấp xỉ hàm liên tục càng chính xác.
- **Định lý:** với đủ hidden unit, mạng nông xấp xỉ được **bất kỳ hàm liên tục nào** trên tập compact với độ chính xác tùy ý (Cybenko 1989 cho sigmoid; Hornik 1991 cho activation phi tuyến tổng quát).

### Input/output đa chiều
- **Output đa chiều:** mỗi output là một hàm tuyến tính **khác nhau** của cùng các hidden unit → các "joint" ở cùng vị trí, nhưng độ dốc và độ cao khác nhau.
- **Input đa chiều:** hidden unit = ReLU của một **mặt phẳng có hướng**; output = mặt liên tục tạo bởi các **vùng đa giác lồi** (convex polytopes), mỗi vùng ứng với một activation pattern.
- **Số vùng tăng rất nhanh theo chiều input:** D siêu phẳng trong không gian Di chiều tạo tối đa Σⱼ₌₀^Di C(D,j) vùng (Zaslavsky 1975); thực tế mạng nông tạo từ **2^Di tới 2^D vùng**. Ví dụ: D=500, Di=100 → hơn **10^107 vùng** chỉ với 51.001 tham số.

### Dạng tổng quát
```
h_d = a[θ_d0 + Σᵢ θ_di·xᵢ]          (hidden unit d)
y_j = φ_j0 + Σ_d φ_jd·h_d           (output j)
```
- Activation **bắt buộc phi tuyến** — nếu không, toàn bộ ánh xạ input→output chỉ còn là tuyến tính.
- Thuật ngữ: **MLP (multi-layer perceptron)** = mạng có ≥1 hidden layer; **shallow** = 1 hidden layer; **deep** = nhiều hidden layer; **feed-forward** = đồ thị kết nối không có vòng lặp; **fully connected** = mọi phần tử lớp này nối tới mọi phần tử lớp sau; **weights** = tham số slope; **biases** = tham số offset; **pre-activations** = giá trị trước ReLU; **activations** = giá trị sau ReLU.

### Activation functions — cả một "vườn thú"
- **Sigmoid / tanh:** bão hòa (đạo hàm ≈ 0 với input lớn) → training chậm/khó.
- **ReLU:** đạo hàm = 1 với input > 0 → train ổn định, hiệu quả; nhưng có **dying ReLU problem** — nếu mọi ví dụ train đều cho input âm vào một ReLU thì gradient = 0, không cải thiện được trọng số đầu vào của nó.
- Biến thể chống dying ReLU: **Leaky ReLU** (dốc 0,1 phần âm), **PReLU** (dốc âm là tham số học được), ELU, SELU, SoftPlus, GELU, SiLU, **Swish** (x/(1+exp(−βx)) — tìm bằng search tự động), HardSwish.
- Sách dùng **ReLU xuyên suốt** vì dễ phân tích số vùng tuyến tính.

### Lịch sử ngắn
- McCulloch & Pitts (1943): neuron nhân tạo đầu tiên, chưa có thuật toán học. Rosenblatt (1958): **perceptron** + thuật toán học trọng số. Minsky & Papert (1969): hàm tuyến tính không đủ, hidden layer + activation phi tuyến có thể học quan hệ tổng quát — nhưng thuật toán của Rosenblatt không học được tham số của chúng. Tới những năm 1980, **backpropagation** (chương 7) mới làm sống lại lĩnh vực.
- **Linear vs affine:** tổng có trọng số không offset = linear; có offset = affine (sách dùng lẫn, gọi chung là linear).

---

## Chương 4 — Deep neural networks (mạng sâu)

### Ghép 2 mạng nông → "gấp" không gian input
- Ghép mạng 1 (3 hidden unit) vào mạng 2 (3 hidden unit): mạng 1 tạo 3 vùng có dốc **đổi dấu xen kẽ** → 3 khoảng x khác nhau map về cùng khoảng y; mạng 2 áp hàm của nó lên khoảng y đó **3 lần** → tổng **9 vùng tuyến tính** (mạng nông 6 unit chỉ tạo được 7 vùng). Ví dụ 2D: 7 vùng → 13 vùng.
- Cách nhìn khác: mạng 1 **"gấp" không gian input** lại lên chính nó; mạng 2 áp hàm lên không gian đã gấp.

### Ghép mạng = trường hợp đặc biệt của mạng sâu 2 hidden layer
- Linear ∘ linear = linear: thay biểu thức y của mạng 1 vào mạng 2, pre-activation của lớp 2 thành hàm tuyến tính của hidden unit lớp 1 (tham số ψ). Mạng sâu 2 lớp là **họ hàm rộng hơn** (ψ tùy ý, trong khi ghép mạng thì ψ bị ràng buộc thành outer product).
- Quy trình mỗi lớp: tính các hàm tuyến tính từng khúc (joint cùng chỗ) → **cắt bằng ReLU** (thêm joint mới) → tổ hợp lại. Cuối cùng vẫn chỉ là **một phương trình** y' = f[x] — chỉ là biểu thức dài khó đọc.
- **Matrix notation:** h_k = a[β^{k−1} + Ω^{k−1}·h_{k−1}]; y = β^K + Ω^K·h_K. Tham số φ = {β^k, Ω^k}. Ω⁰ kích thước D₁×Di; Ω^K kích thước Do×D_K.
- **Hyperparameters:** K = **depth** (số hidden layer), D₁…D_K = **width** (số unit mỗi lớp). Được chọn **trước** khi học tham số. Mạng nơ-ron = "họ của các họ hàm".

### Shallow vs Deep — so sánh
| Tiêu chí | Kết luận |
|---|---|
| Khả năng xấp xỉ | Cả hai đều xấp xỉ được hàm liên tục tùy ý nếu đủ capacity (mạng sâu sao chép được mạng nông bằng hàm identity ở lớp 2). |
| Số vùng / tham số | Mạng nông D unit: D+1 vùng, 3D+1 tham số. Mạng sâu K lớp × D unit: tới **(D+1)^K vùng**, 3D+1+(K−1)D(D+1) tham số. Ví dụ K=5, D=10: 471 tham số → **161.051 vùng**; K=5, D=50, Di=10: 10.801 tham số → hơn **10^40 vùng**. Nhưng các vùng có phụ thuộc/phức tạp đối xứng — chỉ có lợi nếu hàm thật có tính đối xứng tương tự hoặc thật sự là hợp thành của hàm đơn giản. |
| **Depth efficiency** | Có hàm mà mạng nông cần **nhiều hơn theo cấp số nhân** số hidden unit để xấp xỉ bằng mạng sâu. |
| Input lớn, có cấu trúc | Ảnh ~10⁶ pixel: fully connected không khả thi (quá nhiều tham số; cùng một vật ở mọi vị trí nên xử lý giống nhau) → cần xử lý **local-to-global qua nhiều lớp** (chương 10). |
| Train & generalize | Mạng sâu vừa phải thường **dễ train hơn** mạng nông (over-parameterized → nhiều nghiệm tương đương, dễ tìm); quá nhiều lớp lại khó train (chương 11). Mạng sâu **generalize tốt hơn** trong thực tế; kết quả tốt nhất thường ở mạng hàng chục/trăm lớp. Cả hai hiện tượng đều **chưa hiểu rõ** (chương 20). |

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Mental model "model = họ hàm số":** tham số chọn thành viên, loss đo độ sai, train = đi xuống dốc loss. Dùng khung này để giải thích cho bất kỳ ai hỏi "AI học như thế nào" — kể cả khi viết kịch bản giải thích AI cho video.
2. **Underfitting/overfitting:** khi "học" từ dữ liệu cũ (video, bình luận, chủ đề) — quá đơn giản thì không bắt được pattern, quá phức tạp thì nhớ vẹt trường hợp riêng rồi áp sai chỗ mới.
3. **ReLU + piecewise linear:** hiểu vì sao mạng nơ-ron "gấp" không gian input thành vùng — tương tự cách chia khán giả/chủ đề thành các nhóm vùng khác nhau.
4. **Latent variables:** ý tưởng cốt lõi đằng sau mọi generative model (ảnh, nhạc, text) — dữ liệu thực chỉ là bề mặt của ít biến ẩn; nội suy trong không gian ẩn = morph giữa hai ảnh, hai giọng đọc.
5. **Depth efficiency:** vì sao mạng sâu thắng mạng nông với cùng ngân sách tham số — dùng khi chọn kiến trúc model cho tool/app cần viết.
6. **Ethics — 5 mối lo:** checklist khi triển khai AI (bias, explainability, vũ khí hóa, tập trung quyền lực, rủi ro tồn vong) — đặc biệt bias khi dùng AI chấm/đánh giá nội dung hay gợi ý video.

## Tiến độ sách S1
- ✅ **Phần 1 (chương 1–4): Introduction, Supervised learning, Shallow networks, Deep networks — xong 20/11 (ghi chú này).**
- ⏳ Phần 2: chương 5–9 — Loss functions, Fitting models, Gradients & initialization, Measuring performance, Regularization.
- ⏳ Phần 3: chương 10–13 — CNN, Residual networks, Transformers, Graph neural networks.
- ⏳ Phần 4: chương 14–18 — Unsupervised learning, GAN, Normalizing flows, VAE, Diffusion models.
- ⏳ Phần 5: chương 19–21 — Reinforcement learning, Why does deep learning work?, Ethics.
- ⏳ Sau S1: freeCodeCamp Python v9, Udemy 73 khóa, GLM-5.3-Flash build + train — **quy tắc của Hoai (20/11 ~23:19): khóa nào đến bước cần anh làm (thi chứng chỉ, nhập token,…) thì BỎ QUA, học tiếp khóa khác.**

## Nguồn
- Sách: Simon J.D. Prince, *Understanding Deep Learning*, MIT Press 2023 — PDF miễn phí (CC BY-NC-ND), release v5.0.3 (08/02/2026): https://github.com/udlbook/udlbook
- Trang sách + 68 notebook Python: https://udlbook.github.io/udlbook/
- PDF local: `~/workspace/knowledge/understanding-deep-learning/UnderstandingDeepLearning.pdf` (541 trang)
