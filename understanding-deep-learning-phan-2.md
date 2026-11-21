# S1 — Understanding Deep Learning (Simon J.D. Prince): Phần 2 — Chương 5–9

> Sách: **Understanding Deep Learning** — Simon J.D. Prince, MIT Press 2023, 544 trang, **PDF MIỄN PHÍ** (CC BY-NC-ND, open-access). Bản đang đọc: release v5.0.3 (08/02/2026), 541 trang, tải từ https://github.com/udlbook/udlbook/releases (file `UnderstandingDeepLearning_02_09_26_C.pdf`, 22 MB, lưu tại `~/workspace/knowledge/understanding-deep-learning/`).
> Sách có 21 chương + 68 notebook Python thực hành (https://udlbook.github.io/udlbook/). Mục tiêu của sách: hiểu **nguyên lý** đằng sau deep learning — không quá lý thuyết (không có chứng minh), cũng không quá thực hành (gần như không có code).
> Ghi chú này: phần 2 (chương 5–9, trang sách 56–154). Phần 1 (chương 1–4) ở `understanding-deep-learning-phan-1.md`. Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong tài liệu gốc.

---

## Chương 5 — Loss functions (Hàm mất mát)

Chương 2–4 đã xây các họ hàm (hồi quy tuyến tính, mạng nông, mạng sâu) ánh xạ input → output, với thành viên cụ thể do tham số φ quyết định. Chương này định nghĩa "tốt nhất" nghĩa là gì: **hàm loss** L[φ] trả về một số đo độ lệch giữa dự đoán f[xᵢ, φ] và đáp án thật yᵢ trên tập huấn luyện {xᵢ, yᵢ}. Huấn luyện = tìm φ làm loss nhỏ nhất.

### 5.1 Maximum likelihood — công thức xây loss

Đổi góc nhìn: mạng không dự đoán trực tiếp y, mà dự đoán **tham số θ của một phân phối xác suất** Pr(y|θ) trên miền output. Loss khuyến khích mỗi yᵢ có xác suất cao dưới phân phối Pr(yᵢ|xᵢ) mà mạng tính từ xᵢ.

- **Maximum likelihood:** chọn φ để cực đại hoá tích xác suất của toàn bộ dữ liệu huấn luyện: φ̂ = argmax_φ ∏ᵢ Pr(yᵢ|f[xᵢ, φ]). Giả định: (i) cùng dạng phân phối cho mọi điểm dữ liệu, (ii) các Pr(yᵢ|xᵢ) độc lập nên tích phân rã được.
- **Log-likelihood:** lấy log (hàm đơn điệu tăng, không đổi vị trí cực đại) để biến tích thành tổng — tránh số quá nhỏ không biểu diễn được bằng số thực hữu hạn.
- **Negative log-likelihood (NLL):** nhân −1 để thành bài toán cực tiểu hoá — đó chính là hàm loss: L[φ] = −Σᵢ log Pr(yᵢ|f[xᵢ, φ]).
- **Suy luận (inference):** lấy điểm cực đại của phân phối: ŷ = argmax_y Pr(y|f[x, φ̂]). Với phân phối chuẩn, cực đại ở trung bình μ — tức chính output của mạng.

**Công thức 4 bước xây loss (mục 5.2):** (1) Chọn phân phối Pr(y|θ) phù hợp miền của y. (2) Cho mạng dự đoán một hoặc nhiều tham số θ của phân phối đó. (3) Huấn luyện bằng cách cực tiểu hoá NLL. (4) Suy luận: trả về toàn bộ phân phối hoặc điểm cực đại của nó.

### 5.3 Ví dụ 1: hồi quy một biến — least squares từ đâu ra

- Chọn phân phối chuẩn (Gaussian) trên y ∈ R, tham số μ, σ². Mạng dự đoán μ = f[x, φ]; σ² coi là hằng số chưa biết.
- NLL của phân phối chuẩn, sau khi bỏ các hạng không phụ thuộc φ và hệ số tỉ lệ dương 1/(2σ²), rút gọn thành **least squares loss**: L[φ] = Σᵢ (yᵢ − f[xᵢ, φ])².
- → Least squares không phải lựa chọn tuỳ tiện: nó là hệ quả của giả định (i) các dự đoán độc lập, (ii) y tuân theo phân phối chuẩn với trung bình do mạng dự đoán.
- **Ước lượng phương sai:** có thể học luôn σ² (cực tiểu hoá NLL theo cả φ và σ²) — lúc đó σ̂² cho biết độ không chắc chắn của dự đoán.
- **Hồi quy heteroscedastic (phương sai thay đổi theo input):** mạng có 2 đầu ra — đầu 1 dự đoán trung bình μ, đầu 2 dự đoán phương sai σ² (phải qua hàm đảm bảo dương, ví dụ bình phương: σ² = f₂[x, φ]²). Ngược lại là homoscedastic (phương sai hằng số khắp nơi).

### 5.4 Ví dụ 2: phân loại nhị phân — binary cross-entropy

- y ∈ {0, 1} → chọn phân phối **Bernoulli**, tham số λ = P(y = 1): Pr(y|λ) = (1−λ)^(1−y) · λ^y.
- λ phải nằm trong [0, 1] nhưng output mạng là số thực bất kỳ → qua **logistic sigmoid** sig[z] = 1/(1+exp[−z]): λ = sig[f[x, φ]].
- NLL cho ra **binary cross-entropy loss**: L[φ] = −Σᵢ (1−yᵢ)·log(1−sig[f[xᵢ, φ]]) − yᵢ·log(sig[f[xᵢ, φ]]).
- Suy luận: y = 1 nếu λ > 0.5, ngược lại y = 0.

### 5.5 Ví dụ 3: phân loại đa lớp — softmax + multiclass cross-entropy

- y ∈ {1, …, K} → chọn phân phối **categorical**, K tham số λ₁…λ_K (mỗi cái trong [0,1], tổng = 1).
- Mạng có K đầu ra → qua **softmax**: softmax_k[z] = exp[z_k] / Σ_{k'} exp[z_{k'}] (hàm exp đảm bảo dương, mẫu số đảm bảo tổng = 1).
- NLL cho ra **multiclass cross-entropy loss**: L[φ] = −Σᵢ (f_{yᵢ}[xᵢ, φ] − log Σ_{k'} exp[f_{k'}[xᵢ, φ]]).
- Suy luận: lấy lớp có xác suất cao nhất.

### 5.5.1 Các kiểu dữ liệu khác — chỉ cần đổi phân phối

Cùng một công thức, đổi phân phối theo miền của y: Laplace hoặc t-distribution → hồi quy robust (ít nhạy outlier); hỗn hợp Gaussian → hồi quy đa mode; exponential/gamma → dự đoán độ lớn (y ∈ R⁺); beta → tỉ lệ (y ∈ [0,1]); Poisson → đếm sự kiện; von Mises → hướng (góc); Plackett-Luce → xếp hạng (hoán vị).

### 5.6 Nhiều đầu ra

Thường giả định các dự đoán độc lập: Pr(y|f[x, φ]) = ∏_d Pr(y_d|f_d[x, φ]). Khi lấy log âm, tích thành tổng — loss là **tổng các loss từng đầu ra**. Muốn dự đoán 2 kiểu khác nhau cùng lúc (ví dụ hướng gió + cường độ gió) cũng vậy: von Mises cho hướng, exponential cho cường độ, cộng loss lại.

### 5.7 Cross-entropy = negative log-likelihood

- Cross-entropy xuất phát từ ý tưởng khác: cực tiểu hoá **khoảng cách KL divergence** giữa phân phối thực nghiệm q(y) của dữ liệu (tổng các khối điểm Dirac tại yᵢ) và phân phối mô hình Pr(y|θ).
- Khai triển KL: hạng đầu không phụ thuộc θ nên bỏ; hạng còn lại chính là cross-entropy; thay q(y) vào thì rút gọn đúng bằng **NLL** (chỉ khác hệ số tỉ lệ 1/I không ảnh hưởng vị trí cực tiểu).
- → Hai cách nghĩ (cực đại hoá likelihood / cực tiểu hoá khoảng cách tới phân phối dữ liệu) là **tương đương**. Vì vậy "cross-entropy loss" và "negative log-likelihood loss" là một.

**Tóm tắt chương 5:** mạng dự đoán tham số của phân phối xác suất trên output; loss = NLL. Least squares (hồi quy), binary cross-entropy (2 lớp), multiclass cross-entropy (K lớp) đều là trường hợp riêng của cùng một công thức.

---

## Chương 6 — Fitting models (Huấn luyện mô hình)

Bài toán huấn luyện: tìm φ̂ = argmin_φ L[φ]. Các thuật toán chuẩn cho mạng nơ-ron đều **lặp**: khởi tạo tham số theo heuristic, rồi chỉnh dần sao cho loss giảm.

### 6.1 Gradient descent

- Bước 1: tính gradient ∂L/∂φ tại vị trí hiện tại (hướng dốc lên của loss).
- Bước 2: cập nhật φ ← φ − α·∂L/∂φ (đi xuống dốc, α = learning rate, có thể cố định hoặc line search — thử nhiều α để chọn cái giảm loss nhiều nhất).
- Tại điểm cực tiểu, mặt loss phẳng → gradient = 0 → tham số ngừng thay đổi. Thực tế: dừng khi độ lớn gradient đủ nhỏ.
- **Hồi quy tuyến tính:** loss luôn có một cực tiểu toàn cục duy nhất — hàm **lồi** (convex): đi xuống dốc kiểu gì cũng tới đích, huấn luyện không thể thất bại.
- **Mô hình phi tuyến (mạng nông/sâu, ví dụ mô hình Gabor 2 tham số):** loss **không lồi** — nhiều cực tiểu địa phương (local minima) và **điểm yên ngựa** (saddle point: gradient = 0 nhưng tăng theo hướng này, giảm theo hướng khác; gần điểm yên ngựa mặt loss phẳng nên dễ nhầm là đã hội tụ). Gradient descent khởi tạo sai "thung lũng" sẽ kẹt ở cực tiểu địa phương, không có cách nào biết còn nghiệm tốt hơn ở nơi khác.

### 6.2 Stochastic gradient descent (SGD)

- Mỗi lần lặp chỉ dùng một **minibatch** (tập con ngẫu nhiên của dữ liệu) để tính gradient: φ_{t+1} ← φ_t − α·Σ_{i∈B_t} ∂ℓᵢ/∂φ. Batch = toàn bộ dữ liệu thì chính là gradient descent thường (full-batch).
- **Epoch:** một lượt qua toàn bộ dữ liệu (các batch rút không hoàn lại).
- Cách nhìn khác: SGD là gradient descent xác định trên một hàm loss **thay đổi liên tục** theo từng batch (kỳ vọng của loss và gradient vẫn như GD).
- **Ưu điểm:** (i) mỗi bước vẫn cải thiện fit trên một tập con; (ii) mọi ví dụ đều đóng góp như nhau; (iii) rẻ hơn (chỉ tính trên tập con); (iv) có thể **thoát cực tiểu địa phương** (bước đi có thể tạm thời đi lên, nhảy sang thung lũng khác); (v) ít kẹt ở saddle point (luôn có batch cho gradient đáng kể); (vi) bằng chứng thực nghiệm: SGD tìm được tham số tổng quát hoá tốt.
- **Learning rate schedule:** α bắt đầu cao (khám phá, nhảy giữa các thung lũng) rồi giảm dần theo hệ số cố định sau mỗi N epoch (tinh chỉnh khi đã gần đúng chỗ).

### 6.3 Momentum

- Cập nhật bằng tổ hợp có trọng số của gradient hiện tại và hướng đi bước trước: m_{t+1} ← β·m_t + (1−β)·Σ∂ℓᵢ/∂φ; φ_{t+1} ← φ_t − α·m_{t+1}. β ∈ [0,1) điều khiển độ mượt.
- Hiệu ứng: bước đi là tổng có trọng số vô hạn của mọi gradient trước đó (trọng số giảm dần về quá khứ). Gradient cùng hướng nhiều bước → learning rate hiệu dụng tăng; gradient đổi hướng liên tục → triệt tiêu lẫn nhau → giảm dao động trong thung lũng, đường đi mượt hơn.
- **Nesterov accelerated momentum:** tính gradient tại **điểm dự đoán** (φ_t − αβ·m_t — nơi momentum sẽ đưa tới) thay vì tại điểm hiện tại; gradient lúc này đóng vai trò hiệu chỉnh đường đi của momentum.

### 6.4 Adam (Adaptive moment estimation)

- Vấn đề của GD với bước cố định: loss dốc theo hướng này, thoải theo hướng khác → khó chọn learning rate vừa tiến nhanh vừa ổn định.
- Ý tưởng: chuẩn hoá gradient bằng căn bậc hai của **gradient bình phương** (v_{t+1}) để mỗi toạ độ đi một khoảng cố định α, chỉ giữ lại dấu (hướng xuống dốc). Nhưng thuật toán đơn giản này chỉ dao động quanh cực tiểu, không hội tụ.
- **Adam = chuẩn hoá + momentum cho cả hai thống kê:** m_{t+1} ← β·m_t + (1−β)·∂L/∂φ (trung bình gradient); v_{t+1} ← γ·v_t + (1−γ)·(∂L/∂φ)² (trung bình gradient bình phương); **hiệu chỉnh chệch (bias correction)** m̃ = m/(1−β^{t+1}), ṽ = v/(1−γ^{t+1}) vì lúc đầu các ước lượng bị nhỏ giả tạo (quá khứ toàn 0); cập nhật φ_{t+1} ← φ_t − α·m̃/√(ṽ+ε).
- Adam thường dùng trong môi trường stochastic (gradient tính trên minibatch). Ưu điểm thực tế: cân bằng thay đổi giữa các tầng (độ lớn gradient phụ thuộc độ sâu), **ít nhạy với learning rate khởi tạo**, không cần schedule phức tạp.

### 6.5–6.6 Siêu tham số + tóm tắt

- Thuật toán học, batch size, learning rate schedule, hệ số momentum đều là **hyperparameter** (khác với tham số mô hình) — chọn chúng nhiều khi là nghệ thuật, thường phải huấn luyện nhiều mô hình và chọn cái tốt nhất (hyperparameter search — xem chương 8).
- **Tóm tắt:** GD đo gradient rồi đi xuống dốc tới khi hội tụ; hàm không lồi có cực tiểu địa phương + saddle point; SGD thêm nhiễu để thoát bẫy; momentum làm mượt đường đi; Adam chuẩn hoá từng toạ độ + momentum.

---

## Chương 7 — Gradients and initialization (Đạo hàm và khởi tạo)

SGD cần đạo hàm của loss theo tham số ở mọi tầng. Chương này giải quyết 2 việc riêng của mạng nơ-ron: (1) tính đạo hàm hiệu quả — **backpropagation**; (2) khởi tạo tham số sao cho huấn luyện ổn định.

### 7.1–7.2 Bài toán + 2 quan sát trực giác

- Mạng 3 tầng ẩn: h₁ = a[β₀+Ω₀x], h₂ = a[β₁+Ω₁h₁], h₃ = a[β₂+Ω₂h₂], f = β₃+Ω₃h₃. Loss tổng L = Σᵢ ℓᵢ; SGD cần ∂ℓᵢ/∂β_k và ∂ℓᵢ/∂Ω_k ở mọi tầng k.
- **Quan sát 1:** mỗi trọng số nhân với activation của hidden unit nguồn rồi cộng vào unit đích → ảnh hưởng của một thay đổi nhỏ ở trọng số tỉ lệ với activation nguồn. Vì vậy **forward pass**: chạy mạng cho mỗi ví dụ trong batch và **lưu lại toàn bộ activation** các tầng ẩn.
- **Quan sát 2:** thay đổi nhỏ ở một tham số gây hiệu ứng lan truyền qua các tầng sau. Để biết nó ảnh hưởng loss ra sao, cần biết mỗi tầng sau thay đổi tầng kế tiếp thế nào — mà các đại lượng này **dùng chung** cho nhiều tham số. Tính một lần rồi tái sử dụng: tính từ cuối mạng ngược về đầu (**backward pass**).

### 7.3 Ví dụ đồ chơi (toy model) — bản chất backprop

- Mô hình 8 tham số vô hướng, hợp thành sin/exp/cos, loss least squares. Đạo hàm trực tiếp bằng tay (ví dụ ∂ℓᵢ/∂ω₀) là biểu thức khổng lồ, dễ sai, và **không tận dụng tính dư thừa** (cùng một hạng exp xuất hiện nhiều lần).
- Backprop gồm: **forward pass** — tính và lưu các biến trung gian f₀, h₁, f₁, h₂, f₂, h₃, f₃, ℓᵢ theo thứ tự; **backward pass #1** — tính ∂ℓᵢ/∂f_k và ∂ℓᵢ/∂h_k theo thứ tự **ngược lại**, mỗi đạo hàm = đạo hàm đã tính ở bước trước × một hạng đơn giản (quy tắc chuỗi); **backward pass #2** — đạo hàm theo tham số: ∂ℓᵢ/∂β_k = ∂ℓᵢ/∂f_k (vì ∂f_k/∂β_k = 1), ∂ℓᵢ/∂ω_k = h_k·∂ℓᵢ/∂f_k (vì ∂f_k/∂ω_k = h_k) — khớp quan sát 1: đạo hàm trọng số tỉ lệ với activation nguồn đã lưu ở forward pass.
- Backprop đơn giản và hiệu quả hơn hẳn tính từng đạo hàm riêng lẻ.

### 7.4 Thuật toán backpropagation (mạng sâu, ReLU)

- Forward pass (lưu f_k, h_k): f₀ = β₀+Ω₀xᵢ; h_k = a[f_{k−1}]; f_k = β_k+Ω_k h_k.
- Backward pass: ∂ℓᵢ/∂β_k = ∂ℓᵢ/∂f_k; ∂ℓᵢ/∂Ω_k = (∂ℓᵢ/∂f_k)·h_kᵀ; ∂ℓᵢ/∂f_{k−1} = I[f_{k−1}>0] ⊙ Ω_kᵀ·(∂ℓᵢ/∂f_k) — đi ngược qua mạng, xen kẽ (i) nhân với chuyển vị ma trận trọng số Ω_kᵀ và (ii) ngưỡng hoá theo ReLU (đạo hàm ReLU = 0 khi đầu vào < 0, = 1 khi > 0; chỉ cần nhân điểm với vector chỉ thị I[f>0], không cần nhân ma trận đường chéo).
- Cộng đạo hàm của mọi ví dụ trong batch lại → gradient cho bước SGD.
- **Hiệu quả:** bước tốn kém nhất ở cả forward và backward đều là nhân ma trận (Ω và Ωᵀ) — chỉ cộng và nhân. **Nhược điểm:** tốn bộ nhớ — phải lưu mọi giá trị trung gian của forward pass, giới hạn kích thước mô hình huấn luyện được.
- **Algorithmic differentiation (tự động vi phân):** thực tế không ai tự code backprop — PyTorch/TensorFlow tự tính đạo hàm: mỗi thành phần (biến đổi tuyến tính, ReLU, loss) biết cách tính đạo hàm của chính nó, framework biết thứ tự phép toán nên tự chạy forward/backward; khai thác song song trên GPU; xử lý được **đồ thị tính toán không tuần hoàn bất kỳ** (có nhánh, không chỉ tuần tự); batch xử lý song song → input thành tensor nhiều chiều (ảnh RGB: tensor 4D = batch × cao × rộng × kênh).

### 7.5 Khởi tạo tham số — He initialization

- **Vấn đề:** f_k = β_k + Ω_k·a[f_{k−1}]. Khởi tạo bias = 0, trọng số ~ N(0, σ²). Nếu σ² quá nhỏ (vd 10⁻⁵): độ lớn pre-activation **co lại** qua từng tầng (ReLU còn cắt một nửa miền giá trị) → tiến về 0. Nếu σ² quá lớn (vd 10⁵): độ lớn **phình ra** qua từng tầng → tràn số thực dấu phẩy động. Tương tự ở backward pass: nhân liên tục với Ωᵀ làm độ lớn gradient co/phình không kiểm soát → **vanishing gradient** (cập nhật teo về 0) và **exploding gradient** (cập nhật mất ổn định).
- **Lập luận toán học (forward pass):** giả sử pre-activation tầng trước có phương sai σ_f², phân phối đối xứng quanh 0 → ReLU cắt một nửa → E[h²] = σ_f²/2. Phương sai pre-activation tầng sau: σ_{f'}² = D_h·σ_Ω²·σ_f²/2. Muốn σ_{f'}² = σ_f² (ổn định qua các tầng) → **σ_Ω² = 2/D_h** — đây là **He initialization** (D_h = số unit tầng nguồn).
- **Backward pass:** tương tự nhưng nhân với Ωᵀ → σ_Ω² = 2/D_{h'} (D_{h'} = số unit tầng đích). Nếu ma trận không vuông (D_h ≠ D_{h'}), không thỏa cả hai cùng lúc → thoả hiệp: σ_Ω² = 4/(D_h + D_{h'}).
- Thực nghiệm (mạng 50 tầng, 100 unit/tầng): He init giữ phương sai activation và gradient ổn định; giá trị khác đều tăng/giảm theo hàm mũ theo tầng.
- **Lưu ý lịch sử:** He (2015) gần giống Glorot/Xavier (2010) nhưng tính thêm hiệu ứng ReLU nên khác hệ số 2. PyTorch: `nn.init.kaiming_normal_` chính là He init.
- Các cách khác: BatchNorm (chuẩn hoá theo batch ở mọi bước — chương 11), gradient checkpointing (chỉ lưu activation mỗi N tầng, tính lại khi cần — đổi bộ nhớ lấy thời gian), micro-batching, mạng khả nghịch (tính ngược activation từ tầng sau, khỏi lưu gì).

### 7.6 Code mẫu + tóm tắt

- Code PyTorch trong sách: mạng 2 tầng ẩn (10→40→40→5), He init, loss MSE, SGD momentum 0.9, lr 0.01 (thực ra code dùng 0.1 trong optimizer), batch 10, 100 epoch, lr giảm một nửa mỗi 10 epoch. **Điểm rút ra: ý tưởng phức tạp nhưng cài đặt đơn giản — toàn bộ backprop gói trong một dòng `loss.backward()`.**
- **Tóm tắt:** backprop tính gradient hiệu quả bằng cách tái sử dụng đạo hàm theo thứ tự ngược; khởi tạo đúng tránh vanishing/exploding gradient.

---

## Chương 8 — Measuring performance (Đo hiệu năng)

### 8.1 Vì sao cần tập test riêng

- Hiệu năng huấn luyện ≠ hiệu năng thực tế. Ví dụ MNIST-1D: train 4000 ví dụ, mạng 2 tầng ẩn — lỗi train → ~0%, lỗi test chỉ giảm tới ~40% (may rủi là 90%). Loss test giảm 1500 bước đầu rồi **tăng lại** trong khi tỉ lệ lỗi test gần như không đổi: mô hình sai vẫn sai đó nhưng **tự tin hơn** (softmax đẩy pre-activation tới cực trị để xác suất dữ liệu train → 1) → NLL tăng.
- **Tổng quát hoá (generalization)** = mức độ giữ được hiệu năng trên tập test. Cần 3 tập: **train** (học tham số), **validation** (chọn siêu tham số), **test** (ước lượng hiệu năng cuối cùng). Đo trên test để chọn siêu tham số là gian lận — siêu tham số sẽ "vừa khít" test set mà không tổng quát hoá tiếp.

### 8.2 Ba nguồn lỗi: noise, bias, variance

Thí nghiệm tư duy: dữ liệu sinh từ hàm quasi-sinusoid + nhiễu Gaussian, fit bằng mạng nông đơn giản (hàm tuyến tính từng đoạn, khớp nối cách đều).

- **Noise (nhiễu):** quá trình sinh dữ liệu có nhiễu → cùng một x có thể có nhiều y đúng. **Không khắc phục được** — giới hạn cơ bản của hiệu năng test. (Có thể do ngẫu nhiên thật, nhãn sai, hoặc biến giải thích chưa quan sát được.)
- **Bias (độ chệch):** mô hình **không đủ linh hoạt** để khớp hàm thật ngay cả với tham số tốt nhất (vd mô hình 3 đoạn không mô tả nổi đường cong sin).
- **Variance (phương sai):** dữ liệu huấn luyện hữu hạn và nhiễu → fit trên các tập train khác nhau cho ra hàm hơi khác nhau; mô hình "học" cả nhiễu.
- **Công thức phân rã (hồi quy + least squares):** E[L] = **variance** + **bias²** + **noise** (cộng tính). Bias = độ lệch hệ thống giữa trung bình mô hình trên mọi tập train có thể và hàm thật; variance = độ dao động của mô hình quanh trung bình đó theo tập train.

### 8.3 Giảm lỗi — bias-variance trade-off

- **Giảm variance:** thêm dữ liệu huấn luyện (trung bình hoá nhiễu, phủ kín không gian input). Thêm dữ liệu gần như luôn cải thiện test.
- **Giảm bias:** tăng **capacity** (số hidden unit / số tầng) → mô hình linh hoạt hơn, khớp hàm thật sát hơn.
- **Tác dụng phụ:** với tập train cố định, tăng capacity làm **variance tăng** — mô hình dùng sức mạnh dư thừa để "học" nhiễu (**overfitting**). → **Bias-variance trade-off:** tồn tại capacity tối ưu (trong ví dụ đồ chơi: 4 hidden unit) nơi tổng bias+variance nhỏ nhất.

### 8.4 Double descent (giảm kép) — điều bất ngờ

- Thí nghiệm MNIST-1D (10.000 train, 5.000 test, tăng dần số hidden unit): lỗi train → 0 khi số tham số ≈ số ví dụ train. Lỗi test **không tăng lại** như trade-off dự đoán mà **tiếp tục giảm**. Với 15% nhãn nhiễu: lỗi test giảm → tăng (đúng trade-off cổ điển) → **giảm lại lần nữa**, cuối cùng còn tốt hơn mức tốt nhất ở đoạn đầu. Hiện tượng này gọi là **double descent**.
- Ba vùng: **classical / under-parameterized** (đoạn đầu), **critical** (đỉnh ở giữa — model vừa đủ sức ghi nhớ dữ liệu), **modern / over-parameterized** (đoạn sau).
- **Giải thích:** khi model đã fit train gần hoàn hảo, capacity thêm chỉ thay đổi hành vi **giữa các điểm dữ liệu**. Trong không gian nhiều chiều, dữ liệu train cực kỳ thưa (MNIST-1D 40 chiều, 10.000 ví dụ: lượng tử hoá mỗi chiều thành 10 bin → 10⁴⁰ bin, chỉ 10⁴ ví dụ — **curse of dimensionality**). Model càng nhiều capacity càng có khả năng **nội suy mượt mà** giữa các điểm gần nhau; trong lúc không có thông tin gì giữa các điểm, giả định mượt mà là hợp lý và tổng quát hoá tốt.
- **Câu hỏi chưa có lời giải chắc chắn:** tại sao model quá tham số lại *chọn* hàm mượt trong vô số hàm cùng fit train hoàn hảo? Hai khả năng: (i) khởi tạo đã thiên về hàm mượt và huấn luyện không rời khỏi miền đó; (ii) thuật toán huấn luyện "ưa" hội tụ về hàm mượt — tức có tác dụng **implicit regularization** (xem chương 9.2).
- Hệ quả lạ: thêm dữ liệu huấn luyện đôi khi làm test **tệ đi** (đẩy model từ vùng modern về vùng critical của đường cong mới).

### 8.5 Chọn siêu tham số + tóm tắt

- Capacity + thuật toán học + learning rate… đều là siêu tham số. Cách chọn thực tế: huấn luyện nhiều mô hình, đo trên **validation set**, giữ cái tốt nhất, rồi mới đo một lần trên test set.
- Không gian siêu tham số nhỏ hơn không gian tham số nhưng vẫn quá lớn để thử hết; nhiều siêu tham số rời rạc hoặc phụ thuộc nhau → không dùng được gradient descent; các thuật toán tối ưu siêu tham số lấy mẫu thông minh dựa trên kết quả trước — rất tốn kém (mỗi lần thử = huấn luyện cả một mô hình).
- **K-fold cross-validation:** khi dữ liệu ít, chia train+validation thành K phần, lần lượt dùng K−1 phần train + 1 phần validate, chọn siêu tham số theo hiệu năng trung bình — dùng được tỉ lệ dữ liệu lớn hơn để train, giảm variance.
- **Thực tế ≠ test set:** hiệu năng thực có thể tệ hơn nếu thống kê dữ liệu thay đổi theo thời gian (**data drift**): covariate shift (phân phối x đổi), prior shift (phân phối y đổi), concept shift (quan hệ x→y đổi). Mô hình đã triển khai phải được giám sát liên tục.
- **Tóm tắt:** lỗi test = noise + bias + variance. Thêm dữ liệu giảm variance; tăng capacity giảm bias nhưng tăng variance (trade-off cổ điển) — nhưng trong vùng quá tham số, lỗi test lại giảm lần nữa (double descent). Chọn capacity và siêu tham số bằng validation set.

---

## Chương 9 — Regularization (Chính quy hoá)

Câu hỏi của chương 8: trong vô số nghiệm cùng fit train hoàn hảo, làm sao thiên về nghiệm tổng quát hoá tốt? Đáp án: **regularizer** — bất kỳ yếu tố nào thiên vị mô hình về một tập con nghiệm.

### 9.1 Explicit regularization — thêm hạng phạt vào loss

- Loss có thêm hạng: φ̂ = argmin_φ [Σᵢ ℓᵢ + λ·g[φ]], g[φ] lớn khi tham số "không được ưa", λ điều khiển tỉ trọng. Cực tiểu của loss đã regularize thường khác vị trí cũ → hội tụ về tham số khác (mặt loss có ít cực tiểu địa phương hơn).
- **Diễn giải xác suất (MAP):** hạng regularize = **prior** Pr(φ) — kiến thức về tham số trước khi thấy dữ liệu. Maximum likelihood + prior = **maximum a posteriori**: λ·g[φ] = −log Pr(φ).
- **L2 regularization (weight decay / ridge / Tikhonov):** g[φ] = Σⱼ φⱼ² — phạt tổng bình phương tham số. Thường chỉ áp cho **trọng số, không áp bias**. Tác dụng: trọng số nhỏ → output biến thiên ít → hàm **mượt hơn**. λ nhỏ: gần như không ảnh hưởng; λ vừa: hàm mượt, giống hàm thật hơn; λ lớn: regularize lấn át likelihood → hàm quá mượt, fit tệ. Hiểu theo bias-variance: giảm variance (không cần đi qua mọi điểm dữ liệu) đổi lấy tăng bias (chỉ mô tả được hàm mượt); ở vùng quá tham số, regularize ép nội suy mượt giữa các điểm không có dữ liệu.

### 9.2 Implicit regularization — tác dụng phụ của thuật toán học

- **Gradient descent rời rạc:** GD liên tục (bước vô cùng nhỏ) tuân theo dφ/dt = −∂L/∂φ. GD rời rạc với bước α lệch khỏi đường liên tục — độ lệch này tương đương việc cộng thêm hạng phạt **(α/4)·‖∂L/∂φ‖²** vào loss: quỹ đạo bị **đẩy ra khỏi nơi gradient lớn** (mặt dốc). Không đổi vị trí cực tiểu (ở đó gradient = 0) nhưng đổi quỹ đạo → có thể hội tụ về cực tiểu khác. Giải thích vì sao full-batch GD tổng quát hoá tốt hơn với learning rate lớn.
- **SGD:** thêm một hạng nữa — phạt **phương sai của gradient giữa các batch**: SGD ngầm ưa những nơi mọi batch đồng thuận về độ dốc (mọi dữ liệu đều fit tốt), thay vì nơi một phần dữ liệu fit cực tốt còn phần khác fit kém (cùng loss tổng nhưng phương sai batch lớn). Đây có thể là một phần lý do SGD tổng quát hoá tốt hơn GD và **batch nhỏ thường tốt hơn batch lớn**.

### 9.3 Các heuristic cải thiện tổng quát hoá

- **Early stopping:** dừng huấn luyện trước khi hội tụ hẳn. Mô hình nắm được hình dạng thô của hàm thật trước, rồi mới overfit nhiễu sau. Tác dụng tương tự L2 (trọng số khởi tạo nhỏ, chưa kịp lớn lên) / giảm độ phức tạp hiệu dụng (đi ngược xuống đường cong bias-variance từ vùng critical). Chỉ có 1 siêu tham số (số bước dừng) — chọn bằng validation set mà **không cần train nhiều mô hình**: train một lần, theo dõi validation mỗi T bước, giữ lại tham số ở điểm validation tốt nhất.
- **Ensembling:** train nhiều mô hình, trung bình hoá dự đoán (hồi quy: trung bình output; phân loại: trung bình pre-softmax). Giả định lỗi các mô hình độc lập → triệt tiêu lẫn nhau. Cách tạo đa dạng: khởi tạo ngẫu nhiên khác nhau; **bagging** (bootstrap aggregating — lấy mẫu lại dữ liệu train có hoàn lại, mỗi mô hình train trên một bản sao khác nhau; điểm outlier vắng mặt ở một số bản → hàm fit ôn hoà hơn); siêu tham số khác nhau; họ mô hình khác nhau. Chắc chắn cải thiện test, giá phải trả: train + lưu + suy luận nhiều lần.
- **Dropout:** mỗi lần lặp SGD, tắt ngẫu nhiên một tập con hidden unit (thường 50%) — mạng bớt phụ thuộc vào bất kỳ unit nào, trọng số bị ép nhỏ lại. Tác dụng sâu hơn: loại bỏ các "khúc gấp" không mong muốn xa dữ liệu train (3 unit phối hợp tạo một đoạn lồi lõm cục bộ không ảnh hưởng loss nhưng không tổng quát hoá — dropout tắt 1 trong 3 → bước GD sau phải bù lại → dần xoá phụ thuộc đó). Lúc test: dùng đủ unit nhưng **nhân trọng số với (1 − p_dropout)** để bù (weight scaling inference rule); hoặc **Monte Carlo dropout** — chạy nhiều lần với các tập tắt khác nhau rồi kết hợp (gần như ensembling nhưng không cần train/lưu nhiều mạng).
- **Thêm nhiễu:** nhiễu vào **input** → làm mượt hàm học được (với hồi quy, tương đương thêm hạng phạt đạo hàm output theo input); **adversarial training** là biến thể cực đoan (chủ động tìm nhiễu nhỏ gây thay đổi output lớn); nhiễu vào **trọng số** → hội tụ về cực tiểu ở giữa vùng phẳng rộng (đổi trọng số chút ít không sao); nhiễu vào **nhãn** → **label smoothing**: thay vì ép xác suất lớp đúng = 1 (đẩy pre-softmax tới cực trị, quá tự tin), cho nhãn đúng xác suất 1−ρ, các lớp còn lại chia đều ρ — cải thiện tổng quát hoá trong nhiều tình huống.
- **Bayesian inference:** thay vì chọn một bộ tham số khả dĩ nhất (MLE — quá tự tin), coi tham số là biến chưa biết, tính phân phối hậu nghiệm Pr(φ|dữ liệu) bằng quy tắc Bayes, dự đoán = trung bình có trọng số trên mọi tham số (ensemble vô hạn). Đẹp về lý thuyết, cho cả dự đoán lẫn độ không chắc chắn — nhưng với mạng nơ-ron, không có cách thực tế nào biểu diễn toàn bộ phân phối hay tích phân trên đó; mọi phương pháp hiện tại đều xấp xỉ và phức tạp.
- **Transfer learning:** dữ liệu ít → pre-train trên task phụ có nhiều dữ liệu, rồi bỏ tầng cuối, thêm tầng mới cho task chính; chỉ train tầng mới hoặc fine-tune toàn bộ. Mạng đã học được biểu diễn nội bộ tốt từ task phụ — tương đương khởi tạo tham số ở vùng hợp lý.
- **Multi-task learning:** train một mạng làm nhiều task cùng lúc (vd từ một ảnh: phân đoạn + ước lượng độ sâu + chú thích) — các task cùng cần hiểu ảnh nên học chung giúp từng task tốt lên.
- **Self-supervised learning:** tạo nhãn "miễn phí" từ dữ liệu không nhãn để pre-train rồi transfer: **generative** — che một phần dữ liệu, bắt mạng đoán phần thiếu (inpaint ảnh, đoán từ bị che trong câu — nền tảng của mô hình ngôn ngữ chương 12); **contrastive** — phân biệt cặp ví dụ có liên quan (hai bản biến đổi của cùng một ảnh) với cặp không liên quan.
- **Data augmentation:** biến đổi mỗi ví dụ train mà nhãn không đổi (ảnh: xoay, lật, mờ, đổi màu; text: thay từ đồng nghĩa, dịch đi dịch lại; audio: tăng/giảm dải tần) — dạy mô hình **bất biến** trước các biến đổi không liên quan.

### 9.4 Tóm tắt — 4 nguyên lý chung

Mọi phương pháp cải thiện tổng quát hoá đều thuộc một trong 4 nguyên lý: **(i) ép hàm mượt hơn** (vd L2 regularization), **(ii) tăng lượng dữ liệu** (vd augmentation), **(iii) kết hợp nhiều mô hình** (vd ensembling), **(iv) tìm cực tiểu rộng/phẳng hơn** (vd thêm nhiễu vào trọng số).

---

## Điểm rút ra thực hành (từ phần 2)

- **Loss không phải chọn bừa:** mỗi loại bài toán có loss "đúng" của nó từ maximum likelihood — hồi quy → least squares (MSE), phân loại nhị phân → binary cross-entropy (sigmoid), đa lớp → cross-entropy (softmax). Dùng sai loss là sai ngay từ định nghĩa bài toán.
- **Adam là lựa chọn mặc định hợp lý** cho hầu hết huấn luyện mạng nơ-ron: ít nhạy learning rate, cân bằng các tầng. SGD + momentum vẫn là đối thủ khi cần tinh chỉnh tối đa.
- **Khởi tạo He (σ² = 2/số unit đầu vào)** là lý do mạng ReLU sâu huấn luyện được ngay từ đầu — PyTorch `kaiming_normal_` làm sẵn việc này; khởi tạo sai → vanishing/exploding gradient trước khi học được gì.
- **Đừng tin lỗi train:** lỗi train → 0 không có nghĩa gì nếu không có validation/test. Overfitting xuất hiện đúng lúc loss test bắt đầu tăng trong khi loss train vẫn giảm.
- **Double descent thay đổi trực giác cũ:** "mô hình càng lớn càng overfit" chỉ đúng tới vùng critical; vượt qua đó, thêm capacity lại giúp — nhưng không có công thức cho biết cần thêm bao nhiêu, vẫn phải thử bằng validation set.
- **Regularization là bộ công cụ, không phải một kỹ thuật:** L2 + dropout + early stopping + augmentation thường dùng cùng nhau; mỗi cái tấn công overfitting theo một nguyên lý khác nhau trong 4 nguyên lý ở 9.4.
- **Notebook thực hành:** 5.1 Least squares loss · 5.2 Binary cross-entropy · 5.3 Multiclass cross-entropy · 6.1 Line search · 6.2 Gradient descent · 6.3 SGD · 6.4 Momentum · 6.5 Adam · 7.1 Backprop (toy) · 7.2 Backprop · 7.3 Initialization · 8.1 MNIST-1D performance · 8.2 Bias-variance trade-off · 8.3 Double descent · 8.4 High-dimensional spaces · 9.1 L2 regularization · 9.2 Implicit regularization · 9.3 Ensembling · 9.4 Bayesian approach · 9.5 Augmentation — xem tại https://udlbook.github.io/udlbook/.

**Tiếp theo:** phần 3 (chương 10–13: shallow networks, deep networks, residual networks, transformers — các kiến trúc).
