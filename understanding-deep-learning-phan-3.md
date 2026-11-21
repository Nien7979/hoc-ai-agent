# S1 — Understanding Deep Learning (Simon J.D. Prince): Phần 3 — Chương 10–13

> Sách: **Understanding Deep Learning** — Simon J.D. Prince, MIT Press 2023, 544 trang, **PDF MIỄN PHÍ** (CC BY-NC-ND, open-access). Bản đang đọc: release v5.0.3 (08/02/2026), 541 trang, tải từ https://github.com/udlbook/udlbook/releases (file `UnderstandingDeepLearning_02_09_26_C.pdf`, 22 MB, lưu tại `~/workspace/knowledge/understanding-deep-learning/`).
> Sách có 21 chương + 68 notebook Python thực hành (https://udlbook.github.io/udlbook/). Mục tiêu của sách: hiểu **nguyên lý** đằng sau deep learning — không quá lý thuyết (không có chứng minh), cũng không quá thực hành (gần như không có code).
> Ghi chú này: phần 3 (chương 10–13, trang sách 161–261). Phần 1 (chương 1–4) và phần 2 (chương 5–9) ở cùng thư mục `hoc-ai-agent/`. Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong tài liệu gốc.

---

## Chương 10 — Convolutional networks (Mạng tích chập)

### 10.1 Bất biến vs đồng biến

- **Bất biến (invariant):** f[t[x]] = f[x] — output không đổi khi input bị biến đổi. Phân loại ảnh cần bất biến với tịnh tiến: dịch ảnh "núi" sang ngang vẫn phải nhận ra là "núi".
- **Đồng biến (equivariant):** f[t[x]] = t[f[x]] — output biến đổi cùng cách với input. Phân đoạn ảnh theo từng pixel cần đồng biến: dịch ảnh thì bản đồ phân đoạn cũng phải dịch theo.
- Mạng tích chập (convolutional) được thiết kế để **đồng biến với tịnh tiến**; pooling tạo ra một phần tính **bất biến** với tịnh tiến nhỏ.

### 10.2 Tích chập 1D

- **Phép tích chập:** mỗi output zᵢ là tổng có trọng số của các input gần đó, **cùng một bộ trọng số (kernel/filter) dùng ở mọi vị trí**: zᵢ = ω₁xᵢ₋₁ + ω₂xᵢ + ω₃xᵢ₊₁ (kernel size 3). Dịch input thì output dịch theo cùng cách → đồng biến với tịnh tiến. (Chú thích: theo đúng định nghĩa toán học đây là cross-correlation chứ không phải convolution — trọng số không bị lật — nhưng quy ước trong machine learning gọi nó là convolution.)
- **Padding:** ở biên, kernel tràn ra ngoài input. Hai cách: **zero-padding** (coi ngoài biên = 0, output giữ nguyên kích thước) hoặc **valid convolution** (chỉ tính nơi kernel nằm gọn trong input, output nhỏ lại).
- **Stride:** bước nhảy của kernel. Stride 2 → output còn khoảng một nửa. **Kernel size:** càng lớn càng tích hợp vùng rộng hơn (thường lẻ để có tâm); nhược điểm là nhiều trọng số hơn. **Dilation (atrous — "có lỗ"):** chèn số 0 giữa các trọng số của kernel → tích hợp vùng rộng mà không thêm trọng số (dilation rate = số 0 chèn giữa các trọng số + 1).
- **Tầng tích chập = trường hợp riêng của tầng fully connected:** hᵢ = a[β + Σⱼ ωⱼxᵢ₊ⱼ₋₂]. Một tầng fully connected D input → D hidden unit cần D² trọng số + D bias; tầng tích chập kernel 3 chỉ cần **3 trọng số + 1 bias**. FC có thể tái tạo chính xác tích chập nếu đặt phần lớn trọng số = 0 và ép các trọng số còn lại bằng nhau (ma trận trọng số thưa + chia sẻ giá trị). → Tích chập là **ràng buộc mạnh**: ép mạng xử lý mọi vị trí theo cùng một cách.
- **Channels (kênh / feature map):** một phép tích chập duy nhất dễ mất thông tin (trung bình hoá + ReLU cắt phần âm) → chạy **nhiều tích chập song song**, mỗi cái cho một kênh. Với Cᵢ kênh vào, kernel K, Cₒ kênh ra: cần Cᵢ×Cₒ×K trọng số + Cₒ bias. Tầng sau: mỗi kênh ra là tổng có trọng số trên **mọi kênh vào** tại K vị trí gần nhất.
- **Receptive field (trường tiếp nhận):** vùng input gốc "nuôi" một hidden unit. Kernel 3: tầng 1 có receptive field 3; tầng 2: 5; tầng 3 (stride 2): 7… — càng sâu, mỗi unit tích hợp thông tin từ vùng input càng rộng.
- **Ví dụ MNIST-1D:** mạng tích chập 3 tầng (15 kênh, kernel 3, stride 2) có **2.050 tham số**, lỗi test ~17%; mạng fully connected cùng số tầng/unit có **59.065 tham số**, lỗi test ~40% (dù cả hai đều fit train hoàn hảo). Không phải do ít tham số hơn (chương 8 đã chỉ ra quá tham số thường giúp ích) — mà do **inductive bias** của tích chập: dữ liệu được tạo bằng cách dịch ngẫu nhiên template chữ số, nên ép mạng xử lý mọi vị trí giống nhau là kiến thức tiên nghiệm đúng. Nói cách khác, tích chập là một dạng **regularizer** — phạt vô hạn hầu hết nghiệm mà FC có thể mô tả.

### 10.3 Tích chập 2D (ảnh)

- Kernel 2D 3×3: hᵢⱼ = a[β + ΣₘΣₙ ωₘₙ·xᵢ₊ₘ₋₂,ⱼ₊ₙ₋₂] — tổng có trọng số trên vùng vuông 3×3, trượt kernel theo cả hai chiều.
- Ảnh RGB = tín hiệu 2D với **3 kênh**: kernel 3×3 có 3×3×3 trọng số, áp lên 3 kênh tại mỗi vị trí 3×3. Kernel K×K, Cᵢ kênh vào, Cₒ kênh ra → Cᵢ×Cₒ×K×K trọng số + Cₒ bias.

### 10.4 Downsampling, upsampling, đổi số kênh

- **Downsampling (giảm kích thước, thường ×1/2 mỗi chiều):** (i) stride 2 (lấy mẫu mỗi vị trí thứ hai, gộp với tích chập); (ii) **max pooling** — giữ giá trị lớn nhất trong khối 2×2 (tạo một phần bất biến với tịnh tiến 1 pixel: dịch input 1 pixel, nhiều giá trị max vẫn giữ nguyên); (iii) mean pooling — trung bình khối 2×2. Áp riêng cho từng kênh.
- **Upsampling (tăng kích thước, dùng khi output cũng là ảnh):** (i) nhân 4 mỗi giá trị; (ii) **max unpooling** — trả giá trị về đúng vị trí nó đến từ (cần nhớ vị trí max lúc pooling); (iii) nội suy song tuyến (bilinear); (iv) **transposed convolution** — đảo ngược của stride-2: mỗi input đóng góp vào 3 output, ma trận trọng số là chuyển vị của ma trận downsampling. (Lưu ý: transposed conv có thể tạo artifact bàn cờ — Odena et al. 2016.)
- **1×1 convolution:** đổi số kênh mà không đổi kích thước không gian — mỗi output là tổng có trọng số của **mọi kênh tại cùng một vị trí**. Tương đương chạy cùng một mạng fully connected nhỏ trên các kênh tại mỗi vị trí. Thường dùng để gộp/kết hợp các nhánh song song (chương 11).

### 10.5 Ứng dụng

- **Phân loại ảnh — ImageNet:** 1.281.167 ảnh train, 50.000 validation, 100.000 test, 1000 lớp. **AlexNet (2012):** 8 tầng (5 tích chập + 3 FC), ReLU, kernel 11×11 stride 4 → 96 kênh, max pool, kernel 5×5, rồi 3 tầng 3×3, FC 4096–4096–1000, softmax; ~60 triệu tham số (phần lớn ở FC); augmentation ×2048, SGD momentum 0.9, dropout ở FC, weight decay; **16,4% top-5 / 38,1% top-1** — bước nhảy vọt, khởi đầu kỷ nguyên deep learning hiện đại. **VGG (2014):** 19 tầng, 144 triệu tham số, chỉ dùng kernel 3×3 xen kẽ max pool (kích thước không gian giảm dần, số kênh tăng dần); **6,8% top-5 / 23,7% top-1**. Bài học: **độ sâu quan trọng** — nhưng chỉ tới một giới hạn (xem chương 11).
- **Phát hiện vật thể — YOLO:** ảnh 448×448 qua 24 tầng tích chập → lưới 7×7; mỗi ô lưới dự đoán lớp vật thể + 2 bounding box (tâm x,y, cao, rộng, độ tin cậy); sau đó loại box tin cậy thấp và khử trùng lặp (chỉ giữ box tin cậy nhất cho mỗi vật). Dùng transfer learning: pre-train trên ImageNet rồi fine-tune.
- **Phân đoạn ngữ nghĩa:** gán nhãn cho **từng pixel** (21 lớp). Mạng của Noh et al. (2015): nửa đầu là VGG thu nhỏ (encoder — downsample tới 14×14 rồi FC thành vector 4096 chứa thông tin toàn ảnh), nửa sau "giải mã" ngược bằng max unpooling + transposed convolution (decoder) lên lại 224×224×21, softmax tại mỗi vị trí. Kiến trúc encoder–decoder còn gọi là **hourglass** (hình đồng hồ cát).

**Tóm tắt chương 10:** tầng tích chập = tổng có trọng số của input lân cận + bias + activation, trọng số chia sẻ ở mọi vị trí → ít tham số, không tăng theo kích thước ảnh, đồng biến với tịnh tiến. Mạng điển hình: tích chập xen kẽ downsampling ×1/2, kích thước giảm dần, kênh tăng dần, cuối cùng là FC tổng hợp toàn cục; nếu output là ảnh thì có decoder upsample ngược. Tăng độ sâu giúp tới một điểm — sâu hơn nữa thì khó train, đó là động lực của chương 11.

---

## Chương 11 — Residual networks (Mạng dư / kết nối tắt)

### 11.1 Giới hạn của xử lý tuần tự

- Mọi mạng tới giờ đều xử lý **tuần tự**: h₁ = f₁[x], h₂ = f₂[h₁], h₃ = f₃[h₂], y = f₄[h₃] — hàm lồng nhau.
- Thêm tầng giúp cải thiện (VGG 19 tầng > AlexNet 8 tầng) — nhưng thêm nữa thì hiệu năng **giảm lại**, trên cả train lẫn test (hình 11.2: mạng 56 tầng tệ hơn mạng 20 tầng trên CIFAR-10). Vì lỗi train cũng tăng → đây là **vấn đề huấn luyện**, không phải tổng quát hoá.
- **Giả thuyết — shattered gradients (gradient vỡ vụn):** sau khởi tạo, gradient của loss theo tham số tầng đầu thay đổi **không thể đoán trước** khi thay đổi tham số một bước hữu hạn (đạo hàm giả định thay đổi vô cùng nhỏ, còn SGD đi bước hữu hạn). Mặt loss trông như dãy núi nhỏ li ti chứ không phải một cấu trúc mượt. Đo bằng hàm tự tương quan của gradient: mạng nông — gradient tại các input gần nhau tương quan cao; mạng sâu 24 tầng — tương quan rơi về 0 rất nhanh. Nguyên nhân: đạo hàm ∂y/∂f₁ là tích của chuỗi đạo hàm qua mọi tầng; đổi f₁ thì mọi đạo hàm trong chuỗi đều được tính tại vị trí hơi khác → gradient sau mỗi bước cập nhật có thể hoàn toàn khác.

### 11.2 Residual connections (kết nối dư / skip connection)

- **Định nghĩa:** h₁ = x + f₁[x]; h₂ = h₁ + f₂[h₁]; … — đầu vào của mỗi tầng được **cộng trở lại** với đầu ra của nó. Mỗi hàm f_k học một **thay đổi cộng thêm** vào biểu diễn hiện tại (input và output phải cùng kích thước). Mỗi khối cộng này gọi là **residual block**.
- **"Mở" mạng ra:** output cuối = input + tổng của 4 mạng nhỏ hơn (mỗi mạng tương ứng một f_k áp trên các tổ hợp trước đó). Nói cách khác, mạng dư là một **ensemble của các mạng nhỏ** cộng lại; hoặc có 2⁴ = 16 đường đi khác nhau từ input tới output với độ dài khác nhau.
- **Vì sao gradient tốt hơn:** đạo hàm ∂y/∂f₁ giờ có thêm **hạng đơn vị (identity)** — thay đổi ở tầng đầu đóng góp **trực tiếp** vào output, không chỉ qua chuỗi dài. Gradient qua các đường ngắn thì "ngoan" hơn → mạng dư ít bị shattered gradients.
- **Thứ tự phép toán quan trọng:** nếu dùng thứ tự thường (tuyến tính → ReLU), output của khối luôn ≥ 0 → khối dư chỉ có thể **cộng thêm** số không âm. Vì vậy thường đảo thứ tự: **ReLU trước, rồi biến đổi tuyến tính** — khối có thể cộng cả số âm lẫn dương. (Phải bắt đầu mạng bằng một biến đổi tuyến tính, vì nếu input toàn âm thì ReLU đầu tiên sẽ cắt sạch tín hiệu.) Trong thực tế một residual block thường chứa nhiều tầng, kết thúc bằng biến đổi tuyến tính.

### 11.3 Vấn đề phương sai trong mạng dư + BatchNorm

- Mạng dư không bị vanishing gradient (luôn có đường trực tiếp tới output) nhưng **phương sai activation tăng gấp đôi mỗi block**: với He init, khối xử lý giữ nguyên phương sai kỳ vọng; cộng trở lại với input (hai nguồn biến thiên không tương quan) → phương sai ×2 mỗi tầng → tăng theo hàm mũ theo số block → tràn số thực; tương tự với gradient ở backward pass.
- Cách sửa thô: nhân output mỗi block với 1/√2. Cách chuẩn: **batch normalization**.
- **BatchNorm:** với mỗi activation h, tính trung bình m_h và độ lệch chuẩn s_h **trên batch**, chuẩn hoá về trung bình 0 phương sai 1, rồi **co giãn + dịch chuyển bằng tham số học được γ, δ**: h ← γ·(h−m_h)/(s_h+ε) + δ. Áp độc lập cho mỗi hidden unit (mạng tích chập: thống kê tính trên cả batch lẫn vị trí không gian; mỗi kênh có một cặp γ, δ riêng). Lúc test không có batch → dùng thống kê tính trên **toàn bộ tập train** và đóng băng lại.
- **Lợi ích:** (i) **forward ổn định** — với δ=0, γ=1, phương sai mỗi tầng = 1; trong mạng dư phương sai chỉ tăng **tuyến tính** theo số block (không còn hàm mũ); tác dụng phụ: lúc khởi tạo, các tầng sau gần như tính hàm identity → mạng "nông hơn" lúc đầu, tự tăng độ sâu hiệu dụng khi train (tăng γ ở tầng sau). (ii) **Cho phép learning rate cao hơn** — BatchNorm làm mặt loss và gradient thay đổi mượt hơn (giảm shattered gradients); mà chương 9 đã chỉ ra LR cao hơn thì test tốt hơn. (iii) **Regularization** — chuẩn hoá phụ thuộc thống kê batch → activation của một ví dụ bị nhiễu bởi các ví dụ khác trong batch, khác nhau mỗi lần lặp. (iv) Bất biến với việc co giãn trọng số/bias (nhân đôi trọng số → activation nhân đôi → s_h nhân đôi → chuẩn hoá bù lại). **Giá phải trả:** dư thừa tham số (thêm γ, δ cho mỗi unit) + nhiều họ trọng số cho cùng một hiệu ứng.

### 11.5 Các kiến trúc dư nổi tiếng

- **ResNet:** mỗi block = BN → ReLU → conv 3×3, lặp lại 2 lần rồi cộng về input. **Bottleneck block** (mạng rất sâu): conv 1×1 giảm kênh → conv 3×3 trên biểu diễn nhỏ → conv 1×1 tăng kênh lại — tích hợp vùng 3×3 với ít tham số hơn. **ResNet-200:** 200 tầng, bắt đầu bằng conv 7×7 stride 2 + max pool, các bottleneck block xen kẽ downsampling (stride 2) + tăng kênh, cuối cùng average pool + FC → 1000 lớp; **4,8% top-5 / 20,1% top-1 trên ImageNet — mô hình đầu tiên vượt con người (5,1% top-5)**. (Lưu ý: tới lúc sách viết, các mô hình tốt nhất đã là transformer — chương 12.)
- **DenseNet:** thay vì cộng, **nối (concatenate)** output của **mọi tầng trước** vào input tầng hiện tại — tái sử dụng xử lý của tầng trước linh hoạt hơn, hiệu năng tương đương/better ResNet với cùng số tham số. Nhược điểm: số kênh phình nhanh → phải dùng bottleneck 1×1 để nén; chuỗi nối bị đứt khi downsampling (kích thước không gian khác nhau).
- **U-Net:** encoder–decoder cho phân đoạn ảnh y tế (tế bào HeLa): encoder downsample, decoder upsample bằng transposed conv, **nối biểu diễn encoder ở mỗi tỉ lệ vào decoder ở cùng tỉ lệ** (skip connection) để giữ chi tiết độ phân giải cao mà bottleneck ở giữa không "nhớ" nổi. Hoàn toàn tích chập nên sau khi train có thể chạy trên ảnh kích thước bất kỳ.
- **Stacked hourglass (ước lượng tư thế người):** chuỗi các khối hourglass (encoder–decoder, skip connection **cộng** thay vì nối, có xử lý thêm trong skip link); output là một **heatmap cho mỗi khớp**, vị trí khớp = điểm cực đại của heatmap.

### 11.6 Vì sao mạng dư tốt đến vậy?

- Ban đầu tưởng nhờ độ sâu — nhưng bằng chứng ngược: (i) mạng dư **nông nhưng rộng** đôi khi thắng mạng sâu nhưng hẹp với cùng số tham số; (ii) gradient không lan truyền hiệu quả qua các đường rất dài trong mạng đã "mở" — mạng rất sâu thực chất hoạt động như **tổ hợp các mạng nông hơn**.
- Quan điểm hiện tại: kết nối dư **tự nó đã có giá trị** — mặt loss quanh cực tiểu của mạng dư **mượt và dễ đoán hơn** hẳn cùng mạng đó khi bỏ skip connection (hình 11.13) → dễ học nghiệm tốt, nghiệm robust hơn trước sai số nhỏ của tham số → tổng quát hoá tốt hơn.

**Tóm tắt chương 11:** mạng sâu tuần tự khó train vì shattered gradients. Kết nối dư (cộng input vào output mỗi khối) tạo đường tắt cho gradient + biến mạng thành ensemble các mạng nhỏ → train được mạng rất sâu. Giá phải trả là phương sai tăng gấp đôi mỗi block → xử lý bằng BatchNorm (chuẩn hoá theo batch + γ, δ học được), vừa ổn định forward/backward, vừa cho LR cao hơn, vừa regularize nhẹ. ResNet/DenseNet/U-Net đều là biến thể của ý tưởng này.

---

## Chương 12 — Transformers

### 12.1 Động lực: xử lý văn bản

- Ba quan sát: (i) input có thể rất lớn (37 từ × embedding 1024 = 37.888 số) và **độ dài thay đổi** theo từng câu → mạng fully connected không khả thi; cần **chia sẻ tham số** qua các vị trí (như tích chập chia sẻ qua vị trí ảnh). (ii) Ngôn ngữ **mơ hồ**: "it" trong "The restaurant refused to serve me… because it only cooks vegetarian food" phải nối với "restaurant" chứ không phải "ham sandwich" — cần **kết nối giữa các từ**, độ mạnh phụ thuộc vào chính các từ, và trải dài qua nhiều câu (từ "their" ở câu cuối cũng chỉ nhà hàng). (iii) Các kết nối này phải được học từ dữ liệu.

### 12.2 Dot-product self-attention (tự chú ý)

- **Ý tưởng:** một khối self-attention nhận N input x₁…x_N (mỗi cái D chiều, vd mỗi từ) và trả về N output cùng kích thước. Mỗi output là **tổng có trọng số** của các **value** v_m = β^v + Ω^v·x_m (cùng một phép biến đổi tuyến tính cho mọi input): sa_n = Σ_m a[x_m, x_n]·v_m. Các trọng số chú ý a[•, x_n] **không âm, tổng = 1** — self-attention là cơ chế **định tuyến (routing)** các value theo tỉ lệ khác nhau cho mỗi output.
- **Tính trọng số chú ý:** từ mỗi input tính thêm **query** q_n = β^q + Ω^q·x_n và **key** k_m = β^k + Ω^k·x_m; trọng số = softmax trên tích vô hướng: a[x_m, x_n] = exp[k_mᵀq_n] / Σ_{m'} exp[k_{m'}ᵀq_n]. Tích vô hướng đo **độ tương tự** giữa query và key; softmax khiến các key **cạnh tranh** nhau để đóng góp. Đây là một **hypernetwork**: một nhánh mạng tính ra trọng số cho nhánh khác — nên toàn bộ phép tính là **phi tuyến** dù không có hàm activation nào.
- **Dạng ma trận:** xếp N input thành cột của ma trận D×N là X; V = β^v·1ᵀ + Ω^v·X (tương tự Q, K); Sa[X] = V·Softmax[KᵀQ] (softmax theo từng cột). Số trọng số chú ý tăng theo **bậc hai** của độ dài chuỗi N, độc lập với D.
- **Yêu cầu ban đầu được thỏa:** (i) một bộ tham số duy nhất, không phụ thuộc N → áp được cho chuỗi dài bất kỳ; (ii) có kết nối giữa các từ, độ mạnh phụ thuộc chính các từ đó.

### 12.3 Ba mở rộng gần như luôn dùng

- **Positional encoding (mã hoá vị trí):** self-attention **đồng biến với hoán vị** — không biết thứ tự từ ("The woman ate the raccoon" ≠ "The raccoon ate the woman"). Hai cách: **tuyệt đối** — cộng ma trận Π vào X, mỗi cột khác nhau (sinusoid định sẵn hoặc học được; có thể cộng vào input hoặc vào Q, K nhưng không vào V); **tương đối** — học một tham số π_{a,b} cho mỗi độ lệch vị trí giữa key a và query b rồi sửa ma trận chú ý.
- **Scaled dot-product:** tích vô hướng có thể rất lớn → softmax bão hoà (giá trị lớn nhất áp đảo, gradient ~0, khó train). Chia cho √D_q (D_q = chiều của query/key): Sa[X] = V·Softmax[KᵀQ/√D_q].
- **Multi-head attention (đa đầu):** chạy **H cơ chế self-attention song song** (mỗi đầu có Q, K, V riêng, thường chiều D/H), nối dọc output các đầu rồi qua một biến đổi tuyến tính Ω^c để gộp lại. Đa đầu dường như cần thiết để self-attention hoạt động tốt (có giả thuyết: giúp robust trước khởi tạo xấu; nhiều đầu có thể cắt bỏ sau khi train mà hiệu năng không giảm mấy).

### 12.4 Transformer layer (tầng transformer)

- Một tầng gồm: (1) **multi-head self-attention** (cho các từ tương tác) — dạng residual: X ← X + MhSa[X]; (2) **LayerNorm** — giống BatchNorm nhưng chuẩn hoá **mỗi embedding riêng lẻ** bằng thống kê trên D chiều của chính nó (không phụ thuộc batch); (3) **MLP** (mạng fully connected) áp **riêng cho từng vị trí từ** — dạng residual: x_n ← x_n + mlp[x_n]; (4) LayerNorm lần nữa. Input và output tầng cùng kích thước D×N. Mạng thật = chuỗi nhiều tầng như vậy.

### 12.5 Pipeline NLP: tokenization → embedding → transformer

- **Tokenization:** tách text thành token từ một từ vựng. Từ nguyên gặp 3 khó: từ ngoài từ vựng (tên riêng), dấu câu quan trọng, các biến thể hậu tố (walk/walks/walked) không thấy liên quan. Thực tế dùng **sub-word tokenizer** (vd **byte pair encoding — BPE**): bắt đầu từ ký tự, tham lam **gộp cặp token kề nhau xuất hiện thường xuyên nhất** cho tới khi đủ kích thước từ vựng — kết quả là hỗn hợp chữ cái, mảnh từ, và từ thường gặp; từ hiếm được ghép từ các mảnh.
- **Embedding:** mỗi token trong từ vựng V (|V| ~30.000) ánh xạ tới một vector embedding học được, lưu trong ma trận Ω_e ∈ R^{D×|V|} (D thường 1024). Input one-hot T → X = Ω_e·T.
- Ba loại mô hình transformer: **encoder** (biến embedding thành biểu diễn cho nhiều task), **decoder** (dự đoán token tiếp theo), **encoder-decoder** (chuỗi → chuỗi, vd dịch máy).

### 12.6 Encoder — BERT

- **BERT:** vocab 30.000, embedding 1024 chiều, **24 tầng transformer, 16 head**, Q/K/V mỗi đầu 64 chiều, MLP ẩn 4096 chiều; **~340 triệu tham số** (lúc ra mắt là lớn, nay nhỏ hơn nhiều so với SOTA).
- **Pre-training (tự giám sát):** che ngẫu nhiên một số token bằng `<mask>`, bắt mạng đoán từ bị che (dùng cả ngữ cảnh trái lẫn phải); train trên kho ngữ liệu 3,3 tỉ từ, tối đa 512 token/input, batch 256, 1 triệu bước (~50 epoch). Nhiệm vụ này ép mạng học cú pháp + một phần "thường thức" (vd sau "The <mask> pulled into the station", "train" có xác suất cao hơn "peanut"). Nhược điểm: lãng phí dữ liệu (7 token xử lý chỉ cho 2 hạng loss).
- **Fine-tuning:** gắn thêm một tầng nhỏ lên encoder, train trên dữ liệu có nhãn ít hơn: **phân loại text** (dùng embedding của token đặc biệt `<cls>` ở đầu chuỗi → sigmoid → binary cross-entropy, vd phân tích cảm xúc); **phân loại từng từ** (NER: mỗi embedding → softmax trên các loại thực thể — người/địa điểm/tổ chức); **dự đoán đoạn text** (SQuAD: nối câu hỏi + đoạn Wikipedia, mỗi token cho 2 số — khả năng là điểm bắt đầu/kết thúc đáp án).

### 12.7 Decoder — GPT-3 (mô hình ngôn ngữ tự hồi quy)

- **Autoregressive:** xác suất cả câu phân rã thành tích các xác suất có điều kiện: Pr(t₁…t_N) = Pr(t₁)·∏ Pr(t_n|t₁…t_{n−1}). Mô hình học Pr(t_n | các token trước).
- **Masked self-attention:** train hiệu quả bằng cách đưa cả câu vào một lượt — nhưng token đang xét sẽ "nhìn trộm" đáp án và ngữ cảnh phải. Sửa bằng cách đặt các tích vô hướng tương ứng thành **−∞ trước softmax** → trọng số chú ý về 0. Mỗi vị trí output đại diện cho một câu dở dang, mục tiêu là dự đoán token tiếp theo (multiclass cross-entropy tại mọi vị trí — mọi từ đều đóng góp vào loss, hiệu quả dữ liệu hơn BERT).
- **Sinh text:** đưa chuỗi mồi vào, lấy phân phối token tiếp theo, chọn token khả dĩ nhất hoặc lấy mẫu, nối vào chuỗi, lặp lại. Tính toán hiệu quả vì embedding trước không phụ thuộc token sau (nhờ mask) nên tái sử dụng được phần lớn phép tính. Chiến lược giải mã: **beam search** (giữ nhiều giả thuyết hoàn thiện câu song song để tìm chuỗi tổng thể khả dĩ nhất), **top-k sampling** (chỉ lấy mẫu trong K token khả dĩ nhất, tránh đuôi dài xác suất thấp dẫn vào ngõ cụt ngôn ngữ).
- **GPT-3:** chuỗi 2048 token, batch 3,2 triệu token, **96 tầng, embedding 12.288 chiều, 96 head, 175 tỉ tham số**, train trên **300 tỉ token**. Tính chất bất ngờ: **few-shot learning** — đưa vài cặp ví dụ hỏi/đáp đúng làm ngữ cảnh, mô hình thường trả lời đúng câu hỏi mới mà không cần fine-tune (sửa ngữ pháp, sinh code, làm toán, dịch, trả lời câu hỏi). Hiệu năng thực tế thất thường; chưa rõ là ngoại suy thật hay chỉ nội suy/sao chép.

### 12.8 Encoder-decoder — dịch máy

- Encoder xử lý câu nguồn (vd tiếng Anh) thành biểu diễn; decoder sinh câu đích (tiếng Pháp) bằng masked self-attention + dự đoán từ tiếp theo — và **chú ý tới output của encoder** qua **cross-attention**: query tính từ embedding của decoder, key và value tính từ embedding của encoder. Mỗi từ đích do đó phụ thuộc cả vào các từ đích trước đó lẫn toàn bộ câu nguồn.

### 12.9–12.10 Giới hạn và transformer cho ảnh

- **Độ phức tạp bậc hai:** mỗi token tương tác với mọi token khác → chi phí tăng theo N² (decoder: ~N²/2). Cách mở rộng cho chuỗi dài: **thưa hoá ma trận tương tác** — cấu trúc tích chập (mỗi token chỉ tương tác vài token kề; qua nhiều tầng, receptive field mở rộng; có thể dùng dilation), cho một số token đặc biệt tương tác với tất cả (như `<cls>`), hoặc thêm **global token** kết nối mọi token.
- **Transformer cho ảnh — tưởng chừng bất lợi:** ảnh có nhiều pixel hơn nhiều so với từ trong câu (N² là nút thắt), và CNN có inductive bias tốt (đồng biến tịnh tiến, cấu trúc 2D) mà transformer phải học lại. Dù vậy transformer đã **vượt CNN** ở phân loại ảnh nhờ quy mô và dữ liệu pre-train khổng lồ.
  - **ImageGPT:** decoder tự hồi quy trên pixel (ảnh 64×64, màu RGB 24-bit lượng tử hoá thành 9-bit = 512 token); sinh ảnh pixel-by-pixel; embedding cuối trung bình hoá → phân loại (27,4% top-1 ImageNet — tệ hơn CNN thời đó nhưng ấn tượng với ảnh nhỏ).
  - **Vision Transformer (ViT):** chia ảnh thành **patch 16×16**, mỗi patch chiếu tuyến tính thành một embedding, đưa vào transformer encoder với token `<cls>`; pre-train có giám sát trên 303 triệu ảnh có nhãn (18.000 lớp), fine-tune cho task cuối: **11,45% top-1 ImageNet**. Inductive bias mạnh của CNN chỉ bị vượt qua khi dùng dữ liệu cực lớn.
  - **SWin transformer (đa tỉ lệ):** chia ảnh thành cửa sổ, self-attention chỉ trong từng cửa sổ; **dịch cửa sổ ở tầng xen kẽ** để thông tin lan khắp ảnh; định kỳ gộp patch 2×2 (giảm độ phân giải, tăng kênh) — giống CNN; **9,89% top-1**. **DaViT:** xen kẽ attention theo patch và attention theo kênh — **9,60% top-1**, gần SOTA lúc sách viết.

**Tóm tắt chương 12:** self-attention = mỗi output là tổng có trọng số (softmax trên tích vô hướng query–key) của các value — một cơ chế định tuyến mà trọng số phụ thuộc chính dữ liệu. Cộng positional encoding + multi-head + residual + LayerNorm + MLP theo từng vị trí = tầng transformer. Encoder (BERT: đoán từ bị che → fine-tune), decoder (GPT: dự đoán token tiếp theo, masked attention, few-shot), encoder-decoder (dịch máy, cross-attention). Chi phí O(N²) là giới hạn chính; với ảnh, transformer cần patch + dữ liệu pre-train khổng lồ mới vượt được CNN.

---

## Chương 13 — Graph neural networks (Mạng nơ-ron trên đồ thị)

### 13.1 Đồ thị là gì

- Đồ thị = tập **node (đỉnh)** + **edge (cạnh)** nối các cặp node; thường **thưa** (chỉ một phần nhỏ các cạnh có thể tồn tại mới thực sự có mặt).
- Vật thể vốn là đồ thị: mạng đường bộ (node = địa điểm, edge = đường), phân tử hoá học (node = nguyên tử, edge = liên kết), **mạch điện (node = linh kiện/nút nối, edge = dây nối)**.
- Dữ liệu cũng biểu diễn được bằng đồ thị: mạng xã hội (người–bạn bè), trích dẫn khoa học (bài báo–trích dẫn), Wikipedia (bài viết–siêu liên kết), chương trình máy tính (token cú pháp–phép tính), đám mây điểm 3D (điểm–điểm gần nhau), tương tác protein. Một **tập hợp** là đồ thị đầy đủ (mọi thành viên nối với nhau); một **ảnh** là đồ thị có topo đều (mỗi pixel nối với pixel kề).
- **Loại đồ thị:** vô hướng (mạng xã hội — quan hệ đối xứng), có hướng (trích dẫn — một chiều), **knowledge graph** (đa đồ thị có hướng không đồng nhất: node là nhiều loại thực thể, nhiều cạnh khác loại giữa hai node), đồ thị hình học (node có vị trí 3D), đồ thị phân cấp (đồ thị lồng trong đồ thị). Chương này tập trung vào đồ thị **vô hướng**.

### 13.2 Biểu diễn đồ thị

- Ba ma trận: **A — ma trận kề** (N×N, phần tử (m,n)=1 nếu có cạnh m–n; đồ thị vô hướng → đối xứng; đồ thị lớn/thưa thì lưu dạng danh sách cạnh cho tiết kiệm); **X — ma trận node** (D×N, mỗi cột là node embedding độ dài D); **E — ma trận edge** (mỗi cạnh có edge embedding).
- **Tính chất:** A^L tại (m,n) = số **walk** (đường đi, được phép thăm lại node) độ dài L từ m tới n. Nhân vector one-hot của node n với A → các node kề (đi 1 bước); nhân tiếp với A → các node đi 2 bước tới được.
- **Hoán vị chỉ số node:** thứ tự đánh số node là tuỳ ý; hoán vị đổi X' = XP và A' = PᵀAP nhưng đồ thị không đổi. → Mọi phép xử lý trên đồ thị phải **không phụ thuộc thứ tự đánh số** (khác ảnh — hoán vị pixel tạo ảnh khác; khác text — hoán vị từ tạo câu khác).

### 13.3 Task và loss trên đồ thị

- GNN nhận X và A, qua K tầng, node embedding ở đầu chỉ chứa thông tin của chính node; ở cuối chứa cả thông tin node + **ngữ cảnh trong đồ thị** (tương tự embedding từ qua transformer: đầu là từ, cuối là nghĩa của từ trong câu).
- **Graph-level (cả đồ thị):** vd dự đoán nhiệt độ hoá lỏng của phân tử (hồi quy) hay phân tử có độc không (phân loại). Gộp các embedding node (vd trung bình — **mean pooling**), rồi biến đổi tuyến tính → least squares hoặc sigmoid + binary cross-entropy.
- **Node-level (từng node):** vd phân loại mỗi điểm trong đám mây 3D thuộc cánh hay thân máy bay. Loss như trên nhưng độc lập tại mỗi node.
- **Edge prediction (dự đoán cạnh):** vd mạng xã hội có nên gợi ý hai người kết nối? Lấy tích vô hướng hai embedding node → sigmoid → xác suất cạnh tồn tại (phân loại nhị phân).

### 13.4 Graph convolutional network (GCN) — không gian

- **Nguyên tắc thiết kế:** (i) mỗi tầng phải **đồng biến với hoán vị** chỉ số node (H_{k+1}·P = F[H_k·P, PᵀAP]); task graph-level thì output cuối **bất biến** với thứ tự node (mean pooling thỏa điều này); (ii) **chia sẻ tham số** ở mọi node (như tích chập chia sẻ qua vị trí ảnh) — không học riêng cho từng node; (iii) **relational inductive bias** — ưu tiên thông tin từ node kề.
- **Tầng GCN đơn giản:** tại mỗi node n, **tổng hợp (aggregate)** embedding của các node kề: agg[n,k] = Σ_{m∈ne[n]} h_k^(m); rồi biến đổi tuyến tính Ω_k áp cho cả node hiện tại lẫn giá trị tổng hợp, cộng bias, qua activation: h_{k+1}^(n) = a[β_k + Ω_k·h_k^(n) + Ω_k·agg[n,k]]. Dạng ma trận gọn: **H_{k+1} = a[β_k·1ᵀ + Ω_k·H_k·(A + I)]** (cộng I để tính luôn chính node). Mỗi node có số lượng kề khác nhau vẫn xử lý được (không có khái niệm "trên/dưới" như pixel).
- **Ví dụ — phân loại phân tử độc/không độc:** A từ cấu trúc phân tử; X là one-hot nguyên tố (118 nguyên tố); qua K tầng GCN rồi mean pooling → sigmoid → xác suất độc. Train bằng SGD + binary cross-entropy.
- **Batch:** các đồ thị có số node khác nhau → không xếp thành tensor 3D được. Mẹo: coi các đồ thị trong batch là **các thành phần rời nhau của một đồ thị lớn duy nhất**, chạy một lượt; mean pooling tính riêng cho từng đồ thị.

### 13.6 Inductive vs transductive

- **Inductive (quy nạp):** học quy tắc từ tập train có nhãn rồi áp cho dữ liệu test mới (mọi mô hình tới giờ trong sách đều như vậy).
- **Transductive (chuyển đạo):** xét cả dữ liệu có nhãn lẫn **không nhãn cùng lúc**, chỉ cho ra nhãn cho các output chưa biết — không tạo ra quy tắc (còn gọi là **semi-supervised**). Ưu: dùng được mẫu hình trong dữ liệu không nhãn. Nhược: thêm dữ liệu không nhãn mới là phải train lại.
- Graph-level task chỉ xảy ra ở dạng inductive. Node-level và edge prediction có thể ở cả hai dạng: vd đồ thị trích dẫn khoa học khổng lồ, một số node đã biết lĩnh vực, cần gán nhãn các node còn lại — train và test gắn chặt trong cùng một đồ thị.

### 13.7 Node classification trên đồ thị khổng lồ — batch bằng cách nào?

- Hai khó khăn: (i) phải lưu embedding mọi tầng cho cả đồ thị (quá lớn); (ii) chỉ có một đồ thị thì batch là gì?
- **Receptive field trên đồ thị:** node ở tầng k phụ thuộc vào **k-hop neighborhood** (node kề của kề… k lần) — giống receptive field của CNN. Một bước SGD chỉ cần đồ thị hợp của các k-hop neighborhood của các node trong batch.
- **Graph expansion problem:** đồ thị dày + nhiều tầng → receptive field có thể bao cả đồ thị, không giảm được gì. Hai cách xử lý: **neighborhood sampling** — mỗi tầng chỉ lấy mẫu ngẫu nhiên một số lượng cố định node kề (đồ thị vẫn phình theo tầng nhưng có kiểm soát; lấy mẫu mới mỗi batch → có tác dụng regularize giống dropout); **graph partitioning** — cụm đồ thị thành các subgraph rời nhau (tối đa liên kết nội bộ), mỗi subgraph (hoặc tổ hợp vài subgraph) là một batch.
- Sau khi có cách tạo batch, train như inductive bình thường; lúc suy luận chỉ cần forward pass trên k-hop neighborhood của node cần dự đoán (không cần lưu trung gian → tiết kiệm bộ nhớ).

### 13.8 Các biến thể tầng GCN

- **Kết hợp node hiện tại với tổng hợp:** cộng đơn giản (A + I); **diagonal enhancement** — nhân node hiện tại với (1+ε_k) học được trước khi cộng; hoặc dùng **biến đổi tuyến tính riêng Ψ_k** cho node hiện tại (tương đương nối [H_k·A; H_k] rồi nhân một ma trận Ω'_k); **residual connection** — biến đổi phần tổng hợp qua activation rồi nối/cộng với node hiện tại.
- **Cách tổng hợp khác nhau:** **mean aggregation** — trung bình thay vì tổng (độ lớn không phụ thuộc số lượng kề; dùng ma trận bậc D: H_{k+1} = a[β1ᵀ + ΩH(AD⁻¹ + I)]); **Kipf normalization** — chuẩn hoá tổng theo bậc của cả node hiện tại lẫn node kề: agg[n] = Σ h_m/√(|ne[n]|·|ne[m]|) — thông tin từ node có quá nhiều kề bị giảm trọng số (ít độc đáo); dạng ma trận D^(−1/2)·A·D^(−1/2); **max pooling aggregation** — lấy max theo từng phần tử trên các node kề (bất biến với hoán vị).
- **Graph attention (GAT):** trọng số tổng hợp **phụ thuộc dữ liệu** chứ không chỉ topo: biến đổi tuyến tính node, tính độ tương tự s_mn giữa cặp node (nối hai embedding, tích vô hướng với vector tham số học được, qua activation), softmax **chỉ trên node hiện tại + các node kề** (đặt các vị trí không kề thành −∞ trước softmax — giống masked attention). Rất giống self-attention của transformer, khác ở: (i) query/key/value là một (cùng embedding đã biến đổi), (ii) độ đo tương tự khác, (iii) attention bị mask bởi ma trận kề. Cũng mở rộng được thành **multi-head**.
- **Edge graph (đồ thị cạnh / line graph):** để xử lý **edge embedding**, chuyển sang đồ thị mới trong đó mỗi cạnh cũ thành một node, hai cạnh cũ chung một node thì nối nhau trong đồ thị mới. Rồi dùng đúng kỹ thuật GCN như trên. Khi có cả node và edge embedding, chuyển qua lại giữa hai đồ thị: 4 loại cập nhật có thể (node→node, node→edge, edge→node, edge→edge), xen kẽ tuỳ ý.

**Tóm tắt chương 13:** đồ thị = node + edge (+ embedding ở node/cạnh). Thứ tự node là tuỳ ý → tầng GCN phải đồng biến với hoán vị; task graph-level thì bất biến. GCN không gian: mỗi tầng tổng hợp thông tin node kề rồi kết hợp với node hiện tại (chia sẻ tham số, relational inductive bias). Ba task chính: graph-level, node-level, edge prediction. Đồ thị lớn ở dạng transductive cần tạo batch bằng neighborhood sampling hoặc partitioning. Biến thể tầng: mean/Kipf/max aggregation, attention theo dữ liệu (GAT — cầu nối với transformer chương 12), edge graph cho edge embedding.

---

## Điểm rút ra thực hành (từ phần 3)

- **Kiến trúc = kiến thức tiên nghiệm được "đúc" thành mạng:** tích chập ép tính đồng biến tịnh tiến (ảnh), self-attention ép kết nối phụ thuộc dữ liệu (text), GCN ép tính đồng biến hoán vị + ưu tiên node kề (đồ thị). Chọn sai inductive bias thì mạng phải học lại từ đầu những gì kiến trúc đáng lẽ cho miễn phí — ví dụ MNIST-1D: FC 59k tham số thua CNN 2k tham số.
- **Độ sâu có giới hạn của nó:** VGG thắng AlexNet nhờ sâu hơn, nhưng mạng tuần tự quá sâu thì gradient vỡ vụn và train tệ đi — residual connection + BatchNorm là "vé" để đi sâu tiếp (ResNet-200 vượt con người ở ImageNet top-5).
- **BatchNorm và LayerNorm là hai anh em khác cha:** BatchNorm chuẩn hoá theo batch (mỗi unit, thống kê trên batch) — tốt cho CNN/ResNet, có tác dụng regularize nhẹ; LayerNorm chuẩn hoá theo chiều embedding của từng ví dụ — chuẩn cho transformer (nơi độ dài chuỗi thay đổi, batch thống kê không ổn định).
- **Transformer là "máy định tuyến" tổng quát:** cùng một công thức self-attention xài được cho từ (BERT/GPT), patch ảnh (ViT/SWin), và node đồ thị (GAT chỉ thêm mask kề). Hiểu attention là hiểu được một nửa deep learning hiện đại.
- **Pre-train + fine-tune là mô hình làm việc chuẩn từ BERT/GPT trở đi:** học biểu diễn chung trên dữ liệu khổng lồ không nhãn (đoán từ bị che / đoán từ tiếp theo), rồi chuyên hoá bằng ít dữ liệu có nhãn. Đây cũng là lý do mô hình ngôn ngữ lớn có few-shot learning.
- **Chi phí O(N²) của attention là nút thắt thực tế:** chuỗi càng dài càng đắt — mọi kỹ thuật mở rộng (sparse attention, cửa sổ trượt, global token) đều là cách "cắt bớt" ma trận tương tác mà vẫn giữ được phụ thuộc xa.
- **Notebook thực hành:** 10.1 1D convolution · 10.2 Convolution for MNIST-1D · 10.3 2D convolution · 10.4 Downsampling & upsampling · 10.5 Convolution for MNIST · 11.1 Shattered gradients · 11.2 Residual networks · 11.3 BatchNorm · 12.1 Self-attention · 12.2 Multi-head self-attention · 12.3 Tokenization · 12.4 Decoding strategies · 13.1 Encoding graphs · 13.2 Graph classification · 13.3 Neighborhood sampling · 13.4 Graph attention — xem tại https://udlbook.github.io/udlbook/.

**Tiếp theo:** phần 4 (chương 14–18: học không giám sát / mô hình sinh — VAE, GAN, normalizing flows, diffusion, v.v.).
