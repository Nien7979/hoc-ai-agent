# S1 — Understanding Deep Learning (Simon J.D. Prince): Phần 4 — Chương 14–18

> Sách: **Understanding Deep Learning** — Simon J.D. Prince, MIT Press 2023, 544 trang, **PDF MIỄN PHÍ** (CC BY-NC-ND, open-access). Bản đang đọc: release v5.0.3 (08/02/2026), 541 trang, tải từ https://github.com/udlbook/udlbook/releases (file `UnderstandingDeepLearning_02_09_26_C.pdf`, 22 MB, lưu tại `~/workspace/knowledge/understanding-deep-learning/`).
> Sách có 21 chương + 68 notebook Python thực hành (https://udlbook.github.io/udlbook/). Mục tiêu của sách: hiểu **nguyên lý** đằng sau deep learning — không quá lý thuyết (không có chứng minh), cũng không quá thực hành (gần như không có code).
> Ghi chú này: phần 4 (chương 14–18, trang sách 269–367). Phần 1 (chương 1–4), phần 2 (chương 5–9) và phần 3 (chương 10–13) ở cùng thư mục `hoc-ai-agent/`. Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong tài liệu gốc.
> Chú thích: từ phần này, sách yêu cầu nền xác suất vững (phụ lục C của sách).

---

## Chương 14 — Unsupervised learning (Học không giám sát)

### 14.1 Phân loại các mô hình học không giám sát

- Đặc điểm chung: học từ tập dữ liệu quan sát được {xᵢ} **mà không có nhãn**. Mục đích đa dạng: sinh mẫu mới giống dữ liệu, biến đổi / khử nhiễu / nội suy / nén ví dụ, khám phá cấu trúc (phân cụm), hoặc phân biệt ví dụ mới có thuộc cùng tập dữ liệu hay là outlier.
- Chiến lược chung: định nghĩa ánh xạ giữa dữ liệu x và **biến ẩn (latent variable) z** — z thường thấp chiều hơn x, có thể coi là **bản nén** của x, giữ lại những phẩm chất cốt lõi. Ánh xạ có thể theo hai hướng:
  - **x → z:** vd k-means gán mỗi x vào một cụm z ∈ {1…K}.
  - **z → x:** định nghĩa phân phối Pr(z) trên biến ẩn, rồi sinh mẫu bằng cách (i) lấy mẫu từ Pr(z), (ii) ánh xạ sang không gian dữ liệu x. Đây là **mô hình sinh (generative model)**.
- Bốn mô hình sinh ở chương 15–18 đều dùng biến ẩn:
  - **GAN (chương 15):** sinh x* từ z bằng loss khuyến khích mẫu sinh không phân biệt được với mẫu thật. **Không xây phân phối xác suất** → không đánh giá được xác suất của một điểm dữ liệu mới.
  - **Normalizing flow, VAE, diffusion (chương 16–18):** mô hình sinh **xác suất** — vừa sinh mẫu, vừa gán xác suất Pr(x|ϕ) cho mỗi điểm dữ liệu. Train bằng cách **cực đại hoá xác suất của dữ liệu quan sát** (loss = tổng negative log-likelihood): L[ϕ] = −Σᵢ log Pr(xᵢ|ϕ). Vì phân phối phải có tổng bằng 1, việc tăng xác suất ở dữ liệu thật sẽ tự động giảm xác suất ở xa dữ liệu.
  - Ngoại lệ: transformer decoder (chương 12) cũng sinh được và gán được xác suất, nhưng dùng công thức **tự hồi quy**, không qua biến ẩn.

### 14.2 Mô hình sinh tốt cần những gì?

Sáu tính chất mong muốn (hình 14.3 — **không mô hình nào đạt cả sáu**):

| Mô hình | Lấy mẫu hiệu quả | Chất lượng mẫu | Độ phủ | Không gian ẩn "ngoan" | Không gian ẩn "gỡ rối" | Tính likelihood hiệu quả |
|---|---|---|---|---|---|---|
| GAN | ✓ | ✓ | ~ | ✓ | ? | không áp dụng |
| VAE | ✓ | ~ | ? | ✓ | ? | ~ |
| Flow | ✓ | ~ | ? | ✓ | ? | ✓ |
| Diffusion | ~ | ✓ | ? | ~ | ~ | ~ |

- **Efficient sampling:** sinh mẫu rẻ, tận dụng được song song phần cứng.
- **High-quality sampling:** mẫu không phân biệt được với dữ liệu thật.
- **Coverage (độ phủ):** mẫu phải bao phủ toàn bộ phân phối train — sinh ra toàn ảnh giống một tập con là không đủ.
- **Well-behaved latent space:** mọi z đều tương ứng với một x khả dĩ; thay đổi z mượt → x thay đổi mượt (nội suy được).
- **Disentangled latent space:** mỗi chiều của z điều khiển một thuộc tính có thể diễn giải được (vd chủ đề, thì, độ dài của câu).
- **Efficient likelihood computation:** tính được xác suất của ví dụ mới nhanh và chính xác.

### 14.3 Đo hiệu năng

- **Test likelihood:** đo likelihood trên tập test (không phải train — mô hình có thể gán xác suất cực cao đúng tại các điểm train và gần 0 ở giữa, likelihood train cao nhưng không sinh được gì mới). Test likelihood vừa đo khả năng tổng quát hoá, vừa đo độ phủ. Chỉ áp dụng cho mô hình xác suất; GAN không có; VAE và diffusion chỉ tính được **cận dưới** của log-likelihood; **flow là mô hình duy nhất tính được likelihood chính xác và hiệu quả**.
- **Inception score (IS):** dùng mô hình phân loại Inception (pre-train trên ImageNet) chấm ảnh sinh ra. Hai tiêu chí: (i) mỗi ảnh sinh phải trông giống **đúng một** trong 1000 lớp → Pr(y|x*ᵢ) phải **nhọn**; (ii) toàn bộ ảnh sinh phải phủ đều các lớp → Pr(y) trung bình phải **phẳng**. IS = exp của KL divergence kỳ vọng giữa hai phân phối này. Nhược điểm: chỉ hợp lý cho ImageNet, nhạy với mô hình phân loại dùng chấm, **không thưởng cho đa dạng trong cùng một lớp** (chỉ sinh một ví dụ đẹp cho mỗi lớp cũng được điểm cao).
- **Fréchet inception distance (FID):** xấp xỉ cả hai phân phối (thật và sinh) bằng **Gaussian đa biến trên activation của tầng sâu nhất của Inception** rồi đo khoảng cách Fréchet giữa chúng. Có tính đa dạng trong lớp, nhưng chỉ so sánh ở mức ngữ nghĩa (activation sâu nhất), bỏ qua chi tiết mịn; và thông tin nào mạng Inception đã vứt thì không được tính.
- **Manifold precision/recall:** tách riêng hai phẩm chất mà FID gộp lại. Ước lượng manifold bằng hợp các siêu cầu quanh mỗi ví dụ (bán kính = khoảng cách tới điểm láng giềng thứ k). **Precision** = tỉ lệ mẫu sinh nằm trong manifold dữ liệu thật (mẫu có thật không); **recall** = tỉ lệ dữ liệu thật nằm trong manifold mẫu sinh (mô hình phủ được bao nhiêu dữ liệu thật). Thường tính trong không gian đặc trưng của một classifier.

**Tóm tắt chương 14:** học không giám sát = học từ dữ liệu không nhãn. Mô hình sinh ánh xạ biến ẩn z → dữ liệu x. GAN sinh mẫu nhưng không gán xác suất; flow/VAE/diffusion là mô hình sinh xác suất, train bằng maximum likelihood. Không có mô hình nào toàn diện; đánh giá bằng test likelihood (chỉ cho mô hình xác suất), inception score, FID, và precision/recall trên manifold.

---

## Chương 15 — Generative adversarial networks (GAN)

### 15.1 Phân biệt làm tín hiệu huấn luyện

- **Cơ chế:** generator g[z, θ] ánh xạ nhiễu z (vd chuẩn) thành mẫu x* = g[z, θ]. **Discriminator** f[x, ϕ] là một mạng phân loại nhị phân: phân biệt mẫu thật và mẫu sinh. Nếu discriminator không phân biệt nổi → mẫu sinh đủ giống thật → thành công. Nếu phân biệt được → tín hiệu đó dùng để cải thiện generator.
- **Ví dụ đồ chơi 1D:** generator x* = z + θ (dịch mẫu theo trục x), θ khởi tạo = 3,0. Xen kẽ cập nhật discriminator rồi generator; mẫu sinh dần dịch về phía mẫu thật; khi không phân biệt được nữa thì discriminator về mức đoán mò (0,5), sigmoid phẳng dần, gradient thúc đẩy generator nhỏ dần. Discriminator sau đó bị **vứt bỏ**, chỉ giữ generator.
- **Loss GAN:** điều chỉnh binary cross-entropy (chương 5): mẫu thật nhãn y=1, mẫu sinh y=0:
  - Discriminator: L[ϕ] = −Σⱼ log(1 − sig[f[g[zⱼ,θ],ϕ]]) − Σᵢ log sig[f[xᵢ,ϕ]] (minimise).
  - Generator: L[θ] = Σⱼ log(1 − sig[f[g[zⱼ,θ],ϕ]]) (minimise — tức maximise xác suất mẫu sinh bị nhận nhầm là thật).
  - Đây là **trò chơi minimax** — nghiệm là **Nash equilibrium**: đồng thời là cực tiểu theo ϕ và cực đại theo θ. Lúc hội tụ, mẫu sinh cùng phân phối với dữ liệu thật, discriminator về 0,5.
- **DCGAN:** kiến trúc GAN tích chập kinh điển cho ảnh. Generator: z 100 chiều (uniform) → biến đổi tuyến tính thành 4×4×1024 kênh → 4 tầng tích chập **fractionally-strided** (stride 0,5 — gấp đôi độ phân giải mỗi tầng) → tanh → ảnh 64×64×3 trong [−1, 1]. Discriminator: CNN chuẩn, tầng cuối 1×1×1 → sigmoid. Sau train, vứt discriminator; lấy mẫu bằng cách rút z rồi chạy generator. Kết quả: mặt người, cảnh LSUN khá thật (hình 15.4).
- **Khó train:** DCGAN chỉ train ổn định khi (i) dùng strided conv để up/downsample; (ii) BatchNorm ở cả generator lẫn discriminator (trừ tầng cuối generator và tầng đầu discriminator); (iii) leaky ReLU ở discriminator; (iv) Adam với momentum thấp hơn bình thường. Hầu hết mô hình deep learning khác không nhạy cảm tới mức này.
- **Mode dropping / mode collapse:** lỗi thường gặp — generator sinh mẫu thật nhưng chỉ bao phủ **một tập con** của dữ liệu (vd mặt người mà không bao giờ có râu). Cực đoan: generator bỏ qua luôn z, **sụp đổ** mọi mẫu về một hoặc vài điểm giống nhau (hình 15.5).

### 15.2 Vì sao GAN khó train — phân tích loss

- **Loss GAN = Jensen–Shannon divergence:** với discriminator tối ưu, Pr(real|x̃) = Pr(x)/(Pr(x*) + Pr(x)). Thay vào loss ta được (bỏ hằng số) D_JS[Pr(x*) || Pr(x)] — gồm hai hạng KL: hạng thứ nhất phạt vùng có mẫu sinh mà không có dữ liệu thật (**chất lượng**); hạng thứ hai phạt vùng có dữ liệu thật mà không có mẫu sinh (**độ phủ**). Nhưng hạng thứ hai **không phụ thuộc vào generator** → generator không quan tâm độ phủ, hài lòng với việc sinh chính xác một tập con → đây được cho là nguyên nhân của **mode dropping**.
- **Vanishing gradients (gradient biến mất):** nếu hai phân phối hoàn toàn rời nhau thì JS divergence đạt cực đại và mọi thay đổi nhỏ ở generator đều không giảm được loss — discriminator phân biệt hoàn hảo thì điểm của nó ở vùng sigmoid bão hoà, độ dốc ~0 (hình 15.6). Mà thực tế hai phân phối **thường rời nhau**: mẫu sinh nằm trong không gian con có kích thước bằng z, dữ liệu thật cũng nằm trong không gian con thấp chiều (hình 1.9). Thực nghiệm (hình 15.7): đóng băng generator của DCGAN, train discriminator thêm → gradient của generator giảm nhanh (thang log). → Cân bằng mong manh: discriminator **quá giỏi thì generator hết gradient**.
- **Wasserstein distance (khoảng cách earth mover):** lượng "công" cần để vận chuyển khối lượng xác suất từ phân phối này sang phân phối kia (khối lượng × khoảng cách di chuyển). **Xác định tốt ngay cả khi hai phân phối rời nhau** và giảm mượt khi chúng xích lại gần nhau — đúng tính chất cần có. Dạng rời rạc: tối thiểu hoá Σᵢⱼ Pᵢⱼ·|i−j| trên **kế hoạch vận chuyển** P (ma trận, hàng tổng = Pr(x=i), cột tổng = q(x=j), Pᵢⱼ ≥ 0) — bài toán **quy hoạch tuyến tính** (primal). Dạng **dual** tương đương: max trên biến f với ràng buộc |fᵢ₊₁ − fᵢ| < 1. Dạng liên tục: ràng buộc hàm f có **hằng số Lipschitz < 1** (độ dốc tuyệt đối < 1).
- **WGAN loss:** tối ưu f[x, ϕ] (mạng discriminator, nay gọi là critic) bằng L[ϕ] = Σⱼ f[g[zⱼ,θ],ϕ] − Σᵢ f[xᵢ,ϕ]; ràng buộc Lipschitz bằng cách **cắt (clip) trọng số** discriminator về khoảng nhỏ (±0,01), hoặc tốt hơn: **WGAN-GP** — thêm hạng phạt khi độ lớn gradient lệch khỏi 1. WGAN train ổn định hơn hẳn.

### 15.3 Ba mẹo cải thiện chất lượng

- **Progressive growing (mọc dần):** train GAN sinh ảnh **4×4** trước; khi ổn thì thêm tầng để lên 8×8, rồi 16×16… Các tầng độ phân giải cao "fade in" dần (ban đầu ảnh độ phân giải cao là bản upsample của kết quả cũ qua residual connection, tầng mới dần tiếp quản). Nhờ đó train được ảnh độ phân giải cao rất thật (Karras et al. 2018).
- **Minibatch discrimination:** cho discriminator thấy **thống kê trên cả batch** (tóm tắt rồi thêm làm feature map gần cuối) → discriminator phát hiện được khi mẫu sinh thiếu đa dạng → gửi tín hiệu về generator, chống mode collapse.
- **Truncation (cắt ngắn):** lúc lấy mẫu chỉ chọn z gần trung bình (trong τ độ lệch chuẩn) → chất lượng trung bình tăng, đa dạng giảm; τ nhỏ quá thì mẫu gần như giống hệt nhau (hình 15.10). Chọn τ khéo thì tăng chất lượng rõ rệt.

### 15.4 Sinh có điều kiện (conditional generation)

- **Conditional GAN:** đưa thêm vector thuộc tính c vào **cả generator lẫn discriminator** (g[z,c,θ], f[x,c,ϕ]) → điều khiển được thuộc tính của mẫu sinh (màu tóc, tuổi…). Với ảnh, c được biến đổi tuyến tính thành 2D rồi nối làm kênh phụ.
- **ACGAN (auxiliary classifier GAN):** discriminator có C+1 output: 1 sigmoid (thật/sinh) + C softmax (dự đoán lớp). Đơn giản hơn conditional GAN vì không cần đưa c vào discriminator; sinh được nhiều lớp ImageNet (bướm, chim sẻ vàng, hoa cúc…).
- **InfoGAN:** tách biến ẩn thành nhiễu z + thuộc tính ngẫu nhiên c **không đặt trước**; discriminator vừa phân biệt thật/sinh vừa **khôi phục lại c**. Ý tưởng: thuộc tính có ý nghĩa thực tế sẽ dễ dự đoán nhất → c sẽ tự học được chúng → **không gian ẩn tự gỡ rối (disentangled)**. Trên MNIST, InfoGAN tự tìm ra: c₁ rời rạc 10 giá trị = 10 chữ số; c₂ liên tục = góc xoay; c₃ liên tục = độ dày nét (hình 15.15).

### 15.5 Dịch ảnh (image translation) — discriminator làm loss phụ

- **Pix2Pix:** U-Net g[c, θ] dịch ảnh c sang phong cách khác (vd xám → màu, bản đồ → ảnh vệ tinh, phác thảo → ảnh thật). Hai loss: **content loss** ℓ1 ||x − g[c,θ]||₁ (giữ cấu trúc tổng thể đúng) + **adversarial loss** — discriminator f[c, x, ϕ] phân biệt cặp trước/sau thật với cặp trước/sinh. Vì content loss đã lo cấu trúc lớn, discriminator chủ yếu lo **texture cục bộ** → dùng **PatchGAN**: classifier thuần tích chập, mỗi hidden unit ở tầng cuối phán xét vùng trong receptive field của nó là thật hay sinh, rồi trung bình. (Lưu ý: U-Net không nhận nhiễu z — thử thêm nhiễu thì mạng học cách bỏ qua nó.)
- **Adversarial loss (tổng quát):** thêm hạng phạt nếu discriminator phân biệt được output của một mạng có giám sát với ví dụ thật trong miền output — không cần thêm dữ liệu có nhãn. **SRGAN** (siêu phân giải ×4): mạng tích chập + residual, train bằng 3 loss — content (bình phương sai khác), **VGG loss / perceptual loss** (bình phương sai khác activation của mạng VGG giữa ảnh sinh và ảnh thật — ép giống về ngữ nghĩa), và adversarial loss. Kết quả sắc nét hơn hẳn nội suy bicubic.
- **CycleGAN:** xử lý trường hợp có hai tập dữ liệu hai phong cách **không có cặp tương ứng** (vd ảnh thật ↔ tranh Monet). Train hai mô hình cùng lúc (ảnh→Monet và Monet→ảnh) với 3 loss: content ℓ1, adversarial (mỗi chiều), và **cycle-consistency** — dịch đi rồi dịch lại phải về ảnh gốc. Nhờ đó dịch phong cách mà không cần cặp ảnh trước/sau.

### 15.6 StyleGAN

- Ý tưởng: **tách biến thiên** của tập dữ liệu thành các thành phần có ý nghĩa, mỗi thành phần do một tập con biến ẩn điều khiển; tách **style** (quan trọng với con người) khỏi **noise** (biến thiên không quan trọng: vị trí sợi tóc, râu, tàn nhang, lỗ chân lông).
- Kiến trúc: nhánh chính bắt đầu từ **hằng số học được 4×4×512 kênh**, qua các tầng tích chập upsample dần tới ảnh cuối. Tại mỗi tỉ lệ, đưa vào hai loại biến ẩn: (i) **noise** — vector Gaussian độc lập, cộng thêm sau mỗi phép tích chập với hệ số co giãn theo kênh ψ học được (càng gần output thì chi tiết càng mịn); (ii) **style** — z 512 chiều qua mạng fully connected 7 tầng thành biến trung gian **w** (giúp gỡ tương quan để mỗi chiều của w ứng với một yếu tố độc lập như tư thế đầu, màu tóc), rồi biến đổi tuyến tính thành y dùng để đặt **trung bình và phương sai theo kênh** của biểu diễn — gọi là **adaptive instance normalization (AdaIN)**. Cùng một w được đưa vào ở nhiều tỉ lệ khác nhau. Kết quả: thay đổi style ở tỉ lệ thô đổi hình dạng/tư thế đầu; ở tỉ lệ vừa đổi nét mặt; ở tỉ lệ mịn đổi màu tóc/da; thay đổi noise đổi chi tiết nhỏ (hình 15.20).

**Tóm tắt chương 15:** GAN = generator sinh mẫu + discriminator phân biệt thật/sinh, train xen kẽ theo trò chơi minimax; loss tối ưu tương đương Jensen–Shannon divergence. Điểm yếu gốc: gradient biến mất khi hai phân phối rời nhau và generator không bị phạt vì thiếu độ phủ → mode dropping/collapse; WGAN (khoảng cách earth mover + ràng buộc Lipschitz) khắc phục phần lớn. Progressive growing, minibatch discrimination, truncation nâng chất lượng; conditional/ACGAN/InfoGAN cho điều khiển và gỡ rối thuộc tính; discriminator còn dùng làm loss phụ cho dịch ảnh (Pix2Pix, SRGAN, CycleGAN); StyleGAN tách style theo tỉ lệ và noise theo chi tiết.

---

## Chương 16 — Normalizing flows (Dòng chuẩn hoá)

### 16.1 Ví dụ 1D — biến đổi phân phối

- Lấy biến ẩn z có mật độ gốc đã biết Pr(z) (thường chuẩn), biến đổi qua hàm x = f[z, ϕ] → x có phân phối mới. Vùng nào hàm **kéo giãn** (đạo hàm > 1) thì mật độ giảm; vùng nào **nén** (đạo hàm < 1) thì mật độ tăng — tổng diện tích dưới mật độ vẫn bằng 1 (hình 16.2).
- Công thức likelihood: Pr(x|ϕ) = |∂f[z,ϕ]/∂z|⁻¹ · Pr(z), với z = f⁻¹[x, ϕ] là biến ẩn đã tạo ra x. Muốn tính likelihood phải **đảo được hàm** f.
- **Forward (hướng sinh):** z → x để lấy mẫu. **Inverse / normalizing (hướng chuẩn hoá):** x → z để đo likelihood — biến phân phối phức tạp về lại phân phối chuẩn (hình 16.3).
- **Train:** cực đại hoá likelihood dữ liệu train (tối thiểu hoá negative log-likelihood): L = Σᵢ [log|∂f[zᵢ,ϕ]/∂zᵢ| − log Pr(zᵢ)].

### 16.2 Trường hợp tổng quát

- Với x, z ∈ R^D và f là mạng sâu: Pr(x|ϕ) = |det(∂f[z,ϕ]/∂z)|⁻¹ · Pr(z). Định thức Jacobian đo **độ thay đổi thể tích** tại một điểm (tương tự đạo hàm tuyệt đối ở 1D). Mạng gồm K tầng f₁…f_K: forward là hợp thành theo thứ tự, inverse là hợp thành các tầng đảo theo **thứ tự ngược**; định thức Jacobian toàn cục = **tích** các định thức từng tầng → tính được hiệu quả nếu từng tầng có định thức dễ tính. Tên "normalizing flow": mỗi tầng đảo dần "chảy" mật độ dữ liệu về phân phối chuẩn.
- **Bốn yêu cầu đối với tầng mạng:** (1) đủ biểu cảm để ánh xạ chuẩn đa biến thành mật độ tuỳ ý; (2) **khả đảo** (song ánh — một-một, nếu nhiều input về cùng một output thì inverse mơ hồ); (3) tính được inverse hiệu quả (dùng lặp lại trong train); (4) tính được định thức Jacobian hiệu quả.

### 16.3 Các tầng khả đảo (flows)

- **Linear flow:** f[h] = β + Ωh. Đảo và định thức tốn O[D³] với Ω tổng quát; ma trận đặc biệt rẻ hơn: đường chéo O[D] (nhưng các chiều không tương tác), trực giao (định thức cố định, không co giãn từng chiều), tam giác (đảo bằng back-substitution O[D²], định thức = tích đường chéo). Cách hay: tham số hoá trực tiếp theo **phân rã LU**: Ω = PL(U + D) — đảo O[D²], log-định thức = tổng log |đường chéo|. Nhược: hàm tuyến tính biến chuẩn thành chuẩn → **không đủ biểu cảm một mình**.
- **Elementwise flow:** áp cùng một hàm phi tuyến khả đảo cho **từng phần tử** (vd leaky ReLU, hoặc hàm tuyến tính từng đoạn với K đoạn, độ dốc dương tổng = 1, hình 16.5; hoặc spline đơn điệu). Jacobian là ma trận đường chéo → định thức = tích các đạo hàm. Nhược: không trộn các chiều → phải xen kẽ với linear flow.
- **Coupling flow:** chia input h = [h₁, h₂]; giữ nguyên h₁, biến đổi h₂ bằng một flow khả đảo g mà tham số **phụ thuộc vào h₁** (tính bằng một mạng bất kỳ — không cần khả đảo): h'₁ = h₁; h'₂ = g[h₂, ϕ[h₁]]. Inverse dễ: h₁ = h'₁ rồi tính ϕ[h₁] và đảo g. Jacobian tam giác dưới → định thức = tích các đạo hàm của g. Chỉ biến đổi nửa sau theo nửa đầu → **hoán vị (permutation) các phần tử giữa các tầng** để mọi biến cuối cùng đều được biến đổi bởi mọi biến khác (với ảnh: chia kênh làm đôi, trộn kênh bằng conv 1×1).
- **Autoregressive flow:** tổng quát hoá coupling: mỗi chiều output h'_d là hàm khả đảo của h_d với tham số phụ thuộc **mọi chiều trước đó** h₁…h_{d−1}. Với mask phù hợp (như masked self-attention), **forward tính song song** được; nhưng **inverse phải tính tuần tự** (muốn h₂ phải biết h₁ trước…) → chậm khi input lớn. Nếu đủ linh hoạt thì autoregressive flow là **universal approximator** — biểu diễn được mọi phân phối. **Inverse autoregressive flow (IAF):** định nghĩa theo hướng ngược lại — lấy mẫu nhanh, tính likelihood chậm. Mẹo: train một masked autoregressive flow (teacher, likelihood nhanh) rồi **chưng cất** sang IAF (student, lấy mẫu nhanh) — xem §16.5.3.
- **Residual flow (iRevNet):** chia h = [h₁, h₂], hai tầng dư: h'₁ = h₁ + f₁[h₂], h'₂ = h₂ + f₂[h'₁]; đảo bằng cách tính ngược và đổi cộng thành trừ. f₁, f₂ không cần khả đảo. Nhược: **không tính được định thức Jacobian hiệu quả** — nhưng vẫn hữu ích để tiết kiệm bộ nhớ khi train ResNet (không cần lưu activation, tính lại được khi backward).
- **iResNet (contraction mapping):** dùng **định lý điểm bất động Banach**: nếu f là ánh xạ co (khoảng cách sau khi qua f nhỏ hơn β < 1 lần trước đó) thì lặp f sẽ hội tụ về điểm bất động f[z] = z. Để đảo y = z + f[z] (tầng dư) với f là ánh xạ co: lặp z_{k+1} = y* − f[z_k]. Điều kiện co ⇔ hằng số Lipschitz < 1 ⇔ giá trị kỳ dị lớn nhất của mỗi ma trận trọng số < 1 (đạt bằng clipping). Log-định thức không tính trực tiếp được mà xấp xỉ bằng **khai triển chuỗi Taylor của log(I + ∂f/∂h)** + **ước lượng trace Hutchinson** (trace[A] ≈ trung bình ϵᵀAϵ với ϵ ~ chuẩn).

### 16.4 Multi-scale flows

- Không gian ẩn z phải cùng kích thước với dữ liệu x, nhưng dữ liệu tự nhiên thường mô tả được bằng ít biến hơn. **Multi-scale flow:** chia z = [z₁…z_N]; z₁ xử lý qua các tầng cùng chiều, rồi **nối thêm** z₂, cứ thế cho tới khi bằng kích thước x. Hướng chuẩn hoá thì ngược lại: tới điểm z_n được thêm vào thì đánh giá nó theo phân phối gốc luôn, không xử lý tiếp. Tiết kiệm tính toán cả hai hướng (hình 16.10).

### 16.5 Ứng dụng

- **Ước lượng mật độ:** flow là mô hình duy nhất trong bốn mô hình tính được **log-likelihood chính xác** (GAN không có; VAE và diffusion chỉ có cận dưới). Ứng dụng: **phát hiện bất thường** — mô hình hoá phân phối dữ liệu sạch, ví dụ mới có xác suất thấp thì gắn cờ outlier (cẩn thận: có outlier xác suất cao nhưng nằm ngoài typical set — xem hình 8.13).
- **Sinh ảnh — GLOW:** coupling layers (chia kênh làm đôi, nửa sau chịu biến đổi affine theo từng vị trí, tham số tính bằng CNN 2D trên nửa còn lại) xen kẽ **conv 1×1** (tham số hoá LU) để trộn kênh; định kỳ giảm độ phân giải một nửa bằng cách gộp patch 2×2 thành 1 vị trí ×4 kênh; multi-scale (một số kênh tách ra thành z). Ảnh là rời rạc (RGB lượng tử hoá) → thêm nhiễu vào input để likelihood không tăng vô hạn (**dequantization**). Lấy mẫu "thật hơn" bằng cách lấy từ phân phối gốc **luỹ thừa** (chọn gần tâm hơn là đuôi — tương tự truncation của GAN). Chất lượng mẫu kém GAN/diffusion (chưa rõ do giới hạn bản chất của tầng khả đảo hay do ít nghiên cứu hơn). **Nội suy** trong không gian ẩn cho ảnh chuyển đổi mượt giữa hai người thật (hình 16.13).
- **Xấp xỉ một mật độ khác (student–teacher):** flow (student) học sinh mẫu xấp xỉ một mật độ đích q(x) mà ta **tính được nhưng khó lấy mẫu**. Vì mẫu do chính student sinh ra nên biết z tương ứng → tính được likelihood mà không cần đảo → dùng được cả flow mà đảo chậm (masked autoregressive). Loss = **reverse KL divergence** giữa mẫu student và mật độ teacher. Trái với cách dùng thường (maximum likelihood = forward KL từ dữ liệu train tới flow). Ứng dụng: chưng cất WaveNet (lấy mẫu chậm) thành Parallel WaveNet (lấy mẫu nhanh); mô hình hoá posterior trong VAE.

**Tóm tắt chương 16:** normalizing flow = biến đổi khả đảo từ phân phối gốc đơn giản (chuẩn) thành phân phối dữ liệu; likelihood tính qua định thức Jacobian. Ưu điểm duy nhất trong bốn mô hình: **likelihood chính xác + sinh mẫu**. Giá phải trả: mọi tầng phải khả đảo và có Jacobian tính được hiệu quả — linear (LU), elementwise (spline), coupling (chia đôi + hoán vị), autoregressive (mask, universal nhưng đảo chậm), residual (iRevNet/iResNet, Jacobian xấp xỉ). GLOW là ví dụ tổng hợp cho ảnh.

---

## Chương 17 — Variational autoencoders (VAE)

### 17.1 Mô hình biến ẩn (latent variable model)

- Cách gián tiếp để mô tả phân phối Pr(x): mô hình hoá phân phối **hợp** Pr(x, z) của dữ liệu x và biến ẩn z, rồi lấy **biên (marginalize)** theo z: Pr(x) = ∫ Pr(x, z) dz = ∫ Pr(x|z)·Pr(z) dz. Biểu thức đơn giản cho Pr(x|z) và Pr(z) có thể tạo ra Pr(x) phức tạp.
- **Ví dụ — hỗn hợp Gaussian (MoG):** z rời rạc (chọn thành phần n với xác suất λ_n), Pr(x|z=n) = Norm[μ_n, σ_n²]. Pr(x) = Σ_n λ_n·Norm[μ_n, σ_n²] — tổng có trọng số của các Gaussian → phân phối đa đỉnh phức tạp (hình 17.1).

### 17.2 Mô hình biến ẩn phi tuyến

- Prior: Pr(z) = Norm[0, I] (chuẩn đa biến). Likelihood: Pr(x|z, ϕ) = Norm[f[z, ϕ], σ²I] — trung bình là **mạng sâu** f[z, ϕ], hiệp phương sai cầu. z thấp chiều hơn x; f mô tả khía cạnh quan trọng của dữ liệu, phần còn lại quy cho nhiễu σ²I.
- Pr(x|ϕ) = ∫ Norm[f[z,ϕ], σ²I]·Norm[0,I] dz — có thể xem là **tổng vô hạn** các Gaussian cầu với trung bình khác nhau, trọng số Pr(z) (hình 17.2).
- **Sinh mẫu (ancestral sampling):** rút z* ~ prior → tính trung bình f[z*, ϕ] → rút x* ~ Norm[f[z*,ϕ], σ²I].

### 17.3 Train — ELBO (evidence lower bound)

- **Vấn đề:** log-likelihood log Pr(x|ϕ) chứa tích phân không có dạng đóng → không tối ưu trực tiếp được.
- **Công cụ — bất đẳng thức Jensen:** với hàm lõm g (vd log): g[E[y]] ≥ E[g[y]], tức log[E[y]] ≥ E[log[y]] — log của kỳ vọng ≥ kỳ vọng của log (hình 17.4–17.5: đường thẳng nối hai điểm trên đồ thị log luôn nằm **dưới** đường cong).
- **Đạo ra ELBO:** nhân/chia log-likelihood với một phân phối tuỳ ý q(z) rồi áp Jensen:
  log Pr(x|ϕ) = log ∫ q(z)·[Pr(x,z|ϕ)/q(z)] dz ≥ ∫ q(z)·log[Pr(x,z|ϕ)/q(z)] dz =: **ELBO[θ, ϕ]**.
  ELBO luôn nằm **dưới** log-likelihood; tối đa hoá ELBO sẽ đẩy log-likelihood lên theo (hình 17.6).
- **ELBO chặt khi nào?** Viết lại: ELBO = log Pr(x|ϕ) − D_KL[q(z|θ) || Pr(z|x, ϕ)]. KL divergence ≥ 0 → ELBO là cận dưới; **chặt (bằng nhau) khi q(z|θ) = posterior Pr(z|x, ϕ)** — phân phối trên các giá trị z có thể đã tạo ra dữ liệu x (tính bằng quy tắc Bayes, hình 17.7).
- **Dạng thứ ba (dạng dùng trong VAE):** ELBO = E_q[log Pr(x|z, ϕ)] − D_KL[q(z|θ) || Pr(z)] — **hạng tái tạo** (reconstruction: z lấy từ q có giải thích được x không) trừ **khoảng cách KL tới prior** (q có gần prior chuẩn không).

### 17.5–17.6 Xấp xỉ biến phân (variational approximation) và VAE

- Posterior Pr(z|x, ϕ) không tính được (mẫu số Pr(x|ϕ) chính là cái không tính được). **Xấp xỉ biến phân:** chọn một họ phân phối đơn giản cho q — ở đây là **Gaussian đa biến với trung bình μ và hiệp phương sai chéo Σ** — rồi tìm thành viên gần posterior nhất (tối thiểu hoá KL trong công thức trên). Vì posterior phụ thuộc x nên q cũng vậy: q(z|x, θ) = Norm[g_μ[x,θ], g_Σ[x,θ]], với g[x, θ] là một **mạng encoder** dự đoán μ, Σ.
- **VAE** = mạng tính ELBO: ELBO[θ, ϕ] ≈ log Pr(x|z*, ϕ) − D_KL[q(z|x,θ) || Pr(z)], trong đó z* là **một mẫu duy nhất** rút từ q(z|x, θ) (ước lượng Monte Carlo của kỳ vọng — 1 mẫu đã đủ dùng trong thực tế). Hạng KL giữa hai Gaussian có **dạng đóng**: ½(Tr[Σ] + μᵀμ − D_z − log det[Σ]).
- **Thuật toán:** với mỗi x: encoder g[x, θ] → μ, Σ → rút z* → decoder f[z*, ϕ] → dự đoán x → tính ELBO → cập nhật cả θ lẫn ϕ bằng SGD/Adam. Vì vậy gọi là **variational autoencoder**: "variational" vì xấp xỉ posterior bằng Gaussian; "autoencoder" vì x → z (bottleneck thấp chiều) → tái tạo x. Encoder = x→z, decoder = z→x (hình 17.9).
- **Reparameterization trick (mẹo tái tham số hoá):** mạng có bước **lấy mẫu ngẫu nhiên** → không vi phân qua được để cập nhật θ đứng trước nó. Mẹo: rút ϵ* ~ Norm[0, I] ở một nhánh riêng rồi tính z* = μ + Σ^{½}·ϵ* — phần ngẫu nhiên nằm ngoài đường chính, backprop đi qua μ, Σ bình thường (hình 17.11).

### 17.8 Ứng dụng

- **Xấp xỉ xác suất mẫu:** Pr(x) = E_z[Pr(x|z)] — lấy mẫu trực tiếp từ prior thì gần như mọi z đều cho Pr(x|z) rất thấp (curse of dimensionality). Dùng **importance sampling**: lấy mẫu z từ q(z) (nên chọn **posterior biến phân q(z|x)** của encoder — vì Pr(x|z)Pr(z) ∝ posterior) rồi hiệu chỉnh bằng tỉ lệ Pr(z)/q(z). Đủ mẫu thì ước lượng tốt hơn cả cận dưới → dùng để đánh giá mô hình trên test hoặc phát hiện outlier.
- **Sinh mẫu:** rút z ~ prior → decoder → thêm nhiễu. **VAE thường sinh ảnh chất lượng thấp** (hình 17.12a–c): ảnh quá mượt trước khi thêm nhiễu, quá nhiễu sau khi thêm — một phần do nhiễu Gaussian cầu, một phần do prior/posterior Gaussian. Mẹo: lấy mẫu từ **aggregated posterior** q(z) = trung bình posterior trên toàn bộ dữ liệu (hỗn hợp Gaussian, đại diện hơn prior). VAE hiện đại sinh ảnh tốt nhưng cần **prior phân cấp** + kiến trúc/chính quy hoá chuyên biệt. **Diffusion (chương 18) có thể xem là VAE với prior phân cấp.**
- **Resynthesis (tái tổng hợp):** chiếu dữ liệu thật vào không gian ẩn (lấy trung bình của encoder, hoặc tối ưu z để cực đại hoá posterior ∝ Pr(x|z)Pr(z)), rồi **cộng/trừ vector** trong không gian ẩn để đổi thuộc tính (vd vector "cười − trung tính", "mở miệng − ngậm miệng") rồi decode lại; nội suy dùng **slerp** (nội suy trên mặt cầu) thay vì nội suy tuyến tính. GAN và flow cũng làm được resynthesis, nhưng GAN không có encoder nên phải tìm z bằng tối ưu riêng.
- **Disentanglement (gỡ rối):** thêm hạng chính quy hoá vào loss để mỗi chiều của z ứng với một yếu tố thực tế độc lập. **β-VAE:** nhân hạng KL với β > 1 — vì prior là Gaussian cầu (các chiều độc lập) nên ép posterior ít tương quan hơn. **Total correlation VAE:** thêm hạng giảm tương quan toàn phần giữa các biến ẩn. Trên ảnh ghế: một số chiều tự học được ý nghĩa rõ — góc xoay, kích thước tổng thể, loại chân ghế (hình 17.14).
- **Các vấn đề đã biết:** **posterior collapse** — encoder luôn dự đoán prior (khắc phục bằng cách tăng dần hạng KL trong train); **tái tạo mờ** (do nhiễu Gaussian + posterior xấp xỉ chưa tối ưu); **"lỗ" trong không gian ẩn** — vùng z không tương ứng mẫu thật nào (constrained posterior VAE); **information preference** — likelihood phức tạp (vd PixelCNN) khiến output không còn phụ thuộc z.
- **Liên hệ với EM:** ELBO chặt khi q = posterior — đó là cơ sở của **thuật toán expectation–maximization (EM)**: xen kẽ (i) E-step: đặt q = posterior, (ii) M-step: tối đa hoá cận dưới theo ϕ. EM dùng được cho MoG (posterior tính được dạng đóng) nhưng không dùng được cho mô hình biến ẩn phi tuyến.

**Tóm tắt chương 17:** VAE học mô hình biến ẩn phi tuyến Pr(x) = ∫ Pr(x|z)Pr(z)dz. Likelihood không tính được → tối đa hoá **ELBO** = tái tạo − KL tới prior, với posterior được xấp xỉ bằng Gaussian do encoder dự đoán; lấy mẫu bằng **reparameterization trick** để vi phân được. VAE sinh mẫu kém sắc nét hơn GAN/diffusion nhưng có encoder (resynthesis, nội suy), và là nền tảng khái niệm cho diffusion.

---

## Chương 18 — Diffusion models (Mô hình khuếch tán)

### 18.1 Tổng quan

- Diffusion gồm **encoder + decoder**, cả hai đều **ngẫu nhiên** (không xác định như VAE thường).
- **Encoder (forward / quá trình khuếch tán):** lấy mẫu x, đi qua chuỗi biến ẩn z₁…z_T, **trộn dần dữ liệu với nhiễu trắng** theo lịch trình định sẵn. Đủ bước thì q(z_T|x) và q(z_T) đều thành **phân phối chuẩn**. Encoder **không có tham số học được** — mọi tham số đều ở decoder.
- **Decoder (reverse):** học một chuỗi mạng để đi ngược từng cặp z_t → z_{t−1}, khử nhiễu dần cho tới khi còn lại một ví dụ dữ liệu thật. Sinh mẫu mới: rút z_T ~ chuẩn rồi chạy decoder ngược.
- Đây là mô hình sinh xác suất **chất lượng cao nhất** trong bốn mô hình ở thời điểm sách viết (vượt GAN về FID — Dhariwal & Nichol 2021).

### 18.2 Encoder (forward process)

- Công thức từng bước (chuỗi Markov — z_t chỉ phụ thuộc z_{t−1}): z_t = √(1−β_t)·z_{t−1} + √β_t·ϵ_t, với ϵ_t ~ chuẩn, β_t ∈ [0,1] là **noise schedule** (tốc độ trộn nhiễu). Hạng đầu **suy giảm** dữ liệu + nhiễu đã thêm; hạng sau **thêm nhiễu mới**. Dạng xác suất: q(z_t|z_{t−1}) = Norm[√(1−β_t)·z_{t−1}, β_t·I].
- **Diffusion kernel q(z_t|x) — dạng đóng:** không cần tính tuần tự z₁…z_{t−1}; thay trực tiếp: z_t = √ᾱ_t·x + √(1−ᾱ_t)·ϵ, với ᾱ_t = Π_{s=1..t}(1−β_s). Tức q(z_t|x) = Norm[√ᾱ_t·x, (1−ᾱ_t)·I] — trung bình tiến về 0, phương sai tăng dần theo t; t đủ lớn → chuẩn (hình 18.3). Đây là chìa khoá để train hiệu quả.
- **Marginal q(z_t):** lấy phân phối dữ liệu Pr(x), "bôi mờ" bằng diffusion kernel cho mỗi x rồi cộng lại (hình 18.4). Không viết được dạng đóng vì không biết Pr(x).
- **q(z_{t−1}|z_t)** (đảo một bước khi không biết x): không tính được vì cần marginal q(z_{t−1}); dạng tổng quát phức tạp, đa đỉnh, nhưng **thường xấp xỉ tốt bằng Gaussian** (hình 18.5) — đặc biệt khi β_t nhỏ.
- **q(z_{t−1}|z_t, x)** (đảo một bước khi **biết** x): tính được **dạng đóng** bằng quy tắc Bayes (kết hợp q(z_t|z_{t−1}) với diffusion kernel q(z_{t−1}|x) — tích hai Gaussian vẫn là Gaussian): trung bình = [(1−ᾱ_{t−1})/(1−ᾱ_t)]·√(1−β_t)·z_t + [√ᾱ_{t−1}·β_t/(1−ᾱ_t)]·x, phương sai = β_t(1−ᾱ_{t−1})/(1−ᾱ_t)·I (hình 18.6). Phân phối này dùng để **train decoder**.

### 18.3 Decoder (reverse process)

- Xấp xỉ mỗi bước đảo bằng Gaussian: Pr(z_T) = Norm[0, I]; Pr(z_{t−1}|z_t, ϕ_t) = Norm[f_t[z_t, ϕ_t], σ_t²I]; Pr(x|z₁, ϕ₁) = Norm[f₁[z₁, ϕ₁], σ₁²I]. f_t là mạng tính trung bình; σ_t² định trước. Xấp xỉ Gaussian hợp lý khi β_t gần 0 (bước nhỏ) và T lớn.
- Sinh mẫu bằng **ancestral sampling**: rút z_T ~ chuẩn → rút z_{T−1} ~ Pr(z_{T−1}|z_T) → … → rút x ~ Pr(x|z₁).

### 18.4 Train — ELBO

- Likelihood Pr(x|ϕ) = ∫ Pr(x, z₁…T|ϕ) dz₁…T không tính được → dùng **ELBO** như VAE (nhân/chia với q(z₁…T|x), áp Jensen). Vì encoder không có tham số, decoder phải làm **cả hai việc**: vừa làm cho encoder tĩnh xấp xỉ posterior, vừa tối đa hoá cận dưới.
- Rút gọn ELBO: thay q(z_t|z_{t−1}) bằng biểu thức Bayes qua q(z_{t−1}|z_t, x); hầu hết các hạng q(z_{t−1}|x)/q(z_t|x) **triệt tiêu lẫn nhau**, còn lại: ELBO ≈ E[log Pr(x|z₁, ϕ₁)] − Σ_{t=2..T} E[D_KL(q(z_{t−1}|z_t, x) || Pr(z_{t−1}|z_t, ϕ_t))].
  - Hạng đầu = **tái tạo** (như VAE).
  - Mỗi hạng KL là khoảng cách giữa hai Gaussian → có **dạng đóng**; phần lớn hạng không phụ thuộc ϕ nên rút gọn thành **bình phương sai khác giữa hai trung bình** + hằng số.
- **Loss diffusion:** L = −log Norm[xᵢ | f₁[zᵢ₁,ϕ₁], σ₁²I] + Σ_t (1/2σ_t²)·||mục tiêu − f_t[zᵢt, ϕ_t]||², trong đó "mục tiêu" là trung bình của q(z_{t−1}|z_t, xᵢ) — tức train mỗi mạng f_t dự đoán **giá trị khả dĩ nhất của biến ẩn ở bước trước**, cho trước dữ liệu thật đã khử nhiễu xᵢ.

### 18.5 Tái tham số hoá — dự đoán nhiễu thay vì dự đoán z_{t−1}

- Từ diffusion kernel: x = (z_t − √(1−ᾱ_t)·ϵ)/√ᾱ_t. Thay x vào "mục tiêu" ở trên, rút gọn (dùng ᾱ_t/ᾱ_{t−1} = 1−β_t) được: mục tiêu = [1/√(1−β_t)]·z_t − [β_t/(√(1−ᾱ_t)·√(1−β_t))]·ϵ.
- Đổi mô hình: thay vì f_t dự đoán z_{t−1}, dùng **g_t dự đoán chính nhiễu ϵ** đã trộn vào x để tạo z_t: f_t[z_t, ϕ_t] = [1/√(1−β_t)]·z_t − [β_t/(√(1−ᾱ_t)·√(1−β_t))]·g_t[z_t, ϕ_t]. Thay vào loss, các hệ số tỉ lệ **bỏ qua trong thực hành** (cho hiệu quả tốt hơn) → loss cuối cùng cực kỳ đơn giản:
  **L = Σᵢ Σ_t || g_t[√ᾱ_t·xᵢ + √(1−ᾱ_t)·ϵᵢt, ϕ_t] − ϵᵢt ||²**
  — tức **least squares giữa nhiễu dự đoán và nhiễu thật**. Đây là dạng loss dùng trong thực tế (Ho et al. 2020).
- **Thuật toán train (18.1):** lặp: lấy batch x; với mỗi ví dụ: rút **t ngẫu nhiên đều** trong [1..T], rút nhiễu ϵ ~ chuẩn, tính z_t bằng diffusion kernel, loss = ||g_t[z_t, ϕ_t] − ϵ||²; cộng loss batch rồi cập nhật gradient. Ưu điểm: đơn giản; **tự tăng cường dữ liệu** — mỗi x tái dùng nhiều lần ở nhiều t với nhiễu khác nhau.
- **Thuật toán lấy mẫu (18.2):** rút z_T ~ chuẩn; với t = T…2: dự đoán ẑ_{t−1} từ z_t và nhiễu dự đoán g_t, rồi **cộng thêm nhiễu** σ_t·ϵ (trừ bước cuối); cuối cùng sinh x từ z₁ **không thêm nhiễu**. Nhược: phải chạy mạng **tuần tự T lần** → sinh chậm.

### 18.6 Triển khai

- **Kiến trúc cho ảnh:** U-Net (chương 11) là lựa chọn hiển nhiên cho ánh xạ ảnh-nhiễu → nhiễu. Không train T U-Net riêng (quá tốn) mà train **một U-Net duy nhất nhận thêm vector đại diện cho bước thời gian t** (sinusoidal time embedding → mạng nông → cộng vào các kênh ở mọi giai đoạn của U-Net; hình 18.9). U-Net có residual block + **global self-attention** định kỳ (mọi vị trí không gian tương tác với nhau).
- **Tăng tốc sinh mẫu:** loss trên chỉ yêu cầu diffusion kernel có dạng q(z_t|x) = Norm[√ᾱ_t·x, (1−ᾱ_t)I] — có cả một **họ** quá trình forward tương thích với cùng một loss nhưng có quy tắc đảo khác nhau. **DDIM (denoising diffusion implicit model):** quá trình forward không còn ngẫu nhiên sau bước đầu → reverse **xác định** (không cộng nhiễu mỗi bước) → lấy mẫu tốt chỉ với **~50 bước** thay vì 1000. **Accelerated sampling:** forward chỉ định nghĩa trên dãy con các bước → reverse bỏ qua bước. Vẫn chậm hơn GAN/VAE/flow nhưng đã thực dụng.
- **Sinh có điều kiện:**
  - **Classifier guidance:** thêm hạng σ_t²·∇log Pr(c|z_t) vào bước cập nhật z_t → z_{t−1} — dùng gradient của một classifier (học riêng, trên đặc trưng nửa downsampling của U-Net, chia sẻ qua mọi t) để đẩy mẫu về phía lớp c.
  - **Classifier-free guidance:** không cần classifier riêng; đưa thông tin c vào thẳng mô hình g_t[z_t, ϕ_t, c] (thêm embedding của c vào U-Net như time embedding). Train chung cả mục tiêu có điều kiện và không điều kiện bằng cách **rớt ngẫu nhiên** thông tin c trong train → lúc test có thể sinh không điều kiện, có điều kiện, hoặc **tổ hợp có trọng số**. **Tăng trọng số điều kiện quá mức** cho mẫu chất lượng rất cao nhưng hơi khuôn mẫu — tương tự truncation của GAN.
- **Nâng chất lượng:** ước lượng cả **phương sai σ_t²** của quá trình đảo (đặc biệt giúp khi lấy mẫu ít bước); chỉnh **noise schedule** β_t theo từng bước; **cascade** — chuỗi diffusion: mô hình đầu sinh ảnh 64×64 (có thể theo text), mô hình sau sinh 256×256 rồi 1024×1024, mỗi mô hình sau điều kiện trên ảnh độ phân giải thấp hơn (resize rồi nối vào U-Net) + text encoding (hình 18.11).
- **Text-to-image:** điều kiện trên **sentence embedding** tính bằng mô hình ngôn ngữ (kiểu BERT) đưa vào mô hình như time embedding → sinh ảnh rất thật, khớp caption; vì diffusion ngẫu nhiên nên cùng một caption sinh được nhiều ảnh khác nhau (hình 18.13).

**Tóm tắt chương 18:** diffusion = encoder định sẵn trộn dần dữ liệu với nhiễu (chuỗi Markov, diffusion kernel dạng đóng cho phép lấy mẫu z_t trực tiếp từ x) + decoder học cách khử nhiễu từng bước. Train bằng ELBO → rút gọn thành loss least squares **dự đoán nhiễu** — đơn giản mà hiệu quả. Một U-Net duy nhất (nhận thêm time embedding) phục vụ mọi bước t. Sinh chậm (T bước tuần tự) nhưng DDIM và accelerated sampling giảm xuống ~50 bước. Classifier guidance / classifier-free guidance + cascade cho text-to-image chất lượng cao — đây là công nghệ đứng sau các mô hình sinh ảnh hiện đại (Stable Diffusion, DALL·E 2, Imagen…).

---

## Điểm rút ra thực hành (từ phần 4)

- **Bốn mô hình sinh = bốn cách trả lời cùng một câu hỏi "làm sao sinh dữ liệu giống thật":** GAN — "đánh lừa được người phân biệt là đủ" (không xác suất, mẫu đẹp nhưng hay thiếu độ phủ, train khó); Flow — "biến đổi khả đảo từ phân phối đơn giản" (likelihood chính xác duy nhất, nhưng kiến trúc bị gò bó); VAE — "nén rồi tái tạo, tối đa hoá cận dưới likelihood" (có encoder, không gian ẩn đẹp, nhưng mẫu mờ); Diffusion — "học khử nhiễu từng bước nhỏ" (mẫu đẹp nhất, train ổn định nhất, nhưng sinh chậm nhất). Không có mô hình toàn thắng — hình 14.3 là bảng tra cứu nhanh.
- **ELBO là cầu nối của cả VAE lẫn diffusion:** cùng một thủ thuật (Jensen + xấp xỉ biến phân) giải quyết cùng một vấn đề (likelihood chứa tích phân không tính được). Hiểu ELBO = hiểu được một nửa học sinh xác suất hiện đại. Điểm khác biệt duy nhất: VAE học cả encoder lẫn decoder; diffusion giữ encoder cố định và bắt decoder làm mọi việc — và chính sự "cố định" đó lại khiến bài toán train trở nên dễ hơn (mục tiêu khử nhiễu rõ ràng).
- **Loss càng đơn giản càng mạnh:** loss cuối của diffusion chỉ là ||nhiễu dự đoán − nhiễu thật||² — không còn dấu vết của ELBO, KL hay Jacobian. Bài học: một công thức lý thuyết phức tạp, sau khi tái tham số hoá đúng cách, có thể thành một bài toán hồi quy bình phương tối thiểu mà ai cũng train được.
- **Tái sử dụng ý tưởng qua các chương:** masked attention (ch. 12) → masked autoregressive flow (ch. 16); residual connection (ch. 11) → residual flow; U-Net (ch. 11) → backbone của diffusion; posterior/KL (ch. 5) → ELBO; AdaIN của StyleGAN (ch. 15) ≈ cách đưa time embedding và text embedding vào U-Net của diffusion. Deep learning là một bộ ít ý tưởng, phối lại nhiều lần.
- **Đánh giá mô hình sinh vẫn là vấn đề mở:** IS không thưởng đa dạng trong lớp, FID phụ thuộc mạng Inception, test likelihood không áp dụng cho GAN. Khi đọc một bài báo khoe "mô hình tốt nhất", luôn hỏi: đo bằng metric nào, trên dữ liệu nào, và metric đó có mù trước điểm yếu nào không.
- **Liên hệ thực tế với công việc video của kênh:** Flow/Veo (dùng để làm video motion 20:00) và các công cụ sinh ảnh (Kling, MiniMax…) đều là hậu duệ trực tiếp của diffusion — hiểu diffusion giúp hiểu vì sao prompt text điều khiển được ảnh/video (classifier-free guidance + text embedding), vì sao sinh chậm (T bước tuần tự — DDIM giảm xuống ~50 bước), và vì sao cùng một prompt cho nhiều kết quả khác nhau (bản chất ngẫu nhiên của quá trình đảo).
- **Notebook thực hành:** 14.1 Inception score · 15.1 GAN toy example · 15.2 Wasserstein distance · 16.1 1D normalizing flows · 16.2 Autoregressive flows · 16.3 Contraction mappings · 17.1 Latent variable models · 17.2 Reparameterization trick · 17.3 Importance sampling · 18.1 Diffusion encoder · 18.2 1D diffusion model · 18.3 Reparameterized model · 18.4 Families of diffusion models — xem tại https://udlbook.github.io/udlbook/.

**Tiếp theo:** phần 5 (chương 19–21: học tăng cường + đạo đức của AI) — phần cuối của sách.
