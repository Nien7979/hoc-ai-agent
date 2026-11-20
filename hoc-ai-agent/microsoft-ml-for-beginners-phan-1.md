# Bài đã học: Microsoft ML-For-Beginners — Phần 1 (Bài 01–04: Introduction)

- Nguồn: Microsoft, "ML-For-Beginners" (MIT, 92.346 ⭐, 22.954 fork), repo https://github.com/microsoft/ML-For-Beginners — 12 tuần, 26 bài, 52 quiz, học ML cổ điển bằng scikit-learn (không deep learning — deep learning có khóa riêng "AI for Beginners").
- Có bản dịch TIẾNG VIỆT (translations/vi/) cùng 50+ ngôn ngữ khác.
- Ngày học: 20/11/2026 (đêm). Đã đọc toàn bộ 4 bài phần Introduction trên GitHub. Repo đang clone về máy (~/workspace/knowledge/ml-for-beginners/) — repo lớn do chứa 50+ bản dịch nên clone chậm, sẽ xong sau.
- Đây là phần đầu của G11 trong lộ trình (sau top 5 AI Agent). Còn lại: Regression (05–08), Web App (09), Classification (10–13), Clustering (14–15), NLP (16–20), Time Series (21–23), Reinforcement Learning (24–25), Postscript Real-World.

## Bài 01 — Introduction to Machine Learning

- **Định nghĩa**: ML là tập con của AI — dùng thuật toán chuyên biệt để khám phá thông tin có ý nghĩa và tìm mẫu ẩn trong dữ liệu, phục vụ việc ra quyết định hợp lý. Gần gũi với cách não trẻ em học: cảm nhận → phát hiện mẫu → rút ra quy tắc.
- **AI vs ML vs Deep Learning vs Data Science**: AI là tập lớn nhất (máy bắt chước hành vi thông minh); ML nằm trong AI; Deep Learning nằm trong ML (mạng neural nhiều lớp); Data Science giao thoa với cả ba.
- **Khóa này chỉ dạy ML cổ điển bằng scikit-learn.** Không dạy deep learning, neural network, AI tổng quát.
- **Vì sao học ML**: tạo hệ thống tự động học mẫu ẩn từ dữ liệu để hỗ trợ quyết định thông minh. Doanh nghiệp dùng ML thay cho rules engine hard-code khi biến số quá nhiều (vd: dự đoán tử vong từ lịch sử sức khỏe, dự báo thời tiết tháng 4 từ vĩ độ, khí hậu, dòng jet stream...).
- **Chất lượng dữ liệu quyết định tất cả**: dữ liệu kém/nhiễu → dự đoán sai, dù thuật toán có xịn.
- **Ứng dụng**: dự đoán bệnh từ bệnh án, dự báo thời tiết, phân tích cảm xúc văn bản, phát hiện tin giả; tài chính, kinh tế, khoa học trái đất, vũ trụ, y sinh, khoa học nhận thức, cả nhân văn.
- **Chuẩn bị**: cài Python, text editor, Node.js + npm (dùng khi build web app), tài khoản GitHub (fork repo), làm quen scikit-learn.

## Bài 02 — The History of Machine Learning

- **Nền tảng toán học có từ hàng trăm năm trước khi có AI**: định lý Bayes (1763), phương pháp bình phương tối thiểu — least squares (Legendre, 1805), chuỗi Markov (1913), Perceptron (Rosenblatt, 1957 — bộ phân loại tuyến tính, tiền thân của deep learning), Nearest Neighbor (1967), Backpropagation (1970), RNN (1982).
- **1950 — Turing**: "máy biết suy nghĩ", Turing Test (sẽ gặp lại ở phần NLP).
- **1956 — Dartmouth Summer Research Project**: thuật ngữ "artificial intelligence" ra đời. McCarthy: "mọi khía cạnh của học tập hay trí tuệ đều có thể mô tả chính xác tới mức máy có thể mô phỏng được."
- **1956–1974 "Golden years"**: lạc quan cao, Minsky (1967) tin rằng "trong một thế hệ, vấn đề tạo ra AI sẽ cơ bản được giải quyết". Có Shakey the robot, ELIZA (chatbot "nhà trị liệu" đầu tiên), Blocks World/SHRDLU.
- **1974–1980 "AI Winter"**: độ phức tạp bị đánh giá thấp, compute quá yếu, tham số bùng nổ theo cấp số nhân (combinatorial explosion), thiếu dữ liệu. Tài trợ cạn, niềm tin giảm. Phòng "Chinese room" đặt câu hỏi về Turing Test. Phân hóa trường phái "scruffy" (tinh chỉnh tới khi chạy được — ELIZA, SHRDLU) vs "neat" (logic, giải quyết vấn đề hình thức — dần thắng thế vì kết quả giải thích được, tái tạo được).
- **1980s — Expert systems**: hệ lai (rules engine + inference engine), dạng AI đầu tiên thành công thực sự về mặt kinh doanh.
- **1987–1993 "AI Chill"**: hardware chuyên dụng cho expert system quá đặc thù; PC cá nhân lên ngôi, mở đường cho big data.
- **1993–2011**: dữ liệu tăng vọt (smartphone từ 2007), compute tăng theo cấp số nhân, thuật toán tiến hóa — ngành trưởng thành thành một lĩnh vực thật sự.
- **Hiện tại**: ML/AI chạm vào gần như mọi mặt đời sống; cần hiểu rủi ro + trách nhiệm (Brad Smith, Microsoft: công nghệ đặt ra vấn đề về quyền con người như riêng tư, tự do ngôn luận).

## Bài 03 — Fairness and Machine Learning (Responsible AI)

- ML đã tham gia quyết định hàng ngày: chẩn đoán bệnh, duyệt vay, phát hiện gian lận. Model phải đáng tin — và phải giải thích được hành vi của nó khi sai.
- **6 nguyên tắc Responsible AI của Microsoft**:
  1. **Fairness (công bằng)**: đối xử giống nhau với các nhóm tương tự. Thiên kiến của con người lọt vào dữ liệu train — thường là vô tình. 5 loại harm: allocation (ưu ái nhóm này hơn nhóm khác), quality of service (train cho 1 kịch bản, thực tế phức tạp hơn — vd vòi xà phòng không nhận diện được da tối màu), denigration (gán nhãn sai — vd nhận diện ảnh gán người da tối là khỉ đột), over/under-representation (nhóm nào đó vắng mặt trong nghề nghiệp nào đó), stereotyping (dịch Anh–Thổ gán giới tính theo khuôn mẫu).
  2. **Reliability & safety (tin cậy + an toàn)**: hoạt động ổn định cả trong điều kiện bình thường lẫn bất thường/ngoại lệ. Xe tự lái phải tính tới đêm, bão, trẻ chạy qua đường, thú cưng, công trường...
  3. **Inclusiveness (bao hàm)**: thiết kế cho mọi người, kể cả 1 tỷ người khuyết tật trên thế giới — gỡ rào cản là cơ hội đổi mới.
  4. **Security & privacy (bảo mật + riêng tư)**: dữ liệu từ đâu, có phải do user gửi hay công khai? Phải chống rò rỉ + tấn công. Căng thẳng cố hữu: càng nhiều dữ liệu cá nhân thì hệ thống càng "cá nhân hóa" tốt — nhưng càng rủi ro riêng tư. GDPR là động lực lớn.
  5. **Transparency (minh bạch)**: giải thích được model làm gì và vì sao. "Glass box" thay vì "black box". Ngân hàng dùng AI duyệt vay phải chỉ ra được dữ liệu nào ảnh hưởng tới quyết định. Chính phủ bắt đầu quản lý AI theo ngành.
  6. **Accountability (trách nhiệm giải trình)**: người thiết kế + triển khai phải chịu trách nhiệm về hệ thống. Đặc biệt nhạy cảm với nhận diện khuôn mặt (tìm trẻ mất tích vs giám sát hàng loạt).
- **Impact assessment (đánh giá tác động) trước khi train**: mục đích hệ thống là gì, dùng ở đâu, ai tương tác; tác động xấu tới cá nhân; yêu cầu dữ liệu (GDPR/HIPAA, nguồn + lượng đủ không); liệt kê harm tiềm ẩn; đối chiếu với 6 nguyên tắc.
- **Debugging với Responsible AI dashboard**: error analysis (phân bố lỗi), model overview (hiệu năng khác nhau giữa các nhóm dữ liệu), data analysis (phân bố dữ liệu, bias), model interpretability (cái gì ảnh hưởng tới dự đoán).

## Bài 04 — Techniques of Machine Learning (quy trình làm ML)

- **Quy trình 7 bước**: (1) Đặt câu hỏi — câu hỏi mà rules engine đơn giản không trả lời được, thường là dự đoán từ dữ liệu. (2) Thu thập + chuẩn bị dữ liệu — chất lượng + số lượng quyết định độ chính xác; visualize dữ liệu; chia train/test. (3) Chọn phương pháp train — tùy câu hỏi + bản chất dữ liệu, cần thử nghiệm. (4) Train model — cho model học mẫu trong dữ liệu (điều chỉnh trọng số nội bộ). (5) Đánh giá — dùng dữ liệu test chưa từng thấy. (6) Tune tham số — đổi hyperparameter, làm lại. (7) Dự đoán — dùng dữ liệu hoàn toàn mới.
- **Feature (X) và target (y)**: feature là thuộc tính đo được của dữ liệu (cột như date/size/color); target là thứ cần dự đoán (vd: tháng 12, bí ngô màu nào rẻ nhất? khu nào ở San Francisco giá nhà tốt nhất?). Target còn gọi là label.
- **Feature selection vs feature extraction**: selection = chọn tập con của feature gốc; extraction = tạo feature mới từ hàm của feature gốc.
- **Visualize dữ liệu** (Seaborn/Matplotlib): phát hiện tương quan ẩn, phát hiện bias/mất cân bằng.
- **Chia dataset**: train (phần lớn — fit model), test (độc lập — kiểm tra hiệu năng), validation (nhỏ hơn — tune hyperparameter/kiến trúc; không phải lúc nào cũng cần).
- **Train = `model.fit(X, y)`**: đưa feature X + target y vào. Train có thể mất nhiều vòng lặp (epochs) với model lớn.
- **Overfitting vs underfitting**: overfit = học quá kỹ cả chi tiết + nhiễu của dữ liệu train → dự đoán train rất tốt nhưng dữ liệu mới thì kém; underfit = học chưa đủ → cả train lẫn dữ liệu mới đều kém.
- **Prediction**: dùng dữ liệu mới hoàn toàn; trong ứng dụng thực tế thì input của user (vd bấm nút) được đưa vào model để inference.

## Áp dụng cho công việc của Niên và Hoai

- **Quy trình 7 bước (bài 04) = khung chuẩn cho mọi tool/app ML sau này.** Viết tool phân loại video, gợi ý kịch bản, dự đoán video nào sẽ hot — đều đi theo: đặt câu hỏi → thu thập dữ liệu (vd 35 video đã đăng + số liệu) → chọn phương pháp → train → đánh giá trên dữ liệu chưa thấy → tune → dự đoán. Không có bước nào được bỏ qua, đặc biệt là chia test set.
- **Feature/target**: để dự đoán video nào hiệu quả, phải xác định feature trước (giờ đăng, độ dài, chủ đề, cảm xúc, có twist không...) và target (view/lượt thích/bình luận). Không có feature tốt thì model nào cũng vô dụng.
- **Data quality > algorithm**: 35 video + 5 comment của kênh là dữ liệu nhỏ nhưng thật — tốt hơn dữ liệu lớn mà nhiễu. Ghi chép cẩn thận từ đầu (queue JSON, comments-seen.json) chính là đang xây dataset.
- **Overfitting**: cảnh báo cho việc "học thuộc" 1–2 video hot rồi áp dụng máy móc cho mọi video sau — model phải tổng quát được, không học vẹt.
- **Responsible AI (bài 03) áp dụng trực tiếp**: kênh dùng AI tạo nhân vật/kịch bản — phải công bằng (không gán khuôn mẫu giới tính/vùng miền cho nhân vật), minh bạch (khai báo AI disclosure khi đăng — đã làm), bảo mật (không dùng dữ liệu cá nhân thật của khán giả), chịu trách nhiệm (Niên duyệt trước khi đăng).
- **Lịch sử (bài 02)**: AI Winter nhắc rằng hype rồi sẽ qua — giá trị thật nằm ở dữ liệu + bài toán đúng, không phải ở việc chạy theo công nghệ mới nhất. Đúng tinh thần "học để dùng được" của Hoai.
- **Bước tiếp theo**: Regression (bài 05–08, dữ liệu giá bí ngô Bắc Mỹ) — bài thực hành đầu tiên với scikit-learn, chạy được trên máy thường, không cần GPU.
