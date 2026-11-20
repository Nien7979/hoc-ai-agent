# G11 — Microsoft ML-For-Beginners: Phần 5 — NLP (bài 16–20)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Phần 2 (bài 05–09): `microsoft-ml-for-beginners-phan-2.md` — Regression + Web App (Flask + Pickle).
> Phần 3 (bài 10–13): `microsoft-ml-for-beginners-phan-3.md` — Classification + Web App (ONNX).
> Phần 4 (bài 14–15): `microsoft-ml-for-beginners-phan-4.md` — Clustering.
> Ghi chú này: đọc từ clone local `~/workspace/knowledge/ml-for-beginners/` (1,9 GB, HEAD 8e5c0dff). Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong repo khóa học.

---

## Bài 16 — Giới thiệu về Xử lý ngôn ngữ tự nhiên (NLP)

### NLP là gì
- **NLP** (Natural Language Processing) là nhánh của **ngôn ngữ học tính toán (computational linguistics)** — nghiên cứu cách máy tính làm việc với, hiểu, dịch và giao tiếp bằng ngôn ngữ con người.
- NLP có mặt khắp nơi: **đọc chính tả trên điện thoại, trợ lý ảo, chương trình soạn thảo** — bất cứ phần mềm nào "hiểu" chữ bạn gõ/nói đều có NLP bên trong.
- Máy tính giỏi áp dụng **quy tắc hình thức** (formal rules), nhưng hiểu **ý nghĩa** câu mới là phần khó — đặc biệt là hài hước, mỉa mai, cảm xúc.

### Lịch sử ngắn: Turing test → ELIZA
- **Alan Turing (những năm 1950):** "Turing test" — một người trò chuyện bằng chữ với một bên kia (người hoặc máy) mà không biết đối phương là ai; nếu sau một hồi trò chuyện mà người đó không phân biệt được thì có thể nói máy "suy nghĩ" được không? Ý tưởng lấy từ trò chơi tiệc tùng "The Imitation Game".
- **ELIZA (Joseph Weizenbaum, MIT, những năm 1960):** chương trình "nhà trị liệu" — phân tích câu, nhận diện từ khóa và cấu trúc ngữ pháp, rồi **đổi từ + đổi thì** để trả lời cho ra vẻ hiểu. Ví dụ: "I am **sad**" → "How long have **you been sad**?". Thực chất không hiểu gì cả — chỉ là thay từ và thêm từ. Dễ lừa: "You are a **bicycle**" → "How long have **I been** a **bicycle**?"

### Bài tập: bot trò chuyện "Marvin" (phiên bản đơn giản)
- Thư viện: **TextBlob** (`pip install -U textblob`, rồi `python -m textblob.download_corpora`) — thư viện xử lý văn bản đơn giản, đứng trên vai NLTK và pattern.
- Bot chỉ có một khả năng: trả lời **ngẫu nhiên** từ danh sách câu chung chung ("That is quite interesting, please tell me more.", "I see. Do go on.", "Why do you say that?"…). Vòng lặp: in hướng dẫn → nhận input → nếu gõ `bye` thì thoát → in câu trả lời ngẫu nhiên → lặp lại.
- Câu hỏi suy ngẫm: câu trả lời ngẫu nhiên có "lừa" được ai tưởng bot hiểu mình không? Bot cần thêm tính năng gì để hiệu quả hơn? Nếu bot thật sự hiểu ý nghĩa câu, nó có cần **nhớ** các câu trước trong cuộc trò chuyện không?

---

## Bài 17 — Các task và kỹ thuật NLP thường gặp

### Chuỗi task xử lý văn bản (thường làm theo thứ tự)
- **Tokenization:** tách văn bản thành token (từ). Nghe đơn giản nhưng dấu câu và cách phân tách từ/câu khác nhau giữa các ngôn ngữ làm nó phức tạp.
- **Embeddings:** chuyển text thành **số** sao cho từ đồng nghĩa / hay đi cùng nhau thì gom cụm lại gần nhau. Thử bằng TensorFlow Projector: 'toy' gom cụm với 'disney', 'lego', 'playstation', 'console'.
- **Parsing & Part-of-speech Tagging:** gán mỗi từ đã tokenize thành danh/từ, động từ, tính từ… (`fox` = noun, `jumped` = verb). Parsing = nhận ra từ nào liên quan tới từ nào (`the quick red fox jumped` là chuỗi tính từ–danh từ–động từ, tách biệt với `lazy brown dog`).
- **Word and Phrase Frequencies:** xây từ điển đếm tần suất từng từ/cụm. Trong khổ thơ Kipling: `a friend` xuất hiện 2 lần, `the` 6 lần, `travels` 2 lần.
- **N-grams:** tách text thành chuỗi n từ liên tiếp — unigram (1 từ), bigram (2 từ), trigram (3 từ)… như một **cửa sổ trượt** trên câu. Với n = 2, câu "the quick red fox jumped over the lazy brown dog" cho 9 bigram: the quick, quick red, red fox…
- **Noun phrase Extraction:** xác định cụm danh từ là chủ ngữ/tân ngữ của câu (thường có 'a'/'an'/'the' đứng trước). Trong câu ví dụ: **quick red fox** và **lazy brown dog**.
- **Sentiment analysis:** đo cảm xúc — **polarity** từ −1,0 (tiêu cực) tới 1,0 (tích cực), **subjectivity** từ 0,0 (khách quan nhất) tới 1,0 (chủ quan nhất).
- **Inflection:** lấy dạng số ít/số nhiều của từ.
- **Lemmatization:** lemma = từ gốc — *flew, flies, flying* đều có lemma là *fly*.
- **WordNet** (Đại học Princeton): cơ sở dữ liệu từ, từ đồng nghĩa, trái nghĩa… cho nhiều ngôn ngữ — cực hữu ích khi làm dịch, kiểm tra chính tả, công cụ ngôn ngữ.

### Thực hành với TextBlob
- Trích xuất noun phrase bằng `ConllExtractor` — dùng chunk parsing, train trên corpus ConLL-2000 (Wall Street Journal, phần 15–18 train, phần 20 test):
```python
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()
user_input_blob = TextBlob(user_input, np_extractor=extractor)
np = user_input_blob.noun_phrases
```
- **Bot Marvin phiên bản "cảm thông":** phân tích sentiment của input → 4 mức phản hồi:
```python
if user_input_blob.polarity <= -0.5:
    response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
    response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
    response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
    response = "Wow, that sounds great. "
```
- Nếu phát hiện noun phrase → **số nhiều hóa** nó rồi hỏi thêm về chủ đề đó: "I went for a walk and saw a lovely cat" → "Well, that sounds positive. Can you tell me more about **lovely cats**?"
- Câu hỏi kiểm tra: phản hồi cảm thông có làm bot "đáng tin" hơn không? Trích noun phrase có ích gì?

---

## Bài 18 — Dịch và phân tích cảm xúc bằng ML

### Dịch máy: quy tắc vs ML
- Dịch là bài toán rất khó: hàng nghìn ngôn ngữ, ngữ pháp khác nhau.
- **Cách tiếp cận quy tắc:** gán tag từ (danh/động/tính) → dịch từng từ → sai. Ví dụ: "I feel happy" (chủ ngữ–động từ–tính từ) sang tiếng Ireland thành "Tá athas orm" — nghĩa đen "Happy is upon me" (động từ–tính từ–chủ ngữ: cảm xúc "nằm trên" bạn). Dịch từng từ ngây thơ ra câu sai hoàn toàn: "Mise bhraitheann athas" ("me feel happy").
- **Cách tiếp cận ML:** bỏ qua ý nghĩa từng từ, dùng **corpus song song** (văn bản + bản dịch của con người) để phát hiện pattern thành ngữ. Ví dụ "false cognate": "I have no money" dịch từng từ sang Pháp thành "Je n'ai pas de monnaie" (monnaie = tiền lẻ) — bản dịch đúng của con người là "Je n'ai pas d'argent".
- `TextBlob.translate()` thực chất dùng **Google Translate** phía sau (cần internet). Bản dịch câu mở đầu *Pride and Prejudice* của nó được cho là **chính xác hơn cả bản dịch Pháp năm 1932** của con người — vì dịch giả 1932 đã tự thêm chữ vào miệng tác giả "cho rõ nghĩa".

### Phân tích cảm xúc: từ điển vs ML
- **Cách không dùng ML:** liệt kê từ tích cực/tiêu cực, cộng điểm. Dễ bị lừa bởi **mỉa mai**: "Great, that was a wonderful waste of time, I'm glad we are lost on this dark road" — từ 'great', 'wonderful', 'glad' bị tính là tích cực dù câu này tiêu cực. Con người truyền mỉa mai bằng **ngữ điệu** — máy không nghe được ngữ điệu từ text.
- **Cách ML:** thu thập văn bản đã gán nhãn (tweet, review phim kèm điểm số) → pattern tự hiện ra ("Oscar worthy" xuất hiện nhiều ở review tích cực, "gourmet" nhiều hơn "disgusting" ở review nhà hàng tốt). Ví dụ ứng dụng: phân loại email cử tri **ủng hộ / phản đối** một dự luật — train bằng một phần email mỗi phe, model học pattern từ/cụm từ của từng phe (nó **không hiểu nội dung**, chỉ biết từ nào hay xuất hiện ở phe nào), test bằng email chưa dùng để train.
- **TextBlob sentiment:** `polarity` −1..1, `subjectivity` 0..1. Câu mở đầu *Pride and Prejudice*: polarity 0,21, subjectivity 0,27. Câu cuối: polarity 0,7, subjectivity 0,8.
- **Thử thách:** đếm trong *Pride and Prejudice* có bao nhiêu câu **tuyệt đối tích cực** (polarity = 1) và tuyệt đối tiêu cực (polarity = −1). Câu hỏi kiểm tra quan trọng: một số câu được chấm +1 nhưng đọc kỹ thì **không tích cực** — vì Jane Austen viết đầy **mỉa mai** (nhân vật Elizabeth hay "tuyên bố những ý kiến mà thực ra không phải của mình"). Sentiment dựa trên từ không hiểu được mỉa mai.

---

## Bài 19 — Phân tích cảm xúc review khách sạn 1: khám phá dữ liệu

### Dataset: 515K Hotel Reviews ở châu Âu
- Nguồn: Kaggle, **CC0 (public domain)**, scrape từ Booking.com, tác giả Jiashen Liu. **515.738 dòng × 17 cột**, 1.493 khách sạn, 6 thành phố, ~230 MB giải nén.
- Cột chính: `Hotel_Name, Hotel_Address, lat, lng`; `Average_Score, Total_Number_of_Reviews, Additional_Number_of_Scoring`; `Reviewer_Score` (2,5–10); `Negative_Review` ("No Negative" nếu trống), `Positive_Review` ("No Positive"); số từ mỗi loại review; `Review_Date, days_since_review`; `Tags`; `Total_Number_of_Reviews_Reviewer_Has_Given`; `Reviewer_Nationality`.
- Mục tiêu giả định: xây **bot gợi ý khách sạn** bằng sentiment + điểm review.

### Các bẫy trong dữ liệu thật (phần quan trọng nhất bài này)
- **Người review có thể viết nội dung tích cực vào cột Negative_Review** ("there is nothing bad about this hotel") và ngược lại ("there is nothing good about this hotel at all"). Đếm số từ ≠ đo cảm xúc.
- **Điểm thấp nhất là 2,5, không phải 0** — và một review chê thậm tệ vẫn cho 2,5 điểm. Đừng giả định thang điểm bắt đầu từ 0.
- **Tags không chuẩn hóa:** 1.025 giá trị khác nhau chỉ riêng cho "loại nhóm khách"; tên phòng thì vô vàn biến thể (Single room, Deluxe Single Room, Classic Queen Room…). Muốn dùng tags phải xử lý rất nhiều.
- **Các cột tổng hợp không đáng tin:** `Average_Score` được định nghĩa kỳ lạ ("dựa trên bình luận mới nhất trong năm qua"). Ví dụ khách sạn Britannia: dataset ghi 7,1 nhưng **tự tính trung bình từ review trong dataset ra 6,8**. `Total_Number_of_Reviews` ghi 9.086 nhưng chỉ có 4.789 dòng trong dataset + 2.682 `Additional_Number_of_Scoring` = 7.471, vẫn thiếu 1.615. → **Đừng tin cột tổng hợp sẵn; tự tính lại từ dữ liệu thô.**
- **Reviewer_Nationality:** 227 quốc tịch; UK chiếm 245.246 review (gần nửa), Mỹ 35.437, Úc 21.686… Cảnh báo: **đừng đưa định kiến quốc tịch vào model** — mỗi reviewer là cá nhân, điểm số chịu ảnh hưởng bởi chuyến đi, tính khí, trải nghiệm trước đó, không phải quốc tịch.
- **Số liệu đếm được:** "No Negative" 127.890 dòng; "No Positive" 35.946 dòng; cả hai đều trống 127 dòng (chỉ cho điểm, không viết gì — 0,02%, không làm lệch model nhưng đáng biết).
- **Bài học hiệu năng:** đếm bằng `apply` + lambda mất **9,64 giây**; đếm bằng phép **sum vector hóa** của pandas mất **0,19 giây** — cùng kết quả, nhanh gấp ~50 lần. Ưu tiên phép toán vector hóa khi làm việc với dataframe lớn.
- **Lưu ý nhân văn:** một số review tiêu cực có nội dung phân biệt chủng tộc/giới tính/tuổi tác. **Để code đo sentiment, đừng tự đọc** — tránh khó chịu không cần thiết.

---

## Bài 20 — Review khách sạn 2: lọc dữ liệu + chạy sentiment

### Lọc và sửa cột
- Bỏ `lat, lng`. `Hotel_Address` → chỉ giữ thành phố + quốc gia (6 thành phố: Amsterdam 105 khách sạn, Barcelona 211, London 400, Milan 162, Paris 458, Vienna 158).
- Bỏ `Additional_Number_of_Scoring`; **thay** `Total_Number_of_Reviews` và `Average_Score` bằng giá trị tự tính (`groupby('Hotel_Name').transform('count'/'mean')`).
- Bỏ các cột đếm từ, `Review_Date`, `days_since_review`, `Total_Number_of_Reviews_Reviewer_Has_Given`. Giữ `Reviewer_Score`, `Negative_Review`, `Positive_Review`, `Tags` (tạm thời), `Reviewer_Nationality`.

### Xử lý cột Tags — chỉ giữ 2 loại
- Tags là list dạng text, thứ tự và số lượng không cố định, **2.428 tag khác nhau**. Chỉ giữ **loại chuyến đi + loại nhóm khách**; bỏ loại phòng, thiết bị gửi review, số đêm ở.
- Cách làm nhanh: bỏ ngoặc/kép bằng `str.strip` + `str.replace`, tách thành 6 cột tạm theo thứ tự, gộp lại rồi `value_counts()`:
  - Tag hữu ích: Leisure trip 417.778; Couple 252.294; Solo traveler 108.545; Business trip 82.939; Group (+ Travellers with friends) 67.535; Family with young children 61.015; Family with older children 26.349; With a pet 1.405.
- Tạo cột nhị phân 0/1 cho từng tag:
```python
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
# ... tương tự cho Solo_traveler, Business_trip, Group, Family_with_young_children,
# Family_with_older_children, With_a_pet
```
- Lưu thành `Hotel_Reviews_Filtered.csv`.

### Loại bỏ stop words trước khi chạy sentiment
- **Stop words** = từ thông dụng không làm đổi cảm xúc câu (the, is, and…). Loại chúng đi giúp sentiment chạy nhanh hơn mà không kém chính xác.
- Review tiêu cực dài nhất: 395 từ → còn **195 từ** sau khi loại stop words. Xử lý 515K dòng × 2 cột chỉ mất **3,3 giây** trên máy test.
```python
from nltk.corpus import stopwords
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    return " ".join([word for word in review.split() if word not in cache])
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Chạy sentiment bằng VADER
- **VADER** (Hutto & Gilbert, 2014) — mô hình rule-based cho sentiment của text mạng xã hội, có sẵn trong NLTK (`nltk.download('vader_lexicon')`):
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_sentiment = SentimentIntensityAnalyzer()

def calc_sentiment(review):
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]

df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
```
- Chạy hết 515K dòng mất ~120 giây. Lưu thành `Hotel_Reviews_NLP.csv`.
- **Kiểm chứng:** so sánh sentiment với `Reviewer_Score` của cùng review. Sẽ có trường hợp sai — thường giải thích được, ví dụ mỉa mai: "Of course I LOVED sleeping in a room with no heating" bị chấm tích cực dù người viết đang chê.
- **Pipeline hoàn chỉnh:** `Hotel_Reviews.csv` → notebook khám phá → notebook lọc → `Hotel_Reviews_Filtered.csv` → notebook sentiment → `Hotel_Reviews_NLP.csv` → thử thách cuối: dùng clustering… để tìm pattern quanh sentiment.
- Sắp xếp lại thứ tự cột cho con người dễ đọc (chỉ là cosmetic).

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Phân tích cảm xúc bình luận kênh:** polarity + subjectivity cho từng bình luận → tự động phân loại (khen / hỏi / chê / spam) để ưu tiên trả lời. **Lưu ý quan trọng: VADER/TextBlob là từ điển tiếng Anh — bình luận tiếng Việt cần từ điển tiếng Việt hoặc LLM.** Phương pháp thì chuyển được, từ điển thì không.
2. **N-grams cho kịch bản:** đếm bigram/trigram trong các kịch bản video hiệu quả nhất — cụm từ nào hay xuất hiện ở video nhiều view? Đó là cách tìm "công thức câu chữ" bằng dữ liệu thay vì cảm tính.
3. **Mỉa mai là điểm mù của sentiment từ điển:** "hay quá ha" trong bình luận tiếng Việt có thể là khen hoặc chê — đừng auto-reply dựa trên sentiment ngây thơ. Cần đọc ngữ cảnh hoặc dùng model hiểu ngữ cảnh.
4. **Đếm từ ≠ đo cảm xúc:** bình luận dài chưa chắc đã tiêu cực hơn bình luận ngắn; review khách sạn đã chứng minh điều này.
5. **Stop words + vector hóa = xử lý nhanh dữ liệu lớn:** khi xử lý hàng nghìn bình luận, loại stop words trước và dùng phép toán vector hóa của pandas thay vì `apply` từng dòng (nhanh gấp ~50 lần).
6. **Tự tính lại, đừng tin cột tổng hợp sẵn:** bài học từ `Average_Score` của khách sạn — với dữ liệu YouTube Studio export cũng vậy: tự tính trung bình/tỉ lệ từ dữ liệu thô thay vì tin con số có sẵn.
7. **Bot Marvin = mẫu cho trợ lý trả lời bình luận:** sentiment + trích noun phrase → trả lời có nhắc lại đúng thứ khán giả nói ("kể thêm về con mèo dễ thương đi") thay vì câu chung chung — khán giả cảm thấy được lắng nghe thật.

## Tiến độ khóa G11
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ Phần 2 (bài 05–09): Regression + Web App — xong 20/11.
- ✅ Phần 3 (bài 10–13): Classification — xong 20/11.
- ✅ Phần 4 (bài 14–15): Clustering — xong 20/11.
- ✅ **Phần 5 (bài 16–20): NLP — xong 20/11 (ghi chú này).**
- ⏳ Phần 6: Time Series (bài 21–23).
- ⏳ Phần 7: Reinforcement Learning (bài 24–25) + Postscript.
- ⏳ Sau G11: S1 "Understanding Deep Learning" (sách PDF miễn phí), rồi các khóa đã đăng ký khác (freeCodeCamp Python v9, Udemy 73 khóa, GLM-5.3-Flash build + train) — **quy tắc của Hoai (20/11 ~23:19): khóa nào đến bước cần anh làm (thi chứng chỉ, nhập token,…) thì BỎ QUA, học tiếp khóa khác.**

## Nguồn
- Bài 16: https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/README.md
- Bài 17: https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/README.md
- Bài 18: https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/README.md
- Bài 19: https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/README.md
- Bài 20: https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
