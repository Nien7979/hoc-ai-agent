# G11 — Microsoft ML-For-Beginners: Phần 4 — Clustering (bài 14–15)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Phần 2 (bài 05–09): `microsoft-ml-for-beginners-phan-2.md` — Regression + Web App (Flask + Pickle).
> Phần 3 (bài 10–13): `microsoft-ml-for-beginners-phan-3.md` — Classification + Web App (ONNX).
> Ghi chú này: đọc từ clone local `~/workspace/knowledge/ml-for-beginners/` (1,9 GB, HEAD 8e5c0dff). Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong repo khóa học.

---

## Bài 14 — Giới thiệu về Clustering

### Clustering là gì — học KHÔNG giám sát
- **Clustering** là **Unsupervised Learning**: dataset **không có nhãn**, đầu vào không gắn với đầu ra định trước. Thuật toán tự sắp xếp dữ liệu chưa gán nhãn thành các nhóm theo pattern nó nhận ra.
- Đời thường: gom đống quần áo bẩn phân loại theo từng thành viên trong nhà 🧦👕👖. Trong data science: phân tích sở thích người dùng, tìm đặc điểm của dataset chưa gán nhãn. **Clustering giúp hiểu được sự hỗn loạn** — như ngăn kéo tất lộn xộn.
- Ứng dụng thực tế:
  - **Phân khúc thị trường** (market segmentation): nhóm tuổi nào mua món gì.
  - **Phát hiện bất thường** (anomaly detection): phát hiện gian lận từ giao dịch thẻ tín dụng.
  - Phát hiện khối u trong loạt ảnh scan y tế.
  - Gom nhóm kết quả tìm kiếm (link mua sắm, ảnh, review).
  - Giảm kích thước dataset lớn để phân tích chi tiết hơn — học về dữ liệu **trước khi** xây model khác.
- Gốc gác thú vị: phân tích cụm (cluster analysis) bắt nguồn từ **Nhân loại học và Tâm lý học những năm 1930**.
- Mẹo thực tế: sau khi gom cụm, gán cho mỗi điểm một **cluster Id** — có thể dùng Id này thay cho dữ liệu nhận dạng nhạy cảm, **bảo vệ quyền riêng tư** của dataset.

### Các phương pháp clustering trong scikit-learn

| Phương pháp | Dùng khi nào |
|---|---|
| **K-Means** | đa năng, inductive |
| Affinity propagation | nhiều cụm không đều, inductive |
| Mean-shift | nhiều cụm không đều, inductive |
| Spectral clustering | ít cụm đều nhau, transductive |
| Ward hierarchical clustering | nhiều cụm có ràng buộc, transductive |
| Agglomerative clustering | nhiều cụm, có ràng buộc, khoảng cách non-Euclidean, transductive |
| **DBSCAN** | hình học non-flat, cụm không đều, transductive |
| OPTICS | hình học non-flat, cụm không đều với mật độ thay đổi, transductive |
| Gaussian mixtures | hình học flat, inductive |
| BIRCH | dataset lớn có outlier, inductive |

- **Từ vựng cần nhớ:**
  - **Transductive vs Inductive:** suy luận transductive đi từ các case huấn luyện đã quan sát tới các test case cụ thể; inductive đi từ case huấn luyện tới **quy tắc chung** rồi mới áp cho test case. Ví dụ: dataset chỉ gán nhãn một phần ("records", "cds", còn lại trống). Inductive: train model nhận diện "records"/"cds" rồi gán cho dữ liệu chưa nhãn — sẽ lúng túng với thứ thực ra là "cassettes". Transductive: gom các món giống nhau lại rồi gán nhãn cho cả nhóm ("đồ nhạc hình tròn", "đồ nhạc hình vuông") — xử lý dữ liệu chưa biết tốt hơn.
  - **Flat vs non-flat geometry:** "flat" = hình học Euclid (khoảng cách = đoạn thẳng giữa 2 điểm trên mặt phẳng); "non-flat" = hình học phi Euclid (khoảng cách đo dọc theo đường cong). Dữ liệu nhìn không nằm trên một mặt phẳng thì cần thuật toán chuyên biệt.
  - **Distances:** cụm được định nghĩa bằng ma trận khoảng cách. Khoảng cách Euclid: cụm có **centroid** (điểm trung tâm = trung bình các điểm); khoảng cách đo tới centroid. Khoảng cách non-Euclidean: dùng **clustroid** (điểm gần các điểm khác nhất).
  - **Constrained clustering:** đưa học bán giám sát vào clustering — gắn quan hệ "cannot link" / "must-link" giữa các điểm để ép thuật toán chọn tốt hơn (ví dụ: "món này phải làm bằng nhựa", "món này phải phát được nhạc").
  - **Density (mật độ):** dữ liệu "nhiễu" (noisy) được coi là "dày đặc" (dense). Khoảng cách giữa các điểm trong cụm có thể dày/thưa khác nhau → cần đúng phương pháp (K-Means vs HDBSCAN cho dataset nhiễu, mật độ cụm không đều).

### 5 họ thuật toán clustering lớn
- **Hierarchical (phân cấp):** gom cụm theo độ gần với vật lân cận — Agglomerative clustering của scikit-learn thuộc họ này.
- **Centroid (tâm cụm):** chọn **k** (số cụm), thuật toán tìm điểm trung tâm rồi gom dữ liệu quanh đó. **K-Means** là bản phổ biến nhất — trung tâm = mean gần nhất, tối thiểu hóa khoảng cách bình phương tới cụm.
- **Distribution-based (phân phối):** mô hình thống kê — tính **xác suất** một điểm thuộc về cụm nào rồi gán theo đó. Gaussian mixture thuộc họ này.
- **Density-based (mật độ):** gán điểm vào cụm theo mật độ gom nhóm; điểm xa nhóm = outlier/noise. **DBSCAN, Mean-shift, OPTICS** thuộc họ này.
- **Grid-based (lưới):** chia dataset nhiều chiều thành các ô lưới → tạo cụm.

### Bài tập: trực quan hóa dữ liệu nhạc Nigeria
- Dataset: `nigerian-songs.csv` — **530 bài hát**, 16 cột (name, album, artist, artist_top_genre, release_date, length, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature). Dữ liệu từ Spotify.
- `df.info()`: không có null. `df.describe()`: popularity trung bình 17,5 (max 73), danceability trung bình 0,74, tempo trung bình 116 BPM.
- Top genres: afro dancehall, afropop, nigerian pop chiếm đa số. Lọc bỏ genre `'Missing'` (Spotify không phân loại) và bài có `popularity = 0` (không được xếp hạng = nhiễu).
- Chỉ giữ 3 thể loại chính: `afro dancehall`, `afropop`, `nigerian pop`.
- Heatmap tương quan: tương quan mạnh duy nhất là **energy ↔ loudness** (nhạc to thường nhiều năng lượng — đương nhiên). Còn lại tương quan yếu. **Tương quan không có nghĩa là nhân quả!**
- **Jointplot (KDE)** popularity vs danceability: xuất hiện các **vòng tròn đồng tâm** — các thể loại hội tụ quanh một điểm chung. Có phải gu nghe nhạc Nigeria hội tụ ở một mức danceability nhất định? Scatter plot (FacetGrid) cho thấy pattern tương tự.
- Scatterplot là công cụ chủ lực để **nhìn thấy cụm** — cần thành thạo trước khi chạy thuật toán.

---

## Bài 15 — K-Means Clustering

### K-Means là gì
- Phương pháp từ lĩnh vực **xử lý tín hiệu**: chia dữ liệu thành **k cụm**, mỗi điểm được gom vào cụm có **mean (trung tâm) gần nhất**.
- Cụm có thể hình dung bằng **sơ đồ Voronoi**: mỗi điểm "seed" (hạt giống) + vùng tương ứng của nó.
- **Quy trình 3 bước:**
  1. Chọn k điểm trung tâm bằng cách lấy mẫu từ dataset.
  2. Lặp:
     - Gán mỗi mẫu vào **centroid gần nhất**.
     - Tạo centroid mới = **giá trị trung bình** của tất cả mẫu đã gán cho centroid cũ.
     - Tính chênh lệch giữa centroid mới và cũ, lặp lại cho tới khi centroid **ổn định**.
- Nhược điểm: phải **tự chọn k** (số cụm) trước. May là có **elbow method** để ước lượng k tốt (xem dưới).

### Chuẩn bị: chọn cột + encode
```python
import seaborn as sns
# Boxplot từng cột: dữ liệu hơi nhiễu, có outlier ở nhiều cột.
# (Có thể loại outlier, nhưng dữ liệu sẽ còn rất ít — bài này giữ nguyên.)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = df.loc[:, ('artist_top_genre', 'popularity', 'danceability', 'acousticness', 'loudness', 'energy')]
y = df['artist_top_genre']
X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
y = le.transform(y)
```
- Chọn cột có **khoảng giá trị tương tự nhau**; encode cột thể loại thành số.

### Chạy K-Means với k = 3 (3 thể loại nhạc)
```python
from sklearn.cluster import KMeans
nclusters = 3
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X)
y_cluster_kmeans = km.predict(X)   # mảng nhãn cụm 0/1/2 cho từng dòng
```

### Silhouette score — đo chất lượng cụm
```python
from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)   # 0,53
```
- Thang **−1 đến 1**: gần 1 = cụm **dày đặc và tách biệt tốt**; gần 0 = cụm **chồng lấn**, mẫu nằm sát biên quyết định của cụm bên cạnh. **0,53 — ở giữa**: dữ liệu này không hợp lắm với kiểu clustering này, nhưng cứ tiếp tục.

### Elbow method — chọn k bằng WCSS/Inertia
```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
plt.title('Elbow'); plt.xlabel('Number of clusters'); plt.ylabel('WCSS')
plt.show()
```
- **WCSS** (within-cluster sums of squares): trung bình bình phương khoảng cách từ mọi điểm trong cụm tới centroid của cụm.
- **Inertia**: thước đo "cụm gắn kết nội bộ tới đâu" — K-Means cố chọn centroid để **tối thiểu hóa inertia**. Mỗi vòng lặp append `kmeans.inertia_` vào `wcss`.
- **`k-means++`**: khởi tạo centroid **xa nhau** ngay từ đầu → kết quả thường tốt hơn khởi tạo ngẫu nhiên.
- Vẽ WCSS theo k = 1..10, tìm **"khuỷu tay" (bend)** — điểm cong của đường = số cụm tối ưu. Với dataset này, khuỷu tay quanh **k = 3** (khớp với 3 thể loại đã lọc).

### Hiển thị cụm + đo "accuracy"
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)
plt.scatter(df['popularity'], df['danceability'], c=labels)
plt.xlabel('popularity'); plt.ylabel('danceability')

labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'.format(correct_labels / float(y.size)))
```
- **Accuracy không tốt** — và hình dạng cụm cho biết vì sao: dữ liệu **quá mất cân bằng, tương quan quá yếu, phương sai (variance) giữa các cột quá lớn**. Cụm tạo ra có lẽ bị lệch nặng theo 3 thể loại đã định sẵn. Đó cũng là một bài học!
- **Variance** = trung bình bình phương độ lệch so với mean. Ở đây: các con số trong dataset lệch khỏi mean quá nhiều → cụm không tách rõ.

### Cách cải thiện (gợi ý của bài)
- Dọn dữ liệu kỹ hơn (loại outlier), thử cột khác, thử thuật toán khác.
- **Scale dữ liệu (chuẩn hóa)**: đưa các cột về cùng thang đo. Trong notebook có code chú thích dùng StandardScaler — khi scale, **silhouette score giảm** nhưng "khuỷu" của đồ thị elbow **mượt hơn**. Lý do: để dữ liệu không scale thì cột ít phương sai sẽ mang trọng số lớn hơn một cách không công bằng. Đọc thêm: https://stats.stackexchange.com/questions/21222/
- Thử thách: chỉnh tham số, dùng trọng số (weights) cho mẫu, xem có cải thiện được accuracy không. Thử K-Means Simulator + handout K-Means của Stanford.

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Clustering đúng bài toán "khám phá" của kênh:** khi có đống dữ liệu chưa nhãn (bình luận, chủ đề, video), clustering giúp **tìm nhóm tự nhiên** trước khi đặt tên cho chúng — ví dụ: gom bình luận thành nhóm (khen / hỏi / chê / spam) mà không cần gán nhãn trước; gom video thành nhóm theo pattern lượt xem để xem nhóm nào hiệu quả.
2. **Trực quan hóa trước, thuật toán sau:** bài 14 dành cả buổi chỉ để vẽ bar chart, heatmap, jointplot, scatter — vì **chọn sai thuật toán cho bản chất dữ liệu thì kết quả vô nghĩa**. Với dữ liệu kênh cũng vậy: vẽ biểu đồ lượt xem theo slot/chủ đề trước, rồi mới quyết định dùng hồi quy, phân loại hay clustering.
3. **Silhouette score = cách đo "nhóm có tách rõ không":** áp dụng tư duy này khi chia nhóm video/chủ đề — nếu các nhóm chồng lấn nhau (score gần 0) thì cách chia nhóm đó chưa có ý nghĩa, cần đổi đặc trưng hoặc đổi cách chia.
4. **Elbow method = cách chọn "bao nhiêu nhóm là đủ":** khi chia video thành các nhóm chủ đề hay chia khán giả thành các phân khúc, vẽ WCSS theo số nhóm và tìm điểm khuỷu — tránh chia quá nhiều nhóm vô nghĩa hoặc quá ít nhóm gộp chung.
5. **Scale dữ liệu trước khi clustering:** các cột có thang đo khác nhau (lượt xem hàng nghìn, tỉ lệ % nhỏ) sẽ làm lệch cụm — phải chuẩn hóa về cùng thang đo trước, giống bài học StandardScaler ở bài 15.
6. **Cluster Id để ẩn danh:** gán mỗi video/bình luận một cluster Id thay vì dữ liệu nhạy cảm — hữu ích khi chia sẻ dữ liệu phân tích với các phòng mà không lộ thông tin riêng.

## Tiến độ khóa G11
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ Phần 2 (bài 05–09): Regression + Web App — xong 20/11.
- ✅ Phần 3 (bài 10–13): Classification — xong 20/11.
- ✅ **Phần 4 (bài 14–15): Clustering — xong 20/11 (ghi chú này).**
- ⏳ Phần 5: NLP (bài 16–20).
- ⏳ Phần 6: Time Series (bài 21–23).
- ⏳ Phần 7: Reinforcement Learning (bài 24–25) + Postscript.
- ⏳ Sau G11: S1 "Understanding Deep Learning" (sách PDF miễn phí), rồi các khóa đã đăng ký khác (freeCodeCamp Python v9, Udemy 73 khóa, GLM-5.3-Flash build + train) — **quy tắc mới của Hoai (20/11 ~23:19): khóa nào đến bước cần anh làm (thi chứng chỉ, nhập token,…) thì BỎ QUA, học tiếp khóa khác.**

## Nguồn
- Bài 14: https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/README.md
- Bài 15: https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
