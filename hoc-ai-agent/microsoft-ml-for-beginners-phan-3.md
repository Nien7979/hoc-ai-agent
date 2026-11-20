# G11 — Microsoft ML-For-Beginners: Phần 3 — Classification (bài 10–13)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Phần 2 (bài 05–09): `microsoft-ml-for-beginners-phan-2.md` — Regression + Web App (Flask + Pickle).
> Ghi chú này: bài 10, 12, 13 đọc trên GitHub (20/11/2026); bài 11 đọc từ clone local `~/workspace/knowledge/ml-for-beginners/` (clone đã tải xong: 1,9 GB, HEAD 8e5c0dff) vì bản web của bài 11 trả về HTML thô. Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong repo khóa học.

---

## Bài 10 — Giới thiệu về Classification

### Phân loại là gì
- **Classification** là một dạng **supervised learning** (học có giám sát), giống hồi quy nhưng đầu ra là **nhãn/loại** thay vì số.
  - **Binary classification** (2 loại): email này là spam hay không?
  - **Multiclass classification** (nhiều loại): món ăn này thuộc ẩm thực nước nào?
- Cách nói khoa học: phương pháp phân loại tạo ra **mô hình dự đoán ánh xạ biến đầu vào → biến đầu ra**.
- Ví dụ gốc từ thống kê: dùng đặc trưng `smoker, weight, age` để xác định **khả năng mắc bệnh X**.
- So với 2 bài hồi quy đã học:
  - Hồi quy tuyến tính: dự đoán **giá trị** (giá bí ngô tháng 9 so với tháng 12).
  - Hồi quy logistic: phát hiện **loại nhị phân** (bí này cam hay không cam).
  - Classification: nhiều thuật toán khác nhau để xác định **nhãn/loại** của một điểm dữ liệu.
- Câu hỏi của 4 bài này: cho một nhóm nguyên liệu, **đoán món ăn thuộc ẩm thực quốc gia nào** (Ấn Độ, Trung Quốc, Nhật, Hàn, Thái)? Lấy cảm hứng từ show "Chopped" — rổ nguyên liệu bí ẩn, đầu bếp phải nấu ra món gì đó; model ML hẳn đã giúp được!

### Dataset: ẩm thực châu Á (`cuisines.csv`)
- **2.448 món, 385 cột**: 1 cột nhãn `cuisine` + **384 cột nguyên liệu nhị phân** (0/1 — món có dùng nguyên liệu đó không).
- Phân bố **không đều**:
  - korean 799, indian 598, chinese 442, japanese 320, thai 289.
- Khám phá nguyên liệu đặc trưng từng ẩm thực:
```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine', 'Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
    return ingredient_df

thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()   # top 10 nguyên liệu phổ biến nhất của ẩm thực Thái
```
- **Loại bỏ nguyên liệu gây nhiễu:** cơm (rice), tỏi (garlic), gừng (ginger) — món nào cũng dùng, không phân biệt được ẩm thực nào với ẩm thực nào:
```python
feature_df = df.drop(['cuisine', 'Unnamed: 0', 'rice', 'garlic', 'ginger'], axis=1)
labels_df = df.cuisine
```

### Cân bằng dữ liệu bằng SMOTE
- **Vấn đề:** dữ liệu lệch (korean 799 vs thai 289). Nếu đa số dữ liệu là một lớp, model sẽ **dự đoán lớp đó thường xuyên hơn chỉ vì nó nhiều dữ liệu hơn** — không phải vì nó đúng.
- **SMOTE** (Synthetic Minority Over-sampling Technique, gói `imblearn` — `pip install imblearn`): tạo **mẫu mới bằng nội suy** từ các mẫu của lớp thiểu số:
```python
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
# Kết quả: cả 5 ẩm thực đều 799 mẫu
transformed_df = pd.concat([transformed_label_df, transformed_feature_df], axis=1, join='outer')
transformed_df.to_csv("../data/cleaned_cuisines.csv")   # lưu dùng cho các bài sau
```

---

## Bài 11 — Các bộ phân loại ẩm thực 1 (Logistic Regression đa lớp)

### Chuẩn bị dữ liệu
```python
import pandas as pd
cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
cuisines_label_df = cuisines_df['cuisine']
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
```

### Chọn classifier nào?
- Scikit-learn có rất nhiều cách phân loại (Supervised Learning): Linear Models, Support Vector Machines, Stochastic Gradient Descent, Nearest Neighbors, Gaussian Processes, Decision Trees, Ensemble methods, Multiclass/multioutput…
- **Đừng đoán mò:** dùng **Microsoft ML Cheat Sheet** (bảng tra thuật toán) + biểu đồ so sánh các classifier của scikit-learn (KNeighbors, SVC, GaussianProcess, DecisionTree, RandomForest, MLP, AdaBoost, GaussianNB, QDA trên cùng một dataset tổng hợp). **AutoML** còn giải quyết gọn hơn: chạy so sánh trên cloud rồi chọn thuật toán tốt nhất.
- Suy luận loại trừ cho bài toán này:
  - **Neural network quá nặng** — dataset nhỏ, train local bằng notebook.
  - **Không dùng two-class classifier** (one-vs-all) — đây là bài toán đa lớp.
  - **Decision tree hoặc logistic regression** đều có thể làm được.
  - **Multiclass Boosted Decision Trees** hợp với bài toán xếp hạng (nonparametric) — không hợp ở đây.

### Logistic Regression cho đa lớp: `multi_class` + `solver`
- Logistic regression vốn thiết kế cho **nhị phân**; để xử lý đa lớp cần một "scheme":
  - **`ovr` (one-vs-rest):** train một classifier nhị phân cho mỗi lớp ("lớp này vs tất cả các lớp còn lại").
  - **`multinomial`:** dùng **cross-entropy loss** trực tiếp cho đa lớp — chỉ hỗ trợ bởi các solver `lbfgs, sag, saga, newton-cg`.
- **`solver`** = thuật toán tối ưu hóa dùng trong bài toán tối ưu. Không phải solver nào cũng đi được với mọi `multi_class`.

### Train + kiểm tra
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)

lr = LogisticRegression(multi_class='ovr', solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))
accuracy = model.score(X_test, y_test)   # > 80%
```
- Thử một dòng dữ liệu (#50): nguyên liệu `cilantro, onion, pea, potato, tomato, vegetable_oil` → `cuisine: indian`. Xác suất dự đoán:
  - indian **0,716**, chinese 0,229, japanese 0,030, korean 0,017, thai 0,008.
- Classification report (accuracy **0,80**):

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| chinese | 0,73 | 0,71 | 0,72 | 229 |
| indian | 0,91 | 0,93 | 0,92 | 254 |
| japanese | 0,70 | 0,75 | 0,72 | 220 |
| korean | 0,86 | 0,76 | 0,81 | 242 |
| thai | 0,79 | 0,85 | 0,82 | 254 |

- **Indian dễ đoán nhất** (F1 0,92), **Japanese khó nhất** (F1 0,72). Thử thách: đọc sâu về `solver` — chuyện gì xảy ra phía sau hậu trường.

---

## Bài 12 — Các bộ phân loại ẩm thực 2 (SVC, KNN, Ensemble)

### Bản đồ chọn model của scikit-learn
- Đi theo "đường mòn" của cheat sheet khi đã hiểu rõ dữ liệu:
  - Có > 50 mẫu → muốn dự đoán **một loại** → dữ liệu **đã gán nhãn** → ít hơn 100K mẫu → ✨ **Linear SVC**.
  - Không được thì thử ✨ **KNeighbors Classifier** (dữ liệu số) → rồi ✨ **SVC** và ✨ **Ensemble Classifiers**.
- Chia dữ liệu: `train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)`.

### 1. Linear SVC — 78,6%
```python
from sklearn.svm import SVC
classifiers = {'Linear SVC': SVC(kernel='linear', C=10, probability=True, random_state=0)}
```
- **SVC** thuộc họ **Support-Vector Machines**: "ánh xạ các ví dụ huấn luyện thành các điểm trong không gian" để **tối đa hóa khoảng cách giữa hai loại**; dữ liệu mới được ánh xạ vào không gian đó để dự đoán loại.
- **`kernel`**: chọn cách gom cụm nhãn (ở đây `'linear'`). **`C`**: tham số **regularization** — điều chỉnh ảnh hưởng của các tham số. `probability=True` để lấy ước lượng xác suất.

### 2. K-Neighbors Classifier (KNN) — 73,8%
```python
from sklearn.neighbors import KNeighborsClassifier
'KNN classifier': KNeighborsClassifier(C),
```
- Họ "neighbors": tạo trước một số điểm, gom dữ liệu quanh các điểm đó để dự đoán nhãn tổng quát.
- **Tệ hơn SVC.** Điểm đáng chú ý: korean **precision 0,94 nhưng recall chỉ 0,58** — khi model bảo "đây là món Hàn" thì gần như luôn đúng, nhưng nó bỏ sót rất nhiều món Hàn thật.

### 3. Support Vector Classifier (SVC, kernel mặc định RBF) — 83,2%
```python
'SVC': SVC(),
```
- Đổi kernel mặc định (RBF, phi tuyến) thay vì linear → accuracy nhảy lên **83,2%**, tốt nhất tới lúc này. Mọi lớp đều F1 ≥ 0,76.

### 4. Ensemble Classifiers — Random Forest thắng
```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
'RFST': RandomForestClassifier(n_estimators=100),
'ADA': AdaBoostClassifier(n_estimators=100)
```
- **Ensemble** = kết hợp dự đoán của **nhiều estimator cơ sở** để cải thiện chất lượng model.
- **Random Forest** (phương pháp averaging): xây "rừng" cây quyết định, thêm **tính ngẫu nhiên để tránh overfitting**; `n_estimators` = số cây. → **84,5% — tốt nhất bài này.**
- **AdaBoost**: fit một classifier, rồi fit các bản sao tiếp theo **tập trung vào trọng số của những mẫu bị phân loại sai** để sửa lỗi. → chỉ **72,4% — tệ nhất** ở đây.

| Classifier | Accuracy |
|---|---|
| Logistic Regression (bài 11) | ~80% |
| Linear SVC | 78,6% |
| KNN | 73,8% |
| SVC (RBF) | 83,2% |
| **Random Forest** | **84,5%** |
| AdaBoost | 72,4% |

- **Bài học: cùng một dataset, thuật toán khác nhau cho kết quả khác nhau rất nhiều — phải thử nhiều classifier và so sánh trên cùng một thước đo.** Thử thách: nghiên cứu tham số mặc định của từng thuật toán, chỉnh tham số thì chất lượng model thay đổi thế nào.

---

## Bài 13 — Ứng dụng: Web app gợi ý ẩm thực (Recommender)

### Từ phân loại tới hệ thống gợi ý
- Một trong những ứng dụng thực tế hữu ích nhất của ML là **hệ thống gợi ý (recommendation systems)** — bài này là bước đầu tiên theo hướng đó: nhìn vào tủ lạnh còn nguyên liệu gì → gợi ý nấu được món thuộc ẩm thực nào.
- Khác bài 09 (Pickle + Flask = app Python full-stack): bài này xây hệ thống **chạy bằng JavaScript trong trình duyệt**, dùng **ONNX Runtime** — model chạy local, thậm chí **offline** cũng được.

### Train model + chuyển sang ONNX
```python
from sklearn.svm import SVC
model = SVC(kernel='linear', C=10, probability=True, random_state=0)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))   # accuracy 0,79
```
- Chuyển scikit-learn → ONNX bằng `skl2onnx` (`pip install skl2onnx`). Dataset có **380 nguyên liệu** → phải khai đúng số này trong `FloatTensorType`:
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}
onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```
- `nocl=True`: bỏ thông tin lớp khỏi model → **giảm kích thước file**. `zipmap=False`: bỏ ZipMap (đầu ra dạng danh sách dict — không cần cho bài này).
- **Xem model bằng Netron** (phần mềm miễn phí): mở `model.onnx` sẽ thấy 380 đầu vào + classifier — kiểm tra model build đúng chưa trước khi đưa vào app.

### Web app: checkbox nguyên liệu → chạy model trong trình duyệt
- `index.html` cùng thư mục với `model.onnx`. Mỗi checkbox nguyên liệu có `value` = **chỉ số cột của nguyên liệu đó trong dataset** (đếm từ 0; apple = 4). Tra bảng `ingredient_indexes`.
- Nạp ONNX Runtime từ CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script>
```
- Logic JS:
```javascript
const ingredients = Array(380).fill(0);
const checks = [...document.querySelectorAll('.checkbox')];
checks.forEach(check => {
    check.addEventListener('change', function() {
        ingredients[check.value] = check.checked ? 1 : 0;   // bật/tắt nguyên liệu
    });
});

async function startInference() {
    if (!checks.some(check => check.checked)) {
        alert('Please select at least one ingredient.');
        return;
    }
    const session = await ort.InferenceSession.create('./model.onnx');
    const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
    const feeds = { float_input: input };   // tên input phải khớp lúc convert (kiểm tra bằng Netron)
    const results = await session.run(feeds);
    alert('You can enjoy ' + results.label.data[0] + ' cuisine today!');
}
```
- Chạy thử: `http-server` (npm, cài global) trong thư mục chứa `index.html` → mở localhost, tick nguyên liệu, bấm nút là ra ẩm thực gợi ý.
- Thử thách: mở rộng app với đầy đủ nguyên liệu + chỉ số của chúng; đọc thêm về recommendation engines (ScienceDirect, MIT Technology Review).

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Classification đúng bài toán của kênh:** phân loại bình luận (tích cực / tiêu cực / hỏi đáp), phân loại chủ đề video, phân loại video theo "nhóm hiệu quả" — đều là **multiclass** giống bài toán ẩm thực. Dữ liệu đầu vào cũng là dạng 0/1 (có/không có yếu tố X) như bảng nguyên liệu.
2. **Dữ liệu lệch là bẫy thật:** video viral ít, video trung bình nhiều — train model trên dữ liệu thô thì model sẽ chỉ đoán "trung bình" cho mọi video. Phải **cân bằng lớp (SMOTE hoặc class_weight)** trước khi tin kết quả, giống bài 10.
3. **Đừng cưới một thuật toán:** cùng dataset, Random Forest 84,5% còn AdaBoost 72,4%. Quy trình đúng: thử nhiều classifier (Logistic, SVC, KNN, Random Forest) → so sánh **cùng một bộ chỉ số** (accuracy + precision/recall từng lớp) → chọn cái tốt nhất cho dữ liệu của mình.
4. **Nhìn điểm từng lớp, đừng chỉ nhìn accuracy:** Japanese F1 0,72 trong khi Indian 0,92 — accuracy tổng 80% che giấu việc model yếu ở một số lớp. Với video: có chủ đề model sẽ đoán sai hệ thống, phải phát hiện bằng classification report từng lớp.
5. **ONNX chạy trong trình duyệt = ship ML không cần server:** model 380 đầu vào chạy ngay trên trình duyệt, offline được, không tốn tiền server. Mẫu này dùng được cho tool phía client sau này (ví dụ: trang phân loại bình luận/chủ đề chạy ngay trên máy, không cần backend).
6. **Loại bỏ đặc trưng gây nhiễu:** cơm/tỏi/gừng có ở mọi món nên không giúp phân biệt — tương tự, có những yếu tố video nào cũng có (ví dụ: đăng đúng giờ) thì không giúp phân biệt video hay/dở; phải loại chúng ra khỏi đặc trưng trước khi train.

## Tiến độ khóa G11
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ Phần 2 (bài 05–09): Regression + Web App — xong 20/11.
- ✅ **Phần 3 (bài 10–13): Classification — xong 20/11 (ghi chú này).**
- ⏳ Phần 4: Clustering (bài 14–15).
- ⏳ Phần 5: NLP (bài 16–20).
- ⏳ Phần 6: Time Series (bài 21–23).
- ⏳ Phần 7: Reinforcement Learning (bài 24–25) + Postscript.
- ⏳ Sau G11: S1 "Understanding Deep Learning" (sách PDF miễn phí).

## Nguồn
- Bài 10: https://github.com/microsoft/ML-For-Beginners/blob/main/4-Classification/1-Introduction/README.md
- Bài 11: https://github.com/microsoft/ML-For-Beginners/blob/main/4-Classification/2-Classifiers-1/README.md (đọc từ clone local do bản web trả HTML thô)
- Bài 12: https://github.com/microsoft/ML-For-Beginners/blob/main/4-Classification/3-Classifiers-2/README.md
- Bài 13: https://github.com/microsoft/ML-For-Beginners/blob/main/4-Classification/4-Applied/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
