# G11 — Microsoft ML-For-Beginners: Phần 2 — Regression (bài 05–08) + Web App (bài 09)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Ghi chú này: đọc trực tiếp 5 file README trên GitHub (20/11/2026). Clone local `~/workspace/knowledge/ml-for-beginners/` vẫn đang tải dở (mới có `.git`, ~200 MB) — chưa dùng được.

---

## Bài 05 — Bắt đầu với Python và Scikit-learn cho mô hình hồi quy

### Môi trường làm việc của data scientist
- Cài **Python** (nên dùng **virtual environment** — mỗi dự án một phiên bản thư viện khác nhau).
- Cài **Visual Studio Code** + extension Python.
- Cài **Scikit-learn** (`pip install scikit-learn`, Python 3; Mac M1 có hướng dẫn riêng).
- Cài **Jupyter Notebook** — file `.ipynb`: code xen kẽ ghi chú markdown, chạy từng ô (cell) bằng nút play. Đây là môi trường thử nghiệm của data scientist, khác hẳn môi trường của web developer.
- Bài tập: mở `notebook.ipynb`, thêm markdown `# Welcome to your notebook`, chạy `print('hello notebook')`.

### Scikit-learn là gì
- Thư viện ML mã nguồn mở: hỗ trợ **supervised + unsupervised learning**, model fitting, tiền xử lý dữ liệu, chọn mô hình, đánh giá.
- Tập trung vào **dữ liệu số**, có sẵn nhiều dataset mẫu + model dựng sẵn để học.
- Khóa này cố tình tránh neural network / deep learning — đó là phần của khóa "AI for Beginners" sau này.

### Mô hình đầu tiên: dataset tiểu đường (diabetes)
- Dataset có sẵn trong scikit-learn: **442 mẫu, 10 đặc trưng** (age, bmi, bp, s1 tc — T-Cells, …). Mục tiêu `y`: mức độ tiến triển bệnh.
- Chọn bài toán nào thì chọn loại hồi quy đó:
  - Muốn dự đoán **giá trị số** (chiều cao theo tuổi) → **linear regression**.
  - Muốn gán **nhãn loại** (món ăn có phải vegan không) → **logistic regression** (học ở bài 08).
- Code mẫu:
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

X, y = datasets.load_diabetes(return_X_y=True)   # X: (442, 10)
X = X[:, 2].reshape((-1, 1))                      # chỉ lấy cột BMI, reshape thành N×1
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Scaled BMIs'); plt.ylabel('Disease Progression')
plt.show()
```
- Ý nghĩa: đường thẳng chạy qua đám điểm — dùng nó để **dự đoán điểm mới** sẽ rơi ở đâu trên trục y.
- Lưu ý fairness: dataset có đặc trưng `sex` (nhị phân) — nhiều dataset y tế phân loại nhị phân như vậy có thể **loại trừ một phần dân số** khỏi điều trị. Nghĩ về điều này khi chọn đặc trưng.

---

## Bài 06 — Chuẩn bị và trực quan hóa dữ liệu (chợ bí ngô)

### Hỏi đúng câu hỏi trước khi chạm vào dữ liệu
- **Câu hỏi quyết định thuật toán; chất lượng dữ liệu quyết định chất lượng câu trả lời.**
- Câu hỏi của bài này: *"Dự đoán giá một quả bí ngô bán trong một tháng cho trước"* → dùng Regression.
- Dataset: `US-pumpkins.csv` — **1.757 dòng** giá bí ngô theo thành phố, dữ liệu thô từ **USDA** (Specialty Crops Terminal Markets Standard Reports, public domain). Rất lộn xộn: ô trống, chuỗi + số lẫn lộn, cột `Package` đủ kiểu ("sacks", "bins", "1 1/9 bushel"…).
- Hiếm khi được tặng dataset sạch sẵn — **chuẩn bị dữ liệu là việc thường xuyên nhất** của data scientist.

### Dọn dữ liệu bằng Pandas
```python
import pandas as pd
pumpkins = pd.read_csv('../data/US-pumpkins.csv')
pumpkins.head()              # xem 5 dòng đầu
pumpkins.isnull().sum()      # đếm ô trống từng cột
columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.loc[:, columns_to_select]

price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2   # giá trung bình
month = pd.DatetimeIndex(pumpkins['Date']).month               # Date MM/DD/YYYY → tháng
new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'],
                             'Low Price': pumpkins['Low Price'],
                             'High Price': pumpkins['High Price'], 'Price': price})
```
- **Lọc theo đơn vị bán:** chỉ giữ bí bán theo `bushel` (còn ~415 dòng) — vì bí rất khó cân nhất quán, bán theo "each"/"per bin"/"per pound" không so sánh được.
- **Chuẩn hóa giá theo bushel:** bushel là đơn vị **thể tích**, trọng lượng thay đổi theo loại nông sản (1 bushel cà chua ~56 lbs, rau spinach chỉ ~20 lbs). Nên chia giá cho đúng lượng:
```python
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price / (1 + 1/9)
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price / (1/2)
```
- Chi tiết thú vị: bí bán nửa bushel **đắt hơn hẳn** — vì bí nhỏ, một bushel chứa được nhiều quả hơn (bí to rỗng ruột, tốn chỗ).
- **Bài học lớn: phải hiểu bản chất dữ liệu** trước khi mô hình hóa.

### Trực quan hóa: Matplotlib vs Seaborn
- **Matplotlib** (nền tảng, kiểm soát từng chi tiết):
  - Scatter `price` vs `month`: chỉ thấy đám điểm loãng — **không hữu ích**.
  - Bar chart sau khi groupby tháng: `new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')` → giá **cao nhất tháng 9–10** (mùa Halloween). Nhóm dữ liệu lại mới thấy được thông tin.
- **Seaborn** (xây trên Matplotlib, chuyên trực quan thống kê, làm việc trực tiếp với dataframe, ít code hơn, mặc định đẹp):
  - `sns.relplot(x="Price", y="Month", data=new_pumpkins)` — scatter; `kind="line"` vẽ thêm dải **confidence interval** (khoảng tin cậy).
  - `sns.catplot(x="Month", y="Price", data=new_pumpkins, kind="bar")` — tự groupby + tính trung bình + vẽ vạch CI. Xác nhận đỉnh giá tháng 9–10, kèm độ biến thiên trong tháng.
  - **Heatmap tương quan:** `correlations = new_pumpkins[['Month','Low Price','High Price','Price']].corr()` rồi `sns.heatmap(correlations, annot=True, cmap="coolwarm")` — xem sức mạnh quan hệ của **mọi cặp cột cùng lúc**.
    - `Low Price` và `High Price` tương quan gần như hoàn hảo (đương nhiên).
    - `Month` tương quan tuyến tính với giá **rất yếu** — dù bar chart cho thấy đỉnh mùa rõ ràng. **Hệ số tương quan chỉ đo quan hệ đường thẳng, nó bỏ lỡ pattern theo mùa / phi tuyến.** Phải xem cả heatmap lẫn biểu đồ trước khi chọn cột đưa vào model.
- Workflow thực tế: **Seaborn để khám phá nhanh → Matplotlib khi cần tùy chỉnh chi tiết.**

---

## Bài 07 — Hồi quy tuyến tính và đa thức (4 mô hình trong 1 bài)

### Toán nền: bình phương tối thiểu (Least-Squares)
- Mục tiêu: vẽ đường thẳng thể hiện quan hệ giữa các biến + **dự đoán điểm mới**.
- Với mỗi điểm dữ liệu, đo khoảng cách dọc (**residual**) từ điểm thực tới đường hồi quy, rồi **bình phương** lên vì 2 lý do:
  1. **Bỏ dấu:** lỗi −5 và +5 coi như nhau.
  2. **Phạt lỗi lớn:** bình phương làm lỗi lớn nặng hơn, ép đường thẳng bám sát các điểm xa.
- Cộng tất cả lại, tìm đường làm tổng này **nhỏ nhất** → "Least-Squares".
- **Đường phù hợp nhất:** `Y = a + bX` — `X` biến giải thích, `Y` biến phụ thuộc, `b` độ dốc, `a` giao điểm với trục Y (giá trị Y khi X = 0).
- **Hệ số tương quan:** điểm càng xếp thành đường thẳng gọn → tương quan càng cao (gần 1); điểm tung tóe → tương quan thấp (gần 0). Model hồi quy tuyến tính tốt là model có hệ số tương quan cao.

### Khám phá dữ liệu trước khi train
- Thêm cột `DayOfYear` (số ngày từ đầu năm): `pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt - datetime(dt.year,1,1)).days)`.
- Tương quan `Month`–`Price`: **−0,15**; `DayOfYear`–`Price`: **−0,17** — rất yếu. Nhưng scatter plot tô màu theo `Variety` cho thấy **giống bí ảnh hưởng giá nhiều hơn ngày bán** (các cụm giá tách theo giống).
- Chỉ lấy giống **PIE TYPE**: tương quan `DayOfYear`–`Price` lên **−0,27** — đáng train model.
- Dọn sạch trước khi train: hồi quy tuyến tính **không chịu được giá trị thiếu** → `pie_pumpkins.dropna(inplace=True)` (hoặc điền bằng giá trị trung bình của cột).

### Mô hình 1 — Hồi quy tuyến tính đơn giản
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1, 1)  # LinearRegression cần mảng 2D, mỗi hàng = 1 vector đặc trưng
y = pie_pumpkins['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred = lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))   # ~2 điểm, ~17% — chưa tốt
score = lin_reg.score(X_train, y_train)            # R² ≈ 0,06 — rất thấp
```
- `lin_reg.coef_` ≈ **−0,017**: giá giảm ~2 xu/ngày theo thời gian.
- `lin_reg.intercept_` ≈ **21**: giá lúc đầu năm.
- **RMSE** (root mean square error): căn bậc hai trung bình bình phương sai số — đo độ lệch trung bình giữa dự đoán và thực tế.
- **R² (coefficient of determination, `score`)**: 0 = model không dùng input gì (chỉ đoán bằng giá trị trung bình — "worst linear predictor"); 1 = dự đoán hoàn hảo.

### Mô hình 2 — Hồi quy đa thức (Polynomial Regression)
- Quan hệ Date–Price không phải đường thẳng — giá biến động theo mùa → thử **đường cong**.
- Thêm đặc trưng `DayOfYear²` → đường parabol, có điểm cực tiểu trong năm. Dùng **pipeline** (chuỗi các estimator) của scikit-learn:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
```
- `PolynomialFeatures(2)`: thêm mọi đa thức bậc 2 (với 2 biến X, Y sẽ thêm X², XY, Y²). Có thể dùng bậc cao hơn.
- Vẽ đường cong mượt bằng `np.linspace` (tạo 100 giá trị đều) thay vì plot trực tiếp trên test data (sẽ thành đường zigzag).
- Kết quả: RMSE thấp hơn chút, R² cao hơn chút — **không đáng kể**. Giá bí thấp nhất quanh **Halloween** (cung nhiều). Cần thêm đặc trưng khác!

### Đặc trưng phân loại (categorical features) — phần quan trọng nhất bài này
- Muốn **một model dự đoán giá cho mọi giống bí**, phải đưa cột `Variety` (chữ, không phải số) vào model → cần **mã hóa (encode)**.
- **Mã hóa số đơn giản** (thay tên giống bằng chỉ số 0,1,2,3): TỆ cho hồi quy tuyến tính — vì model nhân chỉ số với hệ số, coi như quan hệ giữa chỉ số và giá là tuyến tính, trong khi thực tế không phải.
- **One-hot encoding** (ĐÚNG): thay cột `Variety` bằng 4 cột (FAIRYTALE, MINIATURE, MIXED HEIRLOOM, PIE TYPE), mỗi cột 1 nếu đúng giống đó, 0 nếu không. Mỗi giống có **một hệ số riêng** = "giá khởi điểm" (giá cộng thêm) của giống đó.
```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
# ... train như cũ → R² ~0,77 (dù RMSE tương tự) — nhảy vọt!
```
- Thêm nhiều đặc trưng phân loại + số cùng lúc bằng `join`:
```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
# → RMSE 2,84 (10,5%), R² 0,94
```

### Gộp tất cả: đa thức + mọi đặc trưng = model tốt nhất
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, pred, squared=False)   # 2,23 (~8%)
score = pipeline.score(X_train, y_train)                  # R² ~0,97
```

| Mô hình | RMSE | R² |
|---|---|---|
| `DayOfYear` tuyến tính | 2,77 (17,2%) | 0,07 |
| `DayOfYear` đa thức | 2,73 (17,0%) | 0,08 |
| `Variety` tuyến tính | 5,24 (19,7%) | 0,77 |
| Mọi đặc trưng, tuyến tính | 2,84 (10,5%) | 0,94 |
| **Mọi đặc trưng, đa thức** | **2,23 (8,25%)** | **0,97** |

- **Bài học: đặc trưng tốt quan trọng hơn thuật toán phức tạp.** Chỉ đổi từ "ngày" sang "giống + thành phố + đóng gói" đã đưa R² từ 0,07 lên 0,97.
- Đọc thêm: Stepwise, **Ridge, Lasso, ElasticNet**; khóa Stanford Statistical Learning.

---

## Bài 08 — Hồi quy logistic (dự đoán loại — thực chất là phân loại)

### Vì sao gọi là "hồi quy" mà lại phân loại?
- Logistic regression dùng để **dự đoán loại nhị phân**: kẹo này là chocolate hay không? Bệnh này có lây không? Khách có mua sản phẩm không?
- Bài này: dự đoán **màu bí** — trắng ("ghost" 👻) hay không trắng (cam 🎃). Gọi là hồi quy chỉ vì **thuận tiện ngôn ngữ** — thực chất nó là phương pháp phân loại tuyến tính (sẽ học kỹ ở nhóm bài Classification sau).
- Các dạng khác: **Multinomial** (nhiều loại: cam/trắng/sọc), **Ordinal** (loại có thứ tự: mini, sm, med, lg, xl, xxl).

### Khác biệt với hồi quy tuyến tính
- Tuyến tính: dự đoán **giá trị liên tục** (giá sẽ tăng bao nhiêu). Logistic: dự đoán **loại nhị phân**.
- Tuyến tính cần biến **tương quan cao**; logistic thì **biến KHÔNG cần tương quan** — hợp với dataset bí ngô vốn tương quan yếu.
- Logistic cần **nhiều dữ liệu sạch**; dataset nhỏ (~1000 dòng) không tối ưu, nhưng vẫn làm được.

### Dọn dữ liệu + mã hóa (encoding) — kỹ năng cốt lõi
- Chọn cột: `City Name, Package, Variety, Origin, Item Size, Color`; `dropna()`. Loại 'striped' (quá ít mẫu, tự biến mất sau khi drop null).
- **OrdinalEncoder** — cho biến **có thứ tự** (`Item Size`): mỗi loại = một số theo đúng thứ tự trong cột:
```python
from sklearn.preprocessing import OrdinalEncoder
item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
```
- **OneHotEncoder** — cho biến **không thứ tự** (`City Name, Package, Variety, Origin`): mỗi loại = một cột nhị phân 0/1.
- **ColumnTransformer** — gộp nhiều encoder thành một bước, áp cho đúng cột:
```python
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[
    ('ord', ordinal_encoder, ['Item Size']),
    ('cat', OneHotEncoder(sparse_output=False), ['City Name', 'Package', 'Variety', 'Origin'])
])
ct.set_output(transform='pandas')
encoded_features = ct.fit_transform(pumpkins)
```
- **LabelEncoder** — mã hóa **nhãn** `Color` thành 0/1.

### Trực quan hóa bằng Seaborn
- `sns.catplot(data=pumpkins, y="Variety", hue="Color", kind="count")` — so sánh phân bố theo giống và màu.
- Box plot `Item Size` × `Color` theo từng `Variety` (dùng cột Item Size đã mã hóa).
- **Swarm plot**: `sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins)` — xem phân bố từng điểm; nhiều điểm quá sẽ cảnh báo (giảm `size` của marker để đọc được).

### Toán nền: hàm sigmoid + maximum likelihood
- Hàm **sigmoid** (đường cong logistic, hình chữ S): nhận một giá trị bất kỳ, ánh xạ về khoảng **(0, 1)**. Điểm giữa của đường cong nằm tại x = 0; L = giá trị tối đa; k = độ dốc.
- Kết quả > **0,5** → gán lớp **1**; ≤ 0,5 → lớp **0**.

### Train model + đọc bảng điểm
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print('F1-score: ', f1_score(y_test, predictions))   # 0,746
```
```
              precision    recall  f1-score   support
           0       0.94      0.98      0.96       166
           1       0.85      0.67      0.75        33
    accuracy                           0.92       199
   macro avg       0.89      0.82      0.85       199
weighted avg       0.92      0.92      0.92       199
```
- Lớp 1 (bí trắng) khó hơn: chỉ 33 mẫu, recall 0,67 — model bỏ sót nhiều bí trắng.

### Confusion matrix — đọc model bằng bảng 2×2
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
# array([[162,   4],
#        [ 11,  22]])
```
- Hàng = nhãn thực, cột = nhãn dự đoán:

|  | Dự đoán 0 | Dự đoán 1 |
|---|---|---|
| **Thực 0** | TN = 162 | FP = 4 |
| **Thực 1** | FN = 11 | TP = 22 |

- **Precision** = TP / (TP + FP) = 22/(22+4) = **0,846** — trong số những cái model bảo là trắng, bao nhiêu cái đúng là trắng.
- **Recall** = TP / (TP + FN) = 22/(22+11) = **0,667** — trong số bí trắng thật, model tìm được bao nhiêu.
- **F1-score** = 2·P·R / (P+R) — trung bình hài hòa của precision và recall; 1 là tốt nhất.
- **Support**: số mẫu thật của mỗi nhãn. **Accuracy** = (TP+TN)/tổng = % dự đoán đúng. **Macro avg**: trung bình không trọng số (không quan tâm mất cân bằng lớp). **Weighted avg**: trung bình có trọng số theo support.
- Muốn giảm **false negative** (bỏ sót) thì theo dõi **recall**.

### Đường cong ROC + AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score
y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
plt.plot([0, 1], [0, 1], 'k--')   # đường chéo = đoán mò
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
auc = roc_auc_score(y_test, y_scores[:, 1])   # 0,975
```
- ROC vẽ **True Positive Rate (trục Y)** theo **False Positive Rate (trục X)**. Đường cong càng nhanh vọt lên trên đường chéo càng tốt. **AUC 0,975** (thang 0–1; 1,0 = hoàn hảo) → model này khá tốt.
- Thử thách: lên Kaggle tìm dataset phù hợp, tự build một model logistic.

---

## Bài 09 — Xây web app dùng model ML (Flask + Pickle, dữ liệu UFO)

### Bối cảnh: đưa model "ra đời thực"
- Train trong notebook xong, làm sao cho người khác dùng? Cần **web app**.
- Câu hỏi kiến trúc trước khi code:
  - **Web app hay mobile app?** Mobile/IoT → TensorFlow Lite (Android/iOS).
  - **Model ở đâu?** Trên cloud hay local? App có cần chạy offline không?
  - **Train bằng công nghệ gì?** TensorFlow → chuyển sang TensorFlow.js (chạy trong trình duyệt); PyTorch → xuất **ONNX** (Open Neural Network Exchange) + ONNX Runtime cho web JS; SaaS như Lobe.ai / Azure Custom Vision → xuất model cho nhiều nền tảng hoặc gọi API trên cloud.
  - Cũng có thể train model **ngay trong Flask app** (route `/train`) thay vì notebook — xem thử thách cuối bài.
- Bài này: model train bằng Python + scikit-learn trong notebook → xuất sang app web Python bằng **Flask + Pickle**.

### Hai công cụ
- **Flask**: micro-framework Python + templating engine để làm web page.
- **Pickle** 🥒: module serialize/deserialize object Python — "pickle" model = làm phẳng cấu trúc model thành file `.pkl` để web app đọc được. **CẢNH BÁO: pickle không an toàn về bản chất — cẩn thận khi un-pickle file từ nguồn không tin cậy.**

### Dữ liệu: 80.000 vụ nhìn thấy UFO (NUFORC)
- File `ufos.csv`: `city, state, country, shape, latitude, longitude, duration (seconds)`.
- Dọn dữ liệu:
```python
ufos = pd.read_csv('./data/ufos.csv')
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],
                     'Latitude': ufos['latitude'], 'Longitude': ufos['longitude']})
ufos.dropna(inplace=True)
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
from sklearn.preprocessing import LabelEncoder
ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])  # mã hóa theo thứ tự alphabet
```

### Train model: đoán quốc gia từ thời gian + tọa độ
```python
from sklearn.model_selection import train_test_split
Selected_features = ['Seconds', 'Latitude', 'Longitude']
X = ufos[Selected_features]
y = ufos['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print('Accuracy: ', accuracy_score(y_test, predictions))   # ~95%
```
- Accuracy ~95% không có gì cách mạng — vì **kinh độ/vĩ độ vốn đã suy ra được quốc gia**. Nhưng đây là bài tập trọn vòng: dọn dữ liệu thô → train → xuất → dùng trong web app.

### Pickle model
```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename, 'wb'))
model = pickle.load(open('ufo-model.pkl', 'rb'))
print(model.predict([[50, 44, -12]]))   # → '3' = UK 👽
```

### Xây Flask app
Cấu trúc thư mục:
```
web-app/
  static/
    css/
  templates/
  notebook.ipynb
  ufo-model.pkl
```
- `requirements.txt` (như `package.json` của JS): `scikit-learn, pandas, numpy, flask` → `pip install -r requirements.txt`.
- `static/css/styles.css`: giao diện (nền đen, chữ trắng, form căn giữa).
- `templates/index.html`: form nhập Seconds / Latitude / Longitude, gửi POST tới route `/predict`; dùng cú pháp template Jinja `{{ prediction_text }}` để hiện kết quả.
- `app.py`:
```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("./ufo-model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    countries = ["Australia", "Canada", "Germany", "UK", "US"]
    return render_template("index.html",
        prediction_text="Likely country: {}".format(countries[output]))

if __name__ == "__main__":
    app.run(debug=True)
```
- Chạy `python app.py` → server local, điền form là ra đáp án.
- `debug=True`: sửa code là server tự nạp lại ngay — **chỉ dùng lúc dev, TUYỆT ĐỐI không bật ở production**.
- Điểm khó nhất khi đưa model vào app: **biết đúng hình dạng (shape) dữ liệu đầu vào** mà model cần — cái này phụ thuộc hoàn toàn vào cách model được train (ở đây: 3 số theo đúng thứ tự Seconds, Latitude, Longitude).
- Bài học nghề nghiệp: trong môi trường chuyên nghiệp, **nhóm train model và nhóm làm app phải giao tiếp tốt** — bên train phải nói rõ input/output, bên app mới dùng đúng được.
- Thử thách: chuyển code train từ notebook vào trong Flask app (route `/train`) — cân nhắc ưu/nhược điểm (train trong app thì nặng, chậm, nhưng luôn dùng dữ liệu mới nhất).

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Dự đoán lượt xem video bằng hồi quy — đúng bài toán, nhưng làm theo thứ tự bài 06→07:** hỏi câu hỏi trước ("video nào sẽ viral?"), dọn dữ liệu (slot, chủ đề, độ dài, giờ đăng — mã hóa one-hot vì đều là loại phân loại), xem tương quan trước, rồi mới train. Đừng train mù trên dữ liệu lộn xộn.
2. **Tương quan bỏ lỡ pattern theo mùa:** như giá bí ngô đỉnh tháng 9–10 mà hệ số tương quan tuyến tính gần như bằng 0 — lượt xem kênh cũng có thể có pattern theo ngày lễ/tuần mà một con số tương quan không thấy. Phải vẽ biểu đồ (bar chart theo ngày/giờ) trước khi kết luận.
3. **Đặc trưng tốt > thuật toán phức tạp:** R² từ 0,07 lên 0,97 chỉ nhờ thêm đúng đặc trưng (giống, thành phố, đóng gói). Với dữ liệu kênh: chủ đề + slot + độ dài video có thể quan trọng hơn bất kỳ model "xịn" nào.
4. **Precision vs Recall cho quy trình kiểm duyệt video:** khi kiểm tra video trước khi đăng, **recall quan trọng hơn precision** — thà bắt nhầm vài video tốt (false positive) còn hơn để lọt một video lỗi (false negative). Confusion matrix là cách nghĩ đúng cho khâu kiểm tra.
5. **Pickle + Flask = mẫu để ship model ML thành web app:** đúng hình dạng của app Tính Toán Điện / app Vẽ Mạch Điện, nhưng thay công thức cố định bằng model đã train. Sau này có model ML nào (dự đoán lượt xem, gợi ý chủ đề) thì đóng gói đúng theo mẫu bài 09: train trong notebook → pickle `.pkl` → Flask load → form nhập → trả kết quả.
6. **One-hot encoding là kỹ năng dùng ngay:** bất cứ khi nào đưa dữ liệu loại (chủ đề, slot, giọng đọc) vào model số, phải one-hot — không được gán số thứ tự 0,1,2,3 rồi để model hiểu lầm là có thứ tự.

## Tiến độ khóa G11
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ **Phần 2 (bài 05–09): Regression + Web App — xong 20/11 (ghi chú này).**
- ⏳ Phần 3: Classification (bài 10–13) — KNN, Decision Trees, Random Forests, SVM.
- ⏳ Phần 4: Clustering (14–15), NLP (16–20), Time Series (21–23), Reinforcement Learning (24–25), Postscript.
- ⏳ Sau G11: S1 "Understanding Deep Learning" (sách PDF miễn phí).

## Nguồn
- Bài 05: https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/1-Tools/README.md
- Bài 06: https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/2-Data/README.md
- Bài 07: https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/3-Linear/README.md
- Bài 08: https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/4-Logistic/README.md
- Bài 09: https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
