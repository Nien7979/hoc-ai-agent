# G11 — Microsoft ML-For-Beginners: Phần 6 — Time Series (bài 21–23)

> Khóa: [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) — 92.346⭐, MIT, 12 tuần, 26 bài, 52 quiz. ML cổ điển bằng scikit-learn (KHÔNG deep learning). Có bản dịch tiếng Việt.
> Phần 1 (bài 01–04): `microsoft-ml-for-beginners-phan-1.md` — ML là gì, lịch sử, Responsible AI, quy trình 7 bước.
> Phần 2 (bài 05–09): `microsoft-ml-for-beginners-phan-2.md` — Regression + Web App (Flask + Pickle).
> Phần 3 (bài 10–13): `microsoft-ml-for-beginners-phan-3.md` — Classification + Web App (ONNX).
> Phần 4 (bài 14–15): `microsoft-ml-for-beginners-phan-4.md` — Clustering.
> Phần 5 (bài 16–20): `microsoft-ml-for-beginners-phan-5.md` — NLP.
> Ghi chú này: đọc từ clone local `~/workspace/knowledge/ml-for-beginners/` (1,9 GB, HEAD 8e5c0dff). Ghi chú lưu ở `hoc-ai-agent/`, KHÔNG sửa gì trong repo khóa học.

---

## Bài 21 — Giới thiệu về dự báo chuỗi thời gian (Time Series Forecasting)

### Time series là gì
- **Time series** = chuỗi điểm dữ liệu được sắp xếp theo **thứ tự thời gian**, thường cách đều nhau. Ví dụ: giá đóng cửa hàng ngày của Dow Jones, phụ tải điện theo giờ, thời tiết, động đất.
- **Time series analysis** = phân tích chuỗi đó (miền tần số / miền thời gian, tuyến tính / phi tuyến…).
- **Time series forecasting** = dùng model dự đoán **giá trị tương lai** dựa trên pattern của dữ liệu quá khứ — như "quả cầu pha lê" cho giá cả, tồn kho, chuỗi cung ứng. Deep learning bắt đầu được dùng, nhưng **ML cổ điển vẫn chiếm ưu thế** trong lĩnh vực này.
- Dataset bài học: **GEFCom2014** — 3 năm dữ liệu **phụ tải điện + nhiệt độ theo giờ, 2012–2014** (cuộc thi Global Energy Forecasting Competition 2014). Mục tiêu: dự báo **một bước tiếp theo** chỉ từ lịch sử phụ tải.

### Các đặc tính của dữ liệu chuỗi thời gian ("nhiễu" cần xử lý)
- **Trends (xu hướng):** tăng/giảm đo được theo thời gian.
- **Seasonality (tính mùa vụ):** dao động có chu kỳ — ví dụ đợt mua sắm lễ hội làm tăng doanh số.
- **Outliers (điểm ngoại lai):** lệch xa khỏi phương sai chuẩn của dữ liệu.
- **Long-run cycle (chu kỳ dài):** suy thoái kinh tế kéo dài hơn 1 năm, độc lập với tính mùa vụ.
- **Constant variance (phương sai không đổi):** dao động ổn định theo thời gian, ví dụ mức dùng điện ngày/đêm.
- **Abrupt changes (thay đổi đột ngột):** COVID đóng cửa doanh nghiệp làm dữ liệu đổi hẳn.
- Bài tập trực quan: vẽ toàn bộ 3 năm phụ tải, rồi vẽ riêng **tuần đầu tháng 7/2014** — chu kỳ ngày/đêm hiện rõ như sóng.

---

## Bài 22 — Dự báo chuỗi thời gian với ARIMA

### ARIMA là gì
- **ARIMA = AutoRegressive Integrated Moving Average** — model liên hệ **giá trị hiện tại** của chuỗi với **giá trị quá khứ** và **sai số dự đoán quá khứ**. Đặc biệt phù hợp với dữ liệu **không dừng (non-stationary)**.
- **Stationarity (tính dừng):** phân phối của dữ liệu **không thay đổi khi dịch chuyển theo thời gian**. Dữ liệu không dừng có dao động do trend → phải biến đổi mới phân tích được. Tính mùa vụ có thể loại bằng **seasonal-differencing**.
- **Differencing (lấy hiệu):** biến đổi dữ liệu không dừng thành dừng bằng cách loại bỏ trend không hằng số — "loại bỏ thay đổi về mức của chuỗi, khử trend và tính mùa vụ, ổn định trung bình của chuỗi".
- **AR — AutoRegressive (tự hồi quy):** biến đang xét được hồi quy trên **chính các giá trị lag (trước đó)** của nó. Ví dụ: doanh số bút chì mỗi tháng — tháng này dự đoán từ các tháng trước.
- **I — Integrated (tích hợp):** khác ARMA ở chỗ ARIMA áp dụng **bước differencing** để khử tính không dừng.
- **MA — Moving Average (trung bình trượt):** biến đầu ra được xác định bằng giá trị hiện tại và quá khứ của các lag (sai số dự đoán trước).
- **Tham số:** `p` = phần tự hồi quy (dùng giá trị quá khứ), `d` = lượng differencing, `q` = phần trung bình trượt. Nếu dữ liệu có tính mùa vụ → dùng **SARIMA** với thêm `P, D, Q` cho thành phần mùa vụ + chu kỳ mùa (24 với dữ liệu theo giờ có chu kỳ ngày).

### Xây model ARIMA với dữ liệu phụ tải điện
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# Chia train/test THEO THỜI GIAN — test phải là khoảng thời gian SAU train
# (không được để model "nhìn thấy tương lai")
train_start_dt = '2014-11-01 00:00:00'
test_start_dt  = '2014-12-30 00:00:00'
train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]  # (1416, 1)
test  = energy.copy()[energy.index >= test_start_dt][['load']]                                      # (48, 1)

scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)   # scale về (0, 1)
test['load']  = scaler.transform(test)

order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)   # chu kỳ mùa = 24 giờ
model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
results = model.fit()
```
- Chọn tham số p/d/q là việc **chủ quan và tốn thời gian** — có thể dùng `auto_arima()` của thư viện pmdarima để tự chọn.

### Walk-forward validation — chuẩn vàng đánh giá model chuỗi thời gian
- Quy trình: train trên tập train → dự đoán **bước tiếp theo** → so với giá trị thật → **mở rộng tập train bằng giá trị thật vừa biết** → lặp lại.
- Giữ **cửa sổ train cố định** (720 giờ = 30 ngày) để train hiệu quả: mỗi lần thêm quan sát mới vào cuối, bỏ quan sát cũ nhất ở đầu.
- Chi phí: phải train rất nhiều model — chấp nhận được với dữ liệu nhỏ/model đơn giản, có thể thành vấn đề ở quy mô lớn.
- Trong bài: `HORIZON = 3` (dự báo 3 giờ tới), `order = (2, 1, 0)`, `seasonal_order = (1, 1, 0, 24)`. Ví dụ dự đoán đầu tiên: predicted [0,32; 0,29; 0,28] vs expected [0,329; 0,290; 0,274] — rất sát.

### Đo độ chính xác bằng MAPE
- **MAPE (Mean Absolute Percentage Error)** = trung bình |thực tế − dự đoán| / thực tế × 100%. MAPE = 10 nghĩa là dự đoán lệch trung bình 10%.
- Kết quả bài học: **MAPE dự báo 1 bước = 0,56%**; **MAPE dự báo nhiều bước (3 giờ) = 1,15%** — rất tốt. Ví dụ: dự đoán 3.008,74 vs thực tế 3.023,00.
- Vẽ đồ thị actual (đỏ) vs prediction (xanh) theo thời gian để nhìn trực quan.

---

## Bài 23 — Dự báo chuỗi thời gian với SVR

### SVR là gì, vì sao cần nó
- **SVR (Support Vector Regressor)** = SVM dùng cho hồi quy — tìm **siêu phẳng (hyperplane)** là đường fit tốt nhất có nhiều điểm dữ liệu nhất. **Kernel** biến đổi dữ liệu lên không gian nhiều chiều hơn để dễ tách.
- **Vì sao SVR sau ARIMA:** ARIMA là phương pháp tuyến tính rất thành công, nhưng dữ liệu chuỗi thời gian thường có **tính phi tuyến** mà model tuyến tính không map được. Khả năng xử lý phi tuyến của SVM làm SVR thành công trong dự báo chuỗi thời gian.

### Xây model SVR (cùng dữ liệu phụ tải điện, cùng cách chia train/test)
- **Biến đổi thành dữ liệu theo bước thời gian (timesteps):** lấy `timesteps = 5` — input = 4 giá trị đầu, output = giá trị thứ 5:
```python
timesteps = 5
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[:, :, 0]
# train: (1412, 5) → x_train (1412, 4), y_train (1412, 1)
# test:  (44, 5)   → x_test  (44, 4),   y_test  (44, 1)
```
- **Model:**
```python
from sklearn.svm import SVR
model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
model.fit(x_train, y_train[:, 0])
y_train_pred = model.predict(x_train).reshape(-1, 1)
y_test_pred  = model.predict(x_test).reshape(-1, 1)
# inverse_transform về thang đo gốc rồi tính MAPE
```
- **Kết quả:** MAPE train **1,72%**, MAPE test **1,26%**, MAPE toàn bộ dataset (26.300 mẫu) **2,06%** — rất tốt.
- **Thử thách:** chỉnh hyperparameter (gamma, C, epsilon), thử kernel khác, thử các giá trị `timesteps` (cửa sổ nhìn lại) khác nhau để xem bộ nào tốt nhất trên test.

---

## Áp dụng ngay vào công việc (theo chỉ đạo của Hoai 20/11)

1. **Dự báo lượt xem kênh:** lượt xem theo ngày/slot là một time series — dùng ARIMA/SVR dự báo lượt xem video ngày mai, tuần tới từ lịch sử; biết trước slot nào đang đi xuống để đổi chủ đề kịp thời.
2. **Tách seasonality ra trước khi kết luận:** lượt xem có chu kỳ theo ngày trong tuần (cuối tuần cao hơn?) và theo giờ đăng — giống chu kỳ ngày/đêm của phụ tải điện. Phải tách tính mùa vụ ra, nếu không sẽ kết luận sai "video này kém" trong khi chỉ là ngày thường.
3. **Walk-forward validation = cách đánh giá đúng cho dữ liệu theo thời gian:** không chia train/test ngẫu nhiên (rò rỉ tương lai); test set phải là khoảng thời gian SAU train set. Áp dụng cho mọi phân tích "dự đoán" dữ liệu kênh.
4. **MAPE — thước đo dễ hiểu:** sai bao nhiêu % — dùng để so sánh các cách dự báo lượt xem với nhau, thay vì cảm tính.
5. **Differencing — khử trend trước khi phân tích:** subs/lượt xem kênh đang tăng trưởng theo trend; phải khử trend trước khi tìm pattern, nếu không mọi tương quan đều bị trend làm lệch.
6. **Khi quan hệ phi tuyến, ARIMA không đủ:** lượt xem tăng vọt không theo đường thẳng — thử SVR/kernel để bắt pattern phi tuyến.
7. **Timesteps = cửa sổ nhìn lại:** với kênh, dùng 4–7 ngày gần nhất để dự báo ngày mai — thử nhiều cửa sổ khác nhau, chọn cái cho MAPE thấp nhất.

## Tiến độ khóa G11
- ✅ Phần 1 (bài 01–04): Introduction — xong 20/11.
- ✅ Phần 2 (bài 05–09): Regression + Web App — xong 20/11.
- ✅ Phần 3 (bài 10–13): Classification — xong 20/11.
- ✅ Phần 4 (bài 14–15): Clustering — xong 20/11.
- ✅ Phần 5 (bài 16–20): NLP — xong 20/11.
- ✅ **Phần 6 (bài 21–23): Time Series — xong 20/11 (ghi chú này).**
- ⏳ Phần 7: Reinforcement Learning (bài 24–25) + Postscript.
- ⏳ Sau G11: S1 "Understanding Deep Learning" (sách PDF miễn phí), rồi các khóa đã đăng ký khác (freeCodeCamp Python v9, Udemy 73 khóa, GLM-5.3-Flash build + train) — **quy tắc của Hoai (20/11 ~23:19): khóa nào đến bước cần anh làm (thi chứng chỉ, nhập token,…) thì BỎ QUA, học tiếp khóa khác.**

## Nguồn
- Bài 21: https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/1-Introduction/README.md
- Bài 22: https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/README.md
- Bài 23: https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/README.md
- Khóa học: https://github.com/microsoft/ML-For-Beginners (MIT)
