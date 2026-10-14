
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# =========================================
# 1. CAU HINH STREAMLIT
# =========================================
st.set_page_config(
    page_title="Dự báo cổ phiếu VNM bằng LSTM",
    page_icon="📈",
    layout="wide",
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU VNM")
st.markdown(
    "Ứng dụng sử dụng **LSTM (Deep Learning)** "
    "để dự báo giá đóng cửa cổ phiếu Vinamilk."
)

SEED = 42


def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


# =========================================
# 2. TAI DU LIEU YAHOO FINANCE
# =========================================
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    # Xu ly MultiIndex cua yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [
            col for col in df.columns
            if "Close" in col
        ]

        if not close_cols:
            return pd.DataFrame()

        close = df[close_cols[0]]
        df = close.to_frame(name="Close")
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()

        df = df[["Close"]]

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


# =========================================
# 3. TAO DU LIEU CHUOI CHO LSTM
# =========================================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================================
# 4. THANH DIEU KHIEN
# =========================================
st.sidebar.header("⚙️ Cấu hình mô hình")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="VNM.VN",
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=pd.Timestamp("2018-01-01").date(),
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=pd.Timestamp.today().date(),
    max_value=pd.Timestamp.today().date(),
)

window_size = st.sidebar.selectbox(
    "Số phiên đầu vào",
    options=[30, 60, 90, 120],
    index=1,
)

epochs = st.sidebar.slider(
    "Số Epoch tối đa",
    min_value=5,
    max_value=100,
    value=10,
    step=5,
)

forecast_days = st.sidebar.slider(
    "Số phiên dự báo tương lai",
    min_value=1,
    max_value=30,
    value=10,
)

batch_size = st.sidebar.selectbox(
    "Batch size",
    options=[16, 32, 64],
    index=1,
)

run_model = st.sidebar.button(
    "🚀 Chạy mô hình LSTM",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "Dữ liệu được lấy từ Yahoo Finance. "
    "Kết quả chỉ phục vụ mục đích nghiên cứu."
)

# =========================================
# 5. KIEM TRA DAU VAO
# =========================================
if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

if start_date >= end_date:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

if not run_model:
    st.info(
        "Thiết lập thông số ở thanh bên trái, "
        "sau đó nhấn **Chạy mô hình LSTM**."
    )

    st.markdown("### Hướng dẫn")
    st.write("1. Chọn mã cổ phiếu, mặc định là `VNM.VN`.")
    st.write("2. Chọn thời gian dữ liệu.")
    st.write("3. Điều chỉnh các thông số LSTM nếu cần.")
    st.write("4. Nhấn nút chạy để huấn luyện và xem kết quả.")
    st.stop()

# =========================================
# 6. TAI DU LIEU
# =========================================
with st.spinner("Đang tải dữ liệu cổ phiếu..."):
    df = get_stock_data(
        ticker,
        start_date.isoformat(),
        (end_date + pd.Timedelta(days=1)).isoformat(),
    )

if df.empty:
    st.error(
        "Không lấy được dữ liệu. Kiểm tra mã cổ phiếu, "
        "khoảng ngày hoặc kết nối Internet."
    )
    st.stop()

data = df[["Close"]].to_numpy(dtype=np.float32)

if len(data) < window_size + 30:
    st.error(
        f"Chỉ có {len(data)} phiên dữ liệu. "
        "Hãy chọn khoảng thời gian dài hơn "
        "hoặc giảm số phiên đầu vào."
    )
    st.stop()

# =========================================
# 7. HIEN THI DU LIEU
# =========================================
st.subheader(f"📊 Tổng quan dữ liệu: {ticker}")

col1, col2, col3 = st.columns(3)

col1.metric("Tổng số phiên", f"{len(df):,}")
col2.metric("Giá đóng cửa gần nhất", f"{data[-1, 0]:,.2f}")
col3.metric(
    "Ngày giao dịch cuối",
    df.index[-1].strftime("%d/%m/%Y"),
)

with st.expander("Xem dữ liệu lịch sử"):
    st.dataframe(df, use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, data[:, 0], label="Giá đóng cửa")
ax.set_title(f"Lịch sử giá đóng cửa {ticker}")
ax.set_xlabel("Ngày")
ax.set_ylabel("Giá")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =========================================
# 8. CHIA TRAIN / TEST
# =========================================
set_seed()

train_size = int(0.8 * len(data))

train_data = data[:train_size]
test_data = data[train_size:]

# Fit scaler chi tren train
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train_all, y_train_all = create_sequences(
    train_scaled,
    window_size,
)

# Lay 10% cuoi tap train lam validation
val_size = max(1, int(0.1 * len(X_train_all)))

X_train_final = X_train_all[:-val_size]
y_train_final = y_train_all[:-val_size]

X_val = X_train_all[-val_size:]
y_val = y_train_all[-val_size:]

if len(X_train_final) == 0:
    st.error("Không đủ dữ liệu để huấn luyện mô hình.")
    st.stop()

# Noi window cuoi train voi test de tao chuoi test
test_combined = np.vstack([
    train_scaled[-window_size:],
    test_scaled,
])

X_test, y_test = create_sequences(
    test_combined,
    window_size,
)

# =========================================
# 9. XAY DUNG MO HINH LSTM
# =========================================
model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# =========================================
# 10. HUAN LUYEN
# =========================================
st.subheader("🧠 Huấn luyện mô hình")

progress = st.progress(0)
status = st.empty()


class TrainingProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress.progress(
            min((epoch + 1) / epochs, 1.0)
        )
        status.text(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {logs.get('loss', 0):.6f} | "
            f"Validation Loss: {logs.get('val_loss', 0):.6f}"
        )


with st.spinner("LSTM đang học từ dữ liệu lịch sử..."):
    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[
            early_stop,
            TrainingProgress(),
        ],
        verbose=0,
    )

status.success(
    f"Huấn luyện hoàn tất sau "
    f"{len(history.history['loss'])} epoch."
)

# =========================================
# 11. DU BAO TAP TEST
# =========================================
test_pred_scaled = model.predict(X_test, verbose=0)

test_pred = scaler.inverse_transform(test_pred_scaled)
y_test_real = scaler.inverse_transform(y_test)

rmse = np.sqrt(
    mean_squared_error(y_test_real, test_pred)
)
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)

st.subheader("📏 Đánh giá mô hình trên tập test")

c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{rmse:,.4f}")
c2.metric("MAE", f"{mae:,.4f}")
c3.metric("R²", f"{r2:.4f}")

st.caption(
    "RMSE và MAE càng thấp thường càng tốt. "
    "R² có thể âm nếu mô hình dự báo kém."
)

# =========================================
# 12. BIEU DO LOSS
# =========================================
st.subheader("📉 Train Loss và Validation Loss")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history["loss"], label="Train Loss")
ax.plot(history.history["val_loss"], label="Validation Loss")
ax.set_title("Loss Train vs Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =========================================
# 13. BIEU DO TEST
# =========================================
st.subheader("🔍 Giá thực tế và giá dự báo trên tập test")

test_dates = df.index[train_size:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    test_dates,
    y_test_real.ravel(),
    label="Giá thật",
)
ax.plot(
    test_dates,
    test_pred.ravel(),
    label="Dự báo LSTM",
)
ax.set_title(f"Giá thực tế và dự báo {ticker}")
ax.set_xlabel("Ngày")
ax.set_ylabel("Giá đóng cửa")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =========================================
# 14. DU BAO CAC PHIEN TIEP THEO
# =========================================
st.subheader(
    f"🔮 Dự báo {forecast_days} phiên tiếp theo"
)

all_scaled = np.vstack([train_scaled, test_scaled])

last_window = all_scaled[-window_size:].reshape(
    1, window_size, 1
)

future_scaled = []

for _ in range(forecast_days):
    next_pred = model.predict(last_window, verbose=0)

    future_scaled.append(next_pred[0, 0])

    last_window = np.concatenate(
        [
            last_window[:, 1:, :],
            next_pred.reshape(1, 1, 1),
        ],
        axis=1,
    )

future_predictions = scaler.inverse_transform(
    np.array(future_scaled).reshape(-1, 1)
).ravel()

# Tao ngay lam viec gan dung; chua loai tru ngay nghi le
future_dates = pd.bdate_range(
    start=df.index[-1] + pd.offsets.BDay(1),
    periods=forecast_days,
)

forecast_df = pd.DataFrame({
    "Ngày dự báo": future_dates.strftime("%Y-%m-%d"),
    "Giá dự báo": future_predictions,
})

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Tải kết quả dự báo CSV",
    data=forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=f"{ticker.replace('.', '_')}_forecast.csv",
    mime="text/csv",
)

# =========================================
# 15. BIEU DO DU BAO TUONG LAI
# =========================================
fig, ax = plt.subplots(figsize=(12, 5))

n_history = min(100, len(df))

ax.plot(
    df.index[-n_history:],
    df["Close"].iloc[-n_history:],
    label="Giá lịch sử",
)

ax.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Giá dự báo tương lai",
)

ax.axvline(
    df.index[-1],
    linestyle=":",
    label="Ngày cuối dữ liệu",
)

ax.set_title(f"Dự báo giá cổ phiếu {ticker}")
ax.set_xlabel("Ngày")
ax.set_ylabel("Giá đóng cửa")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.warning(
    "Dự báo tương lai là dự báo đệ quy: mỗi dự báo tiếp theo "
    "sử dụng dự báo trước làm đầu vào. Sai số có thể tích lũy. "
    "Ngày hiển thị là ngày làm việc gần đúng, chưa loại trừ "
    "các ngày nghỉ lễ của thị trường."
)

st.caption(
    "Ứng dụng phục vụ học tập và nghiên cứu, "
    "không phải khuyến nghị mua hoặc bán cổ phiếu."
)
