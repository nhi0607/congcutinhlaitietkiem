
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

# ==================================================
# 1. CAU HINH STREAMLIT
# ==================================================
st.set_page_config(
    page_title="LSTM Stock Forecast & Trading Signals",
    page_icon="📈",
    layout="wide",
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")
st.markdown(
    """
    Ứng dụng kết hợp **LSTM, RSI, SMA20 và SMA50**
    để dự báo giá cổ phiếu và tạo tín hiệu tham khảo
    **MUA / BÁN / GIỮ**.
    """
)

SEED = 42


def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


# ==================================================
# 2. TAI DU LIEU YAHOO FINANCE
# ==================================================
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

    # Xu ly MultiIndex trong cac phien ban yfinance moi
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


# ==================================================
# 3. TAO SEQUENCE LSTM
# ==================================================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# ==================================================
# 4. TINH CHI BAO KY THUAT
# ==================================================
def calculate_indicators(df):
    result = df.copy()
    close = result["Close"].astype(float)

    # Duong trung binh dong
    result["SMA20"] = close.rolling(20).mean()
    result["SMA50"] = close.rolling(50).mean()

    # RSI 14 phien
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["RSI14"] = 100 - (100 / (1 + rs))

    # Neu khong co phien giam, RSI = 100
    result.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI14",
    ] = 100

    # Neu gia khong doi, RSI = 50
    result.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI14",
    ] = 50

    return result


# ==================================================
# 5. TAO TIN HIEU MUA / BAN / GIU
# ==================================================
def generate_signal(
    current_price,
    predicted_price,
    sma20,
    sma50,
    rsi,
    threshold,
):
    """
    Quy tac minh hoa:
    MUA: du bao tang du nguong, gia tren SMA20,
         SMA20 >= SMA50 va RSI tu 50 den duoi 70.
    BAN: du bao giam du nguong, gia duoi SMA20,
         SMA20 <= SMA50 va RSI duoi 50.
    GIU: cac dieu kien khong dong thuan.
    """

    expected_return = (
        predicted_price / current_price - 1
    ) * 100

    buy = (
        expected_return >= threshold
        and current_price > sma20
        and sma20 >= sma50
        and 50 <= rsi < 70
    )

    sell = (
        expected_return <= -threshold
        and current_price < sma20
        and sma20 <= sma50
        and rsi < 50
    )

    if buy:
        return (
            "MUA",
            expected_return,
            "Các điều kiện về xu hướng, RSI và dự báo "
            "đều tích cực. Có thể nghiên cứu điểm mua "
            "sau khi đánh giá rủi ro.",
        )

    if sell:
        return (
            "BÁN",
            expected_return,
            "Các điều kiện cho thấy xu hướng yếu và "
            "giá dự báo giảm. Nếu đang nắm giữ, hãy "
            "đánh giá lại vị thế và kế hoạch quản trị rủi ro.",
        )

    return (
        "GIỮ",
        expected_return,
        "Các điều kiện chưa đồng thuận. Nên chờ tín hiệu "
        "rõ hơn thay vì quyết định chỉ dựa trên LSTM.",
    )


# ==================================================
# 6. SIDEBAR - CAU HINH
# ==================================================
st.sidebar.header("⚙️ Cấu hình")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu Yahoo Finance",
    value="VNM.VN",
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=pd.Timestamp("2018-01-01").date(),
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc dữ liệu",
    value=pd.Timestamp.today().date(),
    max_value=pd.Timestamp.today().date(),
)

window_size = st.sidebar.selectbox(
    "Số phiên đầu vào LSTM",
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

batch_size = st.sidebar.selectbox(
    "Batch size",
    options=[16, 32, 64],
    index=1,
)

forecast_days = st.sidebar.slider(
    "Số phiên dự báo",
    min_value=1,
    max_value=30,
    value=10,
)

threshold = st.sidebar.slider(
    "Ngưỡng biến động dự báo (%)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5,
)

st.sidebar.caption(
    "Tín hiệu chỉ là quy tắc tham khảo, không đảm bảo lợi nhuận."
)

run_model = st.sidebar.button(
    "🚀 Chạy LSTM và phân tích",
    type="primary",
    use_container_width=True,
)


# ==================================================
# 7. KIEM TRA DAU VAO
# ==================================================
if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

if start_date >= end_date:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

if not run_model:
    st.info(
        "Chọn thông số ở thanh bên trái, sau đó nhấn "
        "**Chạy LSTM và phân tích**."
    )

    st.markdown(
        """
        ### Chức năng
        - Dữ liệu lịch sử từ Yahoo Finance.
        - Dự báo giá bằng LSTM hai lớp.
        - Đánh giá bằng RMSE, MAE và R².
        - Phân tích RSI, SMA20 và SMA50.
        - Tín hiệu MUA / BÁN / GIỮ.
        - Bảng dự báo và tải kết quả CSV.
        """
    )
    st.stop()


# ==================================================
# 8. TAI DU LIEU
# ==================================================
with st.spinner("Đang tải dữ liệu cổ phiếu..."):
    df = get_stock_data(
        ticker,
        start_date.isoformat(),
        (end_date + pd.Timedelta(days=1)).isoformat(),
    )

if df.empty:
    st.error(
        "Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu, "
        "khoảng ngày hoặc kết nối Internet."
    )
    st.stop()

data = df[["Close"]].to_numpy(dtype=np.float32)

if len(data) < max(window_size + 30, 60):
    st.error(
        f"Chỉ có {len(data)} phiên dữ liệu. "
        "Hãy chọn khoảng thời gian dài hơn hoặc giảm Window."
    )
    st.stop()

# Tinh chi bao tren toan bo chuoi gia
indicator_df = calculate_indicators(df)

# Lay chi bao tai phien cuoi cung
last_indicators = indicator_df.iloc[-1]

if last_indicators[["SMA20", "SMA50", "RSI14"]].isna().any():
    st.error("Không đủ dữ liệu để tính các chỉ báo kỹ thuật.")
    st.stop()


# ==================================================
# 9. TONG QUAN DU LIEU
# ==================================================
st.subheader(f"📊 Tổng quan: {ticker}")

col1, col2, col3 = st.columns(3)

col1.metric("Số phiên dữ liệu", f"{len(df):,}")
col2.metric(
    "Giá đóng cửa gần nhất",
    f"{float(data[-1, 0]):,.2f}",
)
col3.metric(
    "Ngày giao dịch cuối",
    df.index[-1].strftime("%d/%m/%Y"),
)

with st.expander("Xem dữ liệu lịch sử"):
    st.dataframe(df, use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["Close"], label="Giá đóng cửa")
ax.set_title(f"Lịch sử giá đóng cửa {ticker}")
ax.set_xlabel("Ngày")
ax.set_ylabel("Giá")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ==================================================
# 10. CHIA TRAIN / TEST
# ==================================================
set_seed()

train_size = int(0.8 * len(data))

train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler()

# Fit scaler chi tren train
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train_all, y_train_all = create_sequences(
    train_scaled,
    window_size,
)

val_size = max(1, int(0.1 * len(X_train_all)))

X_train_final = X_train_all[:-val_size]
y_train_final = y_train_all[:-val_size]

X_val = X_train_all[-val_size:]
y_val = y_train_all[-val_size:]

if len(X_train_final) == 0:
    st.error("Không đủ dữ liệu để huấn luyện.")
    st.stop()

test_combined = np.vstack([
    train_scaled[-window_size:],
    test_scaled,
])

X_test, y_test = create_sequences(
    test_combined,
    window_size,
)


# ==================================================
# 11. XAY DUNG VA HUAN LUYEN LSTM
# ==================================================
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

st.subheader("🧠 Huấn luyện LSTM")

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


with st.spinner("Đang huấn luyện mô hình..."):
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


# ==================================================
# 12. DANH GIA TEST
# ==================================================
test_pred_scaled = model.predict(X_test, verbose=0)

test_pred = scaler.inverse_transform(test_pred_scaled)
y_test_real = scaler.inverse_transform(y_test)

rmse = np.sqrt(
    mean_squared_error(y_test_real, test_pred)
)

mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)

st.subheader("📏 Đánh giá mô hình")

c1, c2, c3 = st.columns(3)

c1.metric("RMSE", f"{rmse:,.4f}")
c2.metric("MAE", f"{mae:,.4f}")
c3.metric("R²", f"{r2:.4f}")

st.caption(
    "R² có thể âm. Các chỉ số này đánh giá sai số dự báo "
    "trên tập test, không đảm bảo lợi nhuận giao dịch."
)


# ==================================================
# 13. BIEU DO LOSS
# ==================================================
st.subheader("📉 Train Loss và Validation Loss")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(history.history["loss"], label="Train Loss")
ax.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

ax.set_title("Loss Train vs Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ==================================================
# 14. BIEU DO DU BAO TEST
# ==================================================
st.subheader("🔍 Giá thực tế và dự báo trên tập test")

test_dates = df.index[train_size:]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    test_dates,
    y_test_real.ravel(),
    label="Giá thực tế",
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


# ==================================================
# 15. DU BAO CAC PHIEN TUONG LAI
# ==================================================
st.subheader(f"🔮 Dự báo {forecast_days} phiên tiếp theo")

all_scaled = np.vstack([
    train_scaled,
    test_scaled,
])

last_window = all_scaled[-window_size:].reshape(
    1,
    window_size,
    1,
)

future_scaled = []

for _ in range(forecast_days):
    next_pred = model.predict(
        last_window,
        verbose=0,
    )

    future_scaled.append(float(next_pred[0, 0]))

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
    "⬇️ Tải bảng dự báo CSV",
    data=forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=f"{ticker.replace('.', '_')}_forecast.csv",
    mime="text/csv",
)

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
    label="Dự báo tương lai",
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


# ==================================================
# 16. PHAN TICH RSI, SMA20, SMA50
# ==================================================
st.header("🎯 Tín hiệu giao dịch MUA / BÁN / GIỮ")

current_price = float(df["Close"].iloc[-1])
current_sma20 = float(last_indicators["SMA20"])
current_sma50 = float(last_indicators["SMA50"])
current_rsi = float(last_indicators["RSI14"])

# Gia du bao cuoi ky
predicted_price = float(future_predictions[-1])

signal, expected_return, explanation = generate_signal(
    current_price=current_price,
    predicted_price=predicted_price,
    sma20=current_sma20,
    sma50=current_sma50,
    rsi=current_rsi,
    threshold=threshold,
)

# Hien thi chi bao
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Giá hiện tại",
    f"{current_price:,.2f}",
)

c2.metric(
    "SMA20",
    f"{current_sma20:,.2f}",
    delta=f"{current_price - current_sma20:+,.2f}",
)

c3.metric(
    "SMA50",
    f"{current_sma50:,.2f}",
    delta=f"{current_price - current_sma50:+,.2f}",
)

c4.metric(
    "RSI (14)",
    f"{current_rsi:.2f}",
)

st.markdown("### Kết luận phân tích")

s1, s2 = st.columns(2)

s1.metric(
    f"Giá dự báo sau {forecast_days} phiên",
    f"{predicted_price:,.2f}",
    delta=f"{expected_return:+.2f}%",
)

s2.metric(
    "Tín hiệu hiện tại",
    signal,
)

st.info(explanation)

# Giai thich dieu kien
st.markdown("### Kiểm tra điều kiện")

conditions = pd.DataFrame({
    "Điều kiện": [
        "Giá trên SMA20",
        "SMA20 lớn hơn hoặc bằng SMA50",
        "RSI trong vùng 50 đến dưới 70",
        "Dự báo tăng đạt ngưỡng",
        "RSI dưới 50",
        "Dự báo giảm đạt ngưỡng",
    ],
    "Kết quả": [
        "Đạt" if current_price > current_sma20 else "Không đạt",
        "Đạt" if current_sma20 >= current_sma50 else "Không đạt",
        "Đạt" if 50 <= current_rsi < 70 else "Không đạt",
        "Đạt" if expected_return >= threshold else "Không đạt",
        "Đạt" if current_rsi < 50 else "Không đạt",
        "Đạt" if expected_return <= -threshold else "Không đạt",
    ],
})

st.dataframe(
    conditions,
    use_container_width=True,
    hide_index=True,
)

# ==================================================
# 17. BIEU DO CHI BAO KY THUAT
# ==================================================
st.subheader("📊 Biểu đồ RSI và đường trung bình")

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

axes[0].plot(
    indicator_df.index,
    indicator_df["Close"],
    label="Giá đóng cửa",
)

axes[0].plot(
    indicator_df.index,
    indicator_df["SMA20"],
    label="SMA20",
)

axes[0].plot(
    indicator_df.index,
    indicator_df["SMA50"],
    label="SMA50",
)

axes[0].set_title(f"Giá và đường trung bình {ticker}")
axes[0].set_ylabel("Giá")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    indicator_df.index,
    indicator_df["RSI14"],
    label="RSI14",
)

axes[1].axhline(
    70,
    linestyle="--",
    label="Quá mua (70)",
)

axes[1].axhline(
    30,
    linestyle="--",
    label="Quá bán (30)",
)

axes[1].axhline(
    50,
    linestyle=":",
    label="Mốc 50",
)

axes[1].set_ylim(0, 100)
axes[1].set_ylabel("RSI")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ==================================================
# 18. QUAN TRI RUI RO THAM KHAO
# ==================================================
st.subheader("🛡️ Quản trị rủi ro tham khảo")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    stop_loss_pct = st.number_input(
        "Cắt lỗ tham khảo (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

with risk_col2:
    take_profit_pct = st.number_input(
        "Chốt lời tham khảo (%)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
    )

stop_price = current_price * (
    1 - stop_loss_pct / 100
)

take_profit_price = current_price * (
    1 + take_profit_pct / 100
)

r1, r2 = st.columns(2)

r1.metric(
    "Giá cắt lỗ tham khảo",
    f"{stop_price:,.2f}",
)

r2.metric(
    "Giá chốt lời tham khảo",
    f"{take_profit_price:,.2f}",
)

st.caption(
    "Các mức này chỉ là ví dụ tính theo giá hiện tại, "
    "không phải mức giá tối ưu và không đảm bảo lệnh "
    "sẽ khớp tại giá đó."
)


# ==================================================
# 19. XUAT BAO CAO TIN HIEU
# ==================================================
signal_report = pd.DataFrame({
    "Thông tin": [
        "Mã cổ phiếu",
        "Ngày dữ liệu cuối",
        "Giá hiện tại",
        "Giá dự báo cuối kỳ",
        "Thay đổi dự báo (%)",
        "SMA20",
        "SMA50",
        "RSI14",
        "Tín hiệu",
        "Ngưỡng dự báo (%)",
        "Cắt lỗ tham khảo",
        "Chốt lời tham khảo",
    ],
    "Giá trị": [
        ticker,
        df.index[-1].strftime("%Y-%m-%d"),
        current_price,
        predicted_price,
        expected_return,
        current_sma20,
        current_sma50,
        current_rsi,
        signal,
        threshold,
        stop_price,
        take_profit_price,
    ],
})

st.download_button(
    "⬇️ Tải báo cáo tín hiệu giao dịch CSV",
    data=signal_report.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=f"{ticker.replace('.', '_')}_trading_signal.csv",
    mime="text/csv",
)

st.warning(
    "Tín hiệu MUA/BÁN/GIỮ là quy tắc tham khảo kết hợp "
    "dự báo LSTM và chỉ báo kỹ thuật. Chưa bao gồm phí "
    "giao dịch, thuế, thanh khoản và kiểm định chiến lược. "
    "Không sử dụng tín hiệu này như khuyến nghị đầu tư "
    "độc lập."
)

st.caption(
    "Ứng dụng phục vụ học tập và nghiên cứu. "
    "Hiệu quả của chiến lược cần được kiểm định bằng "
    "backtest trước khi cân nhắc sử dụng thực tế."
)
