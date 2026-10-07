import streamlit as st
import pandas as pd
import numpy as np
import math
st.image("barbie.webp")
# =========================
# CẤU HÌNH
# =========================
st.set_page_config(
    page_title="Smart Savings - Máy tính tiền gửi ngân hàng_Nguyễn Quỳnh Nhi",
    page_icon="💰",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #fff5f5, #fff);
        border: 1px solid #eee;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #eeeeee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# =========================
# HÀM
# =========================
def format_money(value):
    return f"{value:,.0f} VNĐ"


def format_short_money(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f} triệu"
    else:
        return format_money(value)


def tinh_lai_don(principal, rate, months):
    interest = principal * rate * months / 12
    total = principal + interest
    return interest, total


def tinh_lai_kep(principal, rate, months, frequency=12):
    period_rate = rate / frequency
    periods = months / 12 * frequency

    total = principal * (1 + period_rate) ** periods
    interest = total - principal

    return interest, total


def tao_bang_tang_truong(principal, rate, months, compound=True):
    data = []

    for month in range(0, months + 1):

        if compound:
            total = principal * (1 + rate / 12) ** month
        else:
            total = principal * (1 + rate * month / 12)

        interest = total - principal

        data.append({
            "Tháng": month,
            "Tiền gốc": principal,
            "Tiền lãi": interest,
            "Tổng tiền": total
        })

    return pd.DataFrame(data)


def tinh_gui_hang_thang(
    initial,
    monthly_deposit,
    rate,
    months
):
    balance = initial
    total_deposit = initial

    rows = []

    for month in range(1, months + 1):

        interest = balance * rate / 12

        balance += interest
        balance += monthly_deposit

        total_deposit += monthly_deposit

        rows.append({
            "Tháng": month,
            "Tiền gửi thêm": monthly_deposit,
            "Lãi tháng": interest,
            "Tổng tiền": balance,
            "Tổng tiền đã gửi": total_deposit,
            "Tiền lãi": balance - total_deposit
        })

    return pd.DataFrame(rows)


# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">💰 SMART SAVINGS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Công cụ tính toán, so sánh và lập kế hoạch tiền gửi tiết kiệm'
    '</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.header("⚙️ Cài đặt")

    don_vi = st.selectbox(
        "Đơn vị nhập tiền",
        [
            "VNĐ",
            "Triệu VNĐ",
            "Tỷ VNĐ"
        ]
    )

    st.divider()

    st.info(
        """
        💡 **Mẹo**

        Lãi kép thường có lợi hơn lãi đơn khi:
        
        - Kỳ hạn dài
        - Lãi được nhập vào vốn
        - Lãi suất đủ cao
        """
    )

# =========================
# NHẬP DỮ LIỆU
# =========================
st.subheader("📋 Thông tin khoản tiền gửi")

col1, col2, col3, col4 = st.columns(4)

with col1:

    tien_input = st.number_input(
        "💵 Số tiền gửi",
        min_value=0.0,
        value=100.0 if don_vi == "Triệu VNĐ" else 0.1 if don_vi == "Tỷ VNĐ" else 100_000_000.0,
        step=1.0 if don_vi != "VNĐ" else 1_000_000.0
    )

    if don_vi == "VNĐ":
        tien_gui = tien_input
    elif don_vi == "Triệu VNĐ":
        tien_gui = tien_input * 1_000_000
    else:
        tien_gui = tien_input * 1_000_000_000


with col2:

    ky_han = st.number_input(
        "📅 Kỳ hạn (tháng)",
        min_value=1,
        max_value=120,
        value=12,
        step=1
    )


with col3:

    lai_suat = st.number_input(
        "📈 Lãi suất (%/năm)",
        min_value=0.0,
        max_value=100.0,
        value=6.0,
        step=0.1,
        format="%.2f"
    )


with col4:

    loai_lai = st.selectbox(
        "🧮 Phương pháp",
        [
            "Lãi đơn",
            "Lãi kép"
        ]
    )


hinh_thuc_nhan = st.selectbox(
    "💳 Hình thức nhận lãi",
    [
        "Lãnh lãi theo tháng",
        "Lãnh lãi theo quý",
        "Lãnh lãi cuối kỳ"
    ]
)

# =========================
# CẢNH BÁO
# =========================
if hinh_thuc_nhan == "Lãnh lãi theo quý" and ky_han % 3 != 0:

    st.warning(
        "⚠️ Kỳ hạn hiện tại không chia hết cho 3 tháng. "
        "Phần tháng lẻ sẽ không được tính vào chu kỳ nhận lãi theo quý."
    )


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Tính lãi",
    "📊 Biểu đồ",
    "⚖️ So sánh",
    "🎯 Mục tiêu",
    "➕ Gửi thêm hàng tháng"
])


# =====================================================
# TAB 1 - TÍNH LÃI
# =====================================================
with tab1:

    if st.button(
        "🧮 TÍNH LÃI NGAY",
        type="primary",
        use_container_width=True
    ):

        if tien_gui <= 0:

            st.error("Vui lòng nhập số tiền gửi lớn hơn 0.")
            st.stop()

        r = lai_suat / 100
        months = ky_han

        # ---------------------
        # LÃI ĐƠN
        # ---------------------
        if loai_lai == "Lãi đơn":

            tong_lai = tien_gui * r * months / 12
            tong_tien = tien_gui + tong_lai

            if hinh_thuc_nhan == "Lãnh lãi theo tháng":
                lai_dinh_ky = tien_gui * r / 12

            elif hinh_thuc_nhan == "Lãnh lãi theo quý":
                lai_dinh_ky = tien_gui * r / 4

            else:
                lai_dinh_ky = tong_lai

        # ---------------------
        # LÃI KÉP
        # ---------------------
        else:

            if hinh_thuc_nhan == "Lãnh lãi theo quý":

                so_ky = months // 3

                lai_ky = r / 4

                tong_tien = tien_gui * (1 + lai_ky) ** so_ky

                tong_lai = tong_tien - tien_gui

                lai_dinh_ky = (
                    tong_lai / so_ky
                    if so_ky > 0
                    else 0
                )

            else:

                lai_thang = r / 12

                tong_tien = tien_gui * (
                    1 + lai_thang
                ) ** months

                tong_lai = tong_tien - tien_gui

                if hinh_thuc_nhan == "Lãnh lãi theo tháng":
                    lai_dinh_ky = tong_lai / months
                else:
                    lai_dinh_ky = tong_lai

        # =========================
        # KẾT QUẢ
        # =========================
        st.divider()

        st.subheader("📊 Kết quả")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "💵 Tiền gốc",
                format_short_money(tien_gui)
            )

        with c2:
            st.metric(
                "📈 Tổng lãi",
                format_short_money(tong_lai)
            )

        with c3:
            st.metric(
                "💰 Tổng nhận",
                format_short_money(tong_tien)
            )

        with c4:

            ty_suat = tong_lai / tien_gui * 100

            st.metric(
                "📊 Tỷ suất sinh lời",
                f"{ty_suat:.2f}%"
            )

        st.success(
            f"### 💰 Tổng số tiền sau {ky_han} tháng: "
            f"{format_money(tong_tien)}"
        )

        # =========================
        # CHI TIẾT
        # =========================
        st.subheader("📝 Chi tiết khoản tiền gửi")

        data = {
            "Nội dung": [
                "Tiền gửi",
                "Lãi suất",
                "Kỳ hạn",
                "Phương pháp",
                "Hình thức nhận lãi",
                "Lãi định kỳ",
                "Tổng tiền lãi",
                "Tổng gốc + lãi"
            ],
            "Giá trị": [
                format_money(tien_gui),
                f"{lai_suat:.2f}%/năm",
                f"{ky_han} tháng",
                loai_lai,
                hinh_thuc_nhan,
                format_money(lai_dinh_ky),
                format_money(tong_lai),
                format_money(tong_tien)
            ]
        }

        st.table(pd.DataFrame(data))

        # =========================
        # NHẬN XÉT
        # =========================
        if ty_suat >= 20:

            st.success(
                "🔥 Khoản tiền gửi có mức tăng trưởng khá cao "
                "so với số vốn ban đầu."
            )

        elif ty_suat >= 5:

            st.info(
                "👍 Khoản tiền gửi đang tạo ra mức lợi nhuận ổn định."
            )

        else:

            st.warning(
                "💡 Lợi nhuận tương đối thấp. "
                "Bạn có thể thử so sánh với mức lãi suất khác."
            )


# =====================================================
# TAB 2 - BIỂU ĐỒ
# =====================================================
with tab2:

    st.subheader("📊 Biểu đồ tăng trưởng")

    if tien_gui <= 0:

        st.warning("Vui lòng nhập số tiền gửi lớn hơn 0.")

    else:

        r = lai_suat / 100

        df = tao_bang_tang_truong(
            tien_gui,
            r,
            ky_han,
            compound=(loai_lai == "Lãi kép")
        )

        chart_df = df.set_index("Tháng")[
            ["Tiền gốc", "Tiền lãi", "Tổng tiền"]
        ]

        st.line_chart(chart_df)

        st.subheader("📋 Chi tiết từng tháng")

        display_df = df.copy()

        display_df["Tiền gốc"] = display_df[
            "Tiền gốc"
        ].apply(format_money)

        display_df["Tiền lãi"] = display_df[
            "Tiền lãi"
        ].apply(format_money)

        display_df["Tổng tiền"] = display_df[
            "Tổng tiền"
        ].apply(format_money)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# TAB 3 - SO SÁNH
# =====================================================
with tab3:

    st.subheader("⚖️ So sánh lãi đơn và lãi kép")

    r = lai_suat / 100

    lai_don, tong_don = tinh_lai_don(
        tien_gui,
        r,
        ky_han
    )

    lai_kep, tong_kep = tinh_lai_kep(
        tien_gui,
        r,
        ky_han
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### 🟦 Lãi đơn")

        st.metric(
            "Tổng lãi",
            format_money(lai_don)
        )

        st.metric(
            "Tổng nhận",
            format_money(tong_don)
        )

    with c2:

        st.markdown("### 🟩 Lãi kép")

        st.metric(
            "Tổng lãi",
            format_money(lai_kep)
        )

        st.metric(
            "Tổng nhận",
            format_money(tong_kep)
        )

    chenh_lech = lai_kep - lai_don

    if chenh_lech > 0:

        st.success(
            f"📈 Lãi kép mang lại thêm "
            f"**{format_money(chenh_lech)}** "
            f"so với lãi đơn."
        )

    else:

        st.info(
            "Hai phương pháp đang cho kết quả tương đương."
        )

    # Biểu đồ
    compare_df = pd.DataFrame({
        "Phương pháp": [
            "Lãi đơn",
            "Lãi kép"
        ],
        "Tổng tiền": [
            tong_don,
            tong_kep
        ]
    })

    st.bar_chart(
        compare_df.set_index("Phương pháp")
    )


# =====================================================
# TAB 4 - MỤC TIÊU
# =====================================================
with tab4:

    st.subheader("🎯 Tính số tiền cần gửi để đạt mục tiêu")

    muc_tieu = st.number_input(
        "💰 Số tiền mục tiêu (VNĐ)",
        min_value=0.0,
        value=200_000_000.0,
        step=10_000_000.0
    )

    r = lai_suat / 100

    if st.button(
        "🎯 Tính số tiền cần gửi",
        use_container_width=True
    ):

        if muc_tieu <= 0:

            st.error("Mục tiêu phải lớn hơn 0.")

        elif lai_suat <= 0:

            so_tien_can_gui = muc_tieu

            st.info(
                "Lãi suất bằng 0%, do đó số tiền cần gửi "
                "bằng đúng số tiền mục tiêu."
            )

            st.success(
                f"Bạn cần gửi: **{format_money(so_tien_can_gui)}**"
            )

        else:

            # Lãi kép theo tháng
            monthly_rate = r / 12

            so_tien_can_gui = (
                muc_tieu /
                (1 + monthly_rate) ** ky_han
            )

            loi_nhuan = (
                muc_tieu - so_tien_can_gui
            )

            st.success(
                f"### 💰 Cần gửi khoảng "
                f"{format_money(so_tien_can_gui)}"
            )

            st.metric(
                "Lợi nhuận dự kiến",
                format_money(loi_nhuan)
            )


# =====================================================
# TAB 5 - GỬI THÊM HÀNG THÁNG
# =====================================================
with tab5:

    st.subheader("➕ Mô phỏng gửi thêm tiền hàng tháng")

    tien_ban_dau = st.number_input(
        "💵 Tiền ban đầu",
        min_value=0.0,
        value=50_000_000.0,
        step=5_000_000.0
    )

    tien_gui_them = st.number_input(
        "➕ Số tiền gửi thêm mỗi tháng",
        min_value=0.0,
        value=5_000_000.0,
        step=500_000.0
    )

    if st.button(
        "🚀 Mô phỏng tích lũy",
        use_container_width=True
    ):

        if tien_ban_dau <= 0 and tien_gui_them <= 0:

            st.error(
                "Bạn cần nhập tiền ban đầu hoặc tiền gửi thêm."
            )

        else:

            r = lai_suat / 100

            df_tich_luy = tinh_gui_hang_thang(
                tien_ban_dau,
                tien_gui_them,
                r,
                ky_han
            )

            tong_da_gui = df_tich_luy.iloc[-1][
                "Tổng tiền đã gửi"
            ]

            tong_cuoi_ky = df_tich_luy.iloc[-1][
                "Tổng tiền"
            ]

            tong_lai = df_tich_luy.iloc[-1][
                "Tiền lãi"
            ]

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "💵 Tổng tiền đã gửi",
                    format_short_money(tong_da_gui)
                )

            with c2:

                st.metric(
                    "📈 Tổng tiền lãi",
                    format_short_money(tong_lai)
                )

            with c3:

                st.metric(
                    "💰 Giá trị cuối kỳ",
                    format_short_money(tong_cuoi_ky)
                )

            chart = df_tich_luy.set_index(
                "Tháng"
            )[[
                "Tổng tiền đã gửi",
                "Tiền lãi",
                "Tổng tiền"
            ]]

            st.line_chart(chart)

            st.subheader("📋 Bảng tích lũy")

            display_df = df_tich_luy.copy()

            for col in [
                "Tiền gửi thêm",
                "Lãi tháng",
                "Tổng tiền",
                "Tổng tiền đã gửi",
                "Tiền lãi"
            ]:

                display_df[col] = display_df[
                    col
                ].apply(format_money)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# SO SÁNH NHIỀU LÃI SUẤT
# =====================================================
st.divider()

st.subheader("🏦 So sánh nhiều mức lãi suất")

rates = st.multiselect(
    "Chọn các mức lãi suất muốn so sánh",
    [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    default=[5.0, 6.0, 7.0]
)

if rates:

    results = []

    for rate in rates:

        r = rate / 100

        total = tien_gui * (
            1 + r / 12
        ) ** ky_han

        interest = total - tien_gui

        results.append({
            "Lãi suất": f"{rate:.2f}%",
            "Tiền lãi": interest,
            "Tổng tiền": total
        })

    compare = pd.DataFrame(results)

    st.dataframe(
        compare.style.format({
            "Tiền lãi": "{:,.0f} VNĐ",
            "Tổng tiền": "{:,.0f} VNĐ"
        }),
        use_container_width=True,
        hide_index=True
    )

    chart = compare.set_index(
        "Lãi suất"
    )[["Tổng tiền"]]

    st.bar_chart(chart)


# =====================================================
# FOOTER
# =====================================================
st.divider()

st.caption(
    "💡 Công cụ mang tính chất mô phỏng tài chính. "
    "Lãi suất thực tế có thể phụ thuộc vào ngân hàng, "
    "sản phẩm tiền gửi, ngày gửi và điều kiện hợp đồng."
)
