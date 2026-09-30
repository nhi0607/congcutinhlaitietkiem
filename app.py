import streamlit as st

st.set_page_config(
    page_title="Tính lãi tiền gửi tiết kiệm",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Máy tính lãi tiền gửi tiết kiệm")
st.caption("Tính lãi đơn và lãi kép theo kỳ hạn và hình thức nhận lãi")

# =========================
# HÀM ĐỊNH DẠNG TIỀN
# =========================
def format_money(value):
    return f"{value:,.0f} VNĐ"


# =========================
# NHẬP DỮ LIỆU
# =========================
st.subheader("📋 Thông tin tiền gửi")

tien_gui = st.number_input(
    "Số tiền gửi (VNĐ)",
    min_value=0.0,
    value=100_000_000.0,
    step=1_000_000.0,
    format="%.0f"
)

ky_han = st.number_input(
    "Kỳ hạn (tháng)",
    min_value=1,
    max_value=120,
    value=12,
    step=1
)

loai_lai = st.selectbox(
    "Hình thức tính lãi",
    ["Lãi đơn", "Lãi kép"]
)

hinh_thuc_nhan = st.selectbox(
    "Hình thức nhận lãi",
    [
        "Lãnh lãi theo tháng",
        "Lãnh lãi theo quý",
        "Lãnh lãi cuối kỳ"
    ]
)

lai_suat = st.number_input(
    "Lãi suất (%/năm)",
    min_value=0.0,
    max_value=100.0,
    value=6.0,
    step=0.1,
    format="%.2f"
)

# =========================
# TÍNH TOÁN
# =========================
if st.button("🧮 Tính lãi", type="primary", use_container_width=True):

    if tien_gui <= 0:
        st.error("Vui lòng nhập số tiền gửi lớn hơn 0.")
        st.stop()

    # Lãi suất theo năm dạng thập phân
    r = lai_suat / 100

    # Số tháng
    months = ky_han

    # Số quý
    quarters = months / 3

    # -------------------------
    # LÃI ĐƠN
    # -------------------------
    if loai_lai == "Lãi đơn":

        tong_lai = tien_gui * r * (months / 12)

        if hinh_thuc_nhan == "Lãnh lãi theo tháng":
            lai_dinh_ky = tien_gui * r / 12
            so_ky = months

        elif hinh_thuc_nhan == "Lãnh lãi theo quý":
            lai_dinh_ky = tien_gui * r / 4
            so_ky = months // 3

        else:
            lai_dinh_ky = tong_lai
            so_ky = 1

        tong_tien = tien_gui + tong_lai

    # -------------------------
    # LÃI KÉP
    # -------------------------
    else:

        if hinh_thuc_nhan == "Lãnh lãi theo tháng":
            lai_suat_ky = r / 12
            so_ky = months

            tong_tien = tien_gui * (1 + lai_suat_ky) ** so_ky
            tong_lai = tong_tien - tien_gui

            lai_dinh_ky = tong_lai / so_ky

        elif hinh_thuc_nhan == "Lãnh lãi theo quý":

            # Chỉ tính lãi kép theo các quý hoàn chỉnh
            so_ky = months // 3
            lai_suat_ky = r / 4

            tong_tien = tien_gui * (1 + lai_suat_ky) ** so_ky
            tong_lai = tong_tien - tien_gui

            if so_ky > 0:
                lai_dinh_ky = tong_lai / so_ky
            else:
                lai_dinh_ky = 0

        else:
            # Lãi kép cuối kỳ:
            # Ghép lãi theo tháng
            so_ky = months
            lai_suat_ky = r / 12

            tong_tien = tien_gui * (1 + lai_suat_ky) ** so_ky
            tong_lai = tong_tien - tien_gui
            lai_dinh_ky = tong_lai

    # =========================
    # HIỂN THỊ KẾT QUẢ
    # =========================
    st.divider()
    st.subheader("📊 Kết quả")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "💵 Tiền lãi định kỳ",
            format_money(lai_dinh_ky)
        )

    with col2:
        st.metric(
            "📈 Tổng tiền lãi",
            format_money(tong_lai)
        )

    st.success(
        f"### 💰 Tổng số tiền gốc và lãi: {format_money(tong_tien)}"
    )

    # =========================
    # CHI TIẾT
    # =========================
    st.subheader("📝 Chi tiết")

    st.write(f"**Số tiền gốc:** {format_money(tien_gui)}")
    st.write(f"**Kỳ hạn:** {ky_han} tháng")
    st.write(f"**Lãi suất:** {lai_suat:.2f}%/năm")
    st.write(f"**Phương pháp:** {loai_lai}")
    st.write(f"**Hình thức nhận lãi:** {hinh_thuc_nhan}")

    if hinh_thuc_nhan == "Lãnh lãi theo tháng":
        st.info(
            f"Mỗi tháng nhận khoảng **{format_money(lai_dinh_ky)}** tiền lãi."
        )

    elif hinh_thuc_nhan == "Lãnh lãi theo quý":
        st.info(
            f"Mỗi quý nhận khoảng **{format_money(lai_dinh_ky)}** tiền lãi."
        )

    else:
        st.info(
            f"Cuối kỳ nhận tổng cộng **{format_money(tong_lai)}** tiền lãi."
        )

    # =========================
    # BẢNG TÓM TẮT
    # =========================
    st.subheader("📋 Bảng tổng hợp")

    data = {
        "Nội dung": [
            "Tiền gửi",
            "Lãi suất",
            "Kỳ hạn",
            "Phương pháp tính",
            "Hình thức nhận lãi",
            "Tổng tiền lãi",
            "Tổng gốc + lãi"
        ],
        "Giá trị": [
            format_money(tien_gui),
            f"{lai_suat:.2f}%/năm",
            f"{ky_han} tháng",
            loai_lai,
            hinh_thuc_nhan,
            format_money(tong_lai),
            format_money(tong_tien)
        ]
    }

    st.table(data)
