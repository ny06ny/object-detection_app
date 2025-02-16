import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os

st.title("XỬ LÝ ĐỐI TƯỢNG TRONG ẢNH")

# Tạo tùy chọn tải file hoặc nhập URL
option = st.radio("Chọn cách nhập hình ảnh:", ("Tải file lên", "Nhập URL"))

image = None

if option == "Tải file lên":
    uploaded_file = st.file_uploader("Tải lên ảnh của bạn", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Nhập URL":
    url = st.text_input("Nhập URL ảnh:")
    if url: 
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Không thể tải ảnh từ URL. Lỗi: {e}")

if image:
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    # Load mô hình YOLOv8
    model_path = r"C:\Users\laptop\.vscode\runs\detect\train2\weights\best.pt"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Không tìm thấy mô hình tại: {model_path}")
    else:
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"⚠️ Không thể tải mô hình YOLO. Lỗi: {e}")
        else:
            # Chuyển ảnh sang mảng NumPy
            image_np = np.array(image)

            # Thực hiện nhận diện đối tượng
            with st.spinner("🔍 Đang nhận diện đối tượng..."):
                results = model(image_np, conf=0.5)

            # Kiểm tra kết quả và hiển thị ảnh đã nhận diện   
            if results and results[0].boxes is not None:
                annotated_image = results[0].plot()
                st.image(annotated_image, caption="Ảnh đã nhận diện", use_container_width=True)
            else:
                st.warning("⚠️ Không phát hiện đối tượng nào trong ảnh.")
                
            # Nhập tên file trước khi tải xuống
            file_name = st.text_input("Nhập tên file để tải xuống:", "detected_image.png")

            # Chuyển ảnh sang định dạng có thể tải về
            img_pil = Image.fromarray(annotated_image)
            img_buffer = BytesIO()
            img_pil.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # Tạo nút tải ảnh
            st.download_button(
                label="📥 Tải ảnh xuống",
                data=img_bytes,
                file_name=file_name if file_name else "detected_image.png",
                mime="image/png"
            )
