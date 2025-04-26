import cv2

def apply_basic_operations(image):
    import streamlit as st

    basic_op = st.sidebar.selectbox(
        "Select Basic Operation",
        ["Grayscale Conversion", "Resize", "Rotation", "Color Plane Extraction"]
    )

    if basic_op == "Grayscale Conversion":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    elif basic_op == "Resize":
        scale = st.sidebar.slider("Resize Scale (%)", 10, 200, 100)
        resized_image =  cv2.resize(image, None, fx=scale / 100, fy=scale / 100)
        st.write(f"New image size: {resized_image.shape[1]} x {resized_image.shape[0]}")
        return resized_image

    elif basic_op == "Rotation":
        angle = st.sidebar.selectbox("Rotation Angle", [45, 90, 180,270,360])
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    elif basic_op == "Color Plane Extraction":
        channel = st.sidebar.selectbox("Select Channel", ["Red", "Green", "Blue"])
        idx = ["Red", "Green", "Blue"].index(channel)
        return image[:, :, idx]

    return image
