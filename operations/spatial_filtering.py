import cv2
import numpy as np

def apply_spatial_filtering(image):
    import streamlit as st
    
    st.sidebar.subheader("Spatial Filtering Parameters")
    filter_type = st.sidebar.selectbox(
        "Select Filter Type",
        ["Median Filter", "Mean Filter", "Laplacian Sharpening", "High Boost Filtering"]
    )

    
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    if filter_type == "Median Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        kernel_size = max(1, min(31, kernel_size))
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1
        return cv2.medianBlur(image, kernel_size)

    elif filter_type == "Mean Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 3, step=2)
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1
        return cv2.blur(image, (kernel_size, kernel_size))

    elif filter_type == "Laplacian Sharpening":
        image_float = image.astype(np.float32)
        laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
        sharpened = cv2.addWeighted(image_float, 1.5, laplacian, -0.5, 0)
        return cv2.convertScaleAbs(sharpened)

    elif filter_type == "High Boost Filtering":
        boost_factor = st.sidebar.slider("Boost Factor", 1.0, 3.0, 1.5)
        blurred = cv2.GaussianBlur(image, (3,3), 0).astype(np.float32)
        mask = image.astype(np.float32) - blurred
        boosted = image.astype(np.float32) + boost_factor * mask
        return cv2.convertScaleAbs(boosted)

    return image
