import numpy as np

def apply_intensity_transformations(image):
    import streamlit as st

    intensity_op = st.sidebar.selectbox(
        "Select Intensity Transformation",
        ["Negative Transformation", "Log Transform", "Gamma Correction"]
    )

    if intensity_op == "Negative Transformation":
        return 255 - image

    elif intensity_op == "Log Transform":
        c = st.sidebar.slider("Contrast Scale (c)", 1, 100, 40)
        log_image = c * np.log(1 + image.astype(np.float32))
        log_image = np.uint8(log_image / log_image.max() * 255)
        return log_image

    elif intensity_op == "Gamma Correction":
        gamma = st.sidebar.slider("Gamma Value (γ)", 0.1, 5.0, 1.0)
        gamma_corrected_image = np.uint8((image / 255.0) ** gamma * 255)
        return gamma_corrected_image

    return image
