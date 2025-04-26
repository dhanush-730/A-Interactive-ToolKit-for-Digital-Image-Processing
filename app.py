import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

from operations.basic_operations import apply_basic_operations
from operations.intensity_transformations import apply_intensity_transformations
from operations.spatial_filtering import apply_spatial_filtering
from operations.frequency_domain import apply_frequency_domain_operations
from operations.image_restoration import apply_image_restoration
from operations.segmentation import apply_segmentation
from operations.color_processing import apply_color_processing

st.set_page_config(page_title="Image Processing Lab", layout="wide")
st.title("📸 Image Processing Laboratory")

operation_type = st.sidebar.selectbox(
    "Select Operation Category",
    [
        "Basic Operations",
        "Intensity Transformations",
        "Spatial Filtering",
        "Frequency Domain",
        "Image Restoration",
        "Segmentation",
        "Color Processing"
    ]
)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "bmp"])
if uploaded_file:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    col1, col2 = st.columns(2)
    col1.image(img_array, caption="Original Image", use_container_width=True)

    processed_img = None

    try:
        if operation_type == "Basic Operations":
            processed_img = apply_basic_operations(img_array)
        elif operation_type == "Intensity Transformations":
            processed_img = apply_intensity_transformations(img_array)
        elif operation_type == "Spatial Filtering":
            processed_img = apply_spatial_filtering(img_array)
        elif operation_type == "Frequency Domain":
            processed_img = apply_frequency_domain_operations(img_array)
        elif operation_type == "Image Restoration":
            processed_img = apply_image_restoration(img_array)
        elif operation_type == "Segmentation":
            processed_img = apply_segmentation(img_array)
        elif operation_type == "Color Processing":
            processed_img = apply_color_processing(img_array)
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

    if processed_img is not None:
        if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            display_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        else:
            display_img = processed_img

        col2.image(
            display_img,
            caption="Processed Image",
            use_container_width=True,
            channels="BGR" if len(display_img.shape) == 3 else "GRAY"
        )

        st.balloons()

        st.sidebar.subheader("Download Processed Image")
        buffer = io.BytesIO()
        
        if len(processed_img.shape) == 2:
            pil_img = Image.fromarray(processed_img).convert("L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.sidebar.download_button(
            label="⬇️ Download Processed Image",
            data=buffer,
            file_name=f"processed_{operation_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )

else:
    st.warning("Please upload an image to start processing!")
