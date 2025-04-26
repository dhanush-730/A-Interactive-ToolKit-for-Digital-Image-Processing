import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import io

def apply_color_processing(img):
    st.sidebar.subheader("🎨 Color Processing Parameters")
    operation = st.sidebar.selectbox(
        "Select Color Operation",
        ["Color Space Conversion", "Channel Extraction", "Color Space Visualization"]
    )

    if operation == "Color Space Conversion":
        target_space = st.sidebar.selectbox(
            "Target Color Space", 
            ["CMY", "CMYK", "HSV", "Lab", "YUV", "YCbCr"]
        )
        
        if target_space == "CMY":
            cmy = 255 - img
            return cmy.astype(np.uint8)
        
        elif target_space == "CMYK":
            cmy = 255 - img.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                k = np.min(cmy, axis=2)
                c = (cmy[..., 0] - k) / (255 - k + 1e-6) * 255
                m = (cmy[..., 1] - k) / (255 - k + 1e-6) * 255
                y = (cmy[..., 2] - k) / (255 - k + 1e-6) * 255
            cmyk = cv2.merge([c, m, y, k]).astype(np.uint8)
            return cmyk
        
        elif target_space == "HSV":
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        elif target_space == "Lab":
            return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        elif target_space == "YUV":
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
        elif target_space == "YCbCr":
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    elif operation == "Channel Extraction":
        channel = st.sidebar.selectbox(
            "Select Channel to Extract",
            ["Cyan (CMY)", "Magenta (CMY)", "Yellow (CMY)", "Black (CMYK)",
             "Hue (HSV)", "Saturation (HSV)", "Value (HSV)"]
        )
        
        if channel in ["Cyan (CMY)", "Magenta (CMY)", "Yellow (CMY)"]:
            cmy = 255 - img
            idx = ["Cyan (CMY)", "Magenta (CMY)", "Yellow (CMY)"].index(channel)
            return cmy[..., idx]
        
        elif channel == "Black (CMYK)":
            cmy = 255 - img.astype(float)
            return np.min(cmy, axis=2).astype(np.uint8)
        
        elif channel in ["Hue (HSV)", "Saturation (HSV)", "Value (HSV)"]:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            idx = ["Hue (HSV)", "Saturation (HSV)", "Value (HSV)"].index(channel)
            return hsv_img[..., idx]

    elif operation == "Color Space Visualization":
        space_type = st.sidebar.selectbox(
            "Select Color Space for 3D Visualization",
            ["RGB Cube", "HSV Cylinder"]
        )
        
        fig = None
        if space_type == "RGB Cube":
            fig = visualize_rgb_cube(img)
        elif space_type == "HSV Cylinder":
            fig = visualize_hsv_cylinder(img)
        
        if fig:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            buffer.seek(0)
            
            st.download_button(
                label=f"📥 Download {space_type} Visualization",
                data=buffer,
                file_name=f"{space_type.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
            plt.close(fig)
        
        return None

    return img

def visualize_rgb_cube(img):
    """Generate RGB Cube visualization and return the figure object."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    img_normalized = img.astype(float)/255
    r, g, b = img_normalized[...,0].flatten(), img_normalized[...,1].flatten(), img_normalized[...,2].flatten()
    
    ax.scatter(r, g, b, c=img_normalized.reshape(-1,3), marker='o', s=2, alpha=0.3)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("RGB Color Space Visualization", fontsize=14)
    st.pyplot(fig)
    return fig

def visualize_hsv_cylinder(img):
    """Generate HSV Cylinder visualization and return the figure object."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_img[...,0].flatten(), hsv_img[...,1].flatten(), hsv_img[...,2].flatten()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    h_norm = h/179.0  
    s_norm = s/255.0
    v_norm = v/255.0
    
    ax.scatter(h_norm, s_norm, v_norm, c=plt.cm.hsv(h_norm), s=2, alpha=0.3)
    ax.set_xlabel("Hue (Angle)")
    ax.set_ylabel("Saturation (Radius)")
    ax.set_zlabel("Value (Height)")
    ax.set_title("HSV Color Space Visualization", fontsize=14)
    st.pyplot(fig)
    return fig
