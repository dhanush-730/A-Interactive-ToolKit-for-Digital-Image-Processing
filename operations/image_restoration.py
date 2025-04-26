import cv2
import numpy as np
from numba import njit, prange  

def apply_image_restoration(img):
    import streamlit as st
    
    st.sidebar.subheader("Restoration Parameters")
    operation = st.sidebar.selectbox(
        "Select Restoration Operation",
        [
            "Add Noise", 
            "Noise Removal (Mean Filter)",
            "Noise Removal (Median Filter)",
            "Adaptive Median Filter"
        ]
    )

   
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if operation == "Add Noise":
        noise_type = st.sidebar.selectbox("Noise Type", ["Salt & Pepper", "Gaussian"])
        amount = st.sidebar.slider("Noise Amount", 0.0, 1.0, 0.05)
        
        noisy = img.copy()
        if noise_type == "Salt & Pepper":
            num_pixels = int(amount * img.size)
            
            
            coords = [np.random.randint(0, i-1, num_pixels) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 255
            
            
            coords = [np.random.randint(0, i-1, num_pixels) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 0
            return noisy

        elif noise_type == "Gaussian":
            var = st.sidebar.slider("Variance", 0.0, 1.0, 0.01)
            noise = np.random.normal(0, var**0.5, img.shape) * 255
            noisy = img + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)

    elif operation == "Noise Removal (Mean Filter)":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 3, step=2)
        kernel_size = max(3, min(31, kernel_size))
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1
        return cv2.blur(img, (kernel_size, kernel_size))

    elif operation == "Noise Removal (Median Filter)":
        kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 3, step=2)
        kernel_size = max(3, min(31, kernel_size))
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size - 1
        return cv2.medianBlur(img, kernel_size)

    elif operation == "Adaptive Median Filter":
        max_window = st.sidebar.slider("Max Window Size", 3, 31, 7, step=2)
        return optimized_adaptive_median_filter(img, max_window)

    return img

@njit(parallel=True)
def optimized_adaptive_median_filter(img, max_window):
    filtered = np.empty_like(img)
    rows, cols, ch = img.shape
    
    for c in prange(ch):
        for y in prange(rows):
            for x in prange(cols):
                filtered[y, x, c] = fast_adaptive_pixel(
                    img[y, x, c], 
                    img[:, :, c], 
                    x, y, 
                    max_window
                )
    return filtered

@njit
def fast_adaptive_pixel(original_val, channel, x, y, max_size):
    window_size = 3
    rows, cols = channel.shape
    
    while window_size <= max_size:
        half = window_size // 2  
        
       
        y_start = max(0, y - half)
        y_end = min(rows, y + half + 1)
        x_start = max(0, x - half)
        x_end = min(cols, x + half + 1)
        
        window = channel[y_start:y_end, x_start:x_end]
        if window.size == 0:
            return original_val
        
        z_min = np.min(window)
        z_med = np.median(window)
        z_max = np.max(window)
        
        if z_min < z_med < z_max:
            if z_min < original_val < z_max:
                return original_val
            else:
                return z_med
        else:
            window_size += 2  
            
    return z_med

