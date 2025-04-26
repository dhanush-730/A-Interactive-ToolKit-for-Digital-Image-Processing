import cv2
import numpy as np
from scipy.fftpack import fftshift, ifftshift

def apply_frequency_domain_operations(img):
    import streamlit as st
    st.sidebar.subheader("Frequency Domain Settings")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    f = np.fft.fft2(gray)
    fshift = fftshift(f)
    
    operation = st.sidebar.selectbox(
        "Select Frequency Operation",
        ["Low-Pass Filter", "High-Pass Filter", "Laplacian in Frequency Domain"]
    )


    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    cutoff = st.sidebar.slider("Cutoff Radius", 1, min(rows, cols) // 2, 30)

    if operation == "Low-Pass Filter":
        filter_type = st.sidebar.selectbox(
            "Filter Type", ["Ideal", "Butterworth", "Gaussian"]
        )
        n = st.sidebar.slider("Butterworth Order", 1, 10, 2) if filter_type == "Butterworth" else 0
        

        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if filter_type == "Ideal":
                    mask[i, j] = 1 if d <= cutoff else 0
                elif filter_type == "Butterworth":
                    mask[i, j] = 1 / (1 + (d / cutoff)**(2 * n))
                else:  
                    mask[i, j] = np.exp(-(d**2) / (2 * (cutoff**2)))

       
        fshift_filtered = fshift * mask
        f_ishift = ifftshift(fshift_filtered)
        processed = np.abs(np.fft.ifft2(f_ishift))

    elif operation == "High-Pass Filter":
        filter_type = st.sidebar.selectbox(
            "Filter Type", ["Ideal", "Butterworth", "Gaussian"]
        )
        n = st.sidebar.slider("Butterworth Order", 1, 10, 2) if filter_type == "Butterworth" else 0
        
        
        mask = np.ones((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if filter_type == "Ideal":
                    mask[i, j] = 0 if d <= cutoff else 1
                elif filter_type == "Butterworth":
                    mask[i, j] = 1 - (1 / (1 + (d / cutoff)**(2 * n)))
                else:  
                    mask[i, j] = 1 - np.exp(-(d**2) / (2 * (cutoff**2)))

        fshift_filtered = fshift * mask
        f_ishift = ifftshift(fshift_filtered)
        processed = np.abs(np.fft.ifft2(f_ishift))

    elif operation == "Laplacian in Frequency Domain":
        scale = st.sidebar.slider("Laplacian Scale Factor", 
                                  0.0001, 0.1, 0.01,
                                  help="Adjust this to control edge detection strength")
        
        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        
        mask = -4 * (np.pi**2) * (u**2 + v**2) * scale
        
        fshift_filtered = fshift * mask
        f_ishift = ifftshift(fshift_filtered)
        processed = np.fft.ifft2(f_ishift).real

    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return processed.astype(np.uint8)
