import cv2
import numpy as np
from skimage.segmentation import flood_fill

def apply_segmentation(img):
    import streamlit as st

    st.sidebar.subheader("Segmentation Parameters")
    operation = st.sidebar.selectbox(
        "Select Segmentation Method",
        [
            "Edge Detection", 
            "Thresholding", 
            "Region Growing",
            "Hough Transform"
        ]
    )

   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if operation == "Edge Detection":
        edge_type = st.sidebar.selectbox(
            "Edge Detection Method",
            ["Sobel", "Prewitt", "Laplacian", "Canny"]
        )
        
        if edge_type == "Sobel":
            
            ksize = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
            ksize = max(1, min(31, ksize))
            ksize = ksize if ksize % 2 != 0 else ksize - 1

            
            dx = st.sidebar.slider("X Derivative", 0, 1, 1)
            dy = st.sidebar.slider("Y Derivative", 0, 1, 1)
            
            
            if dx == 0 and dy == 0:
                st.sidebar.warning("At least one derivative must be >0. Using X-direction.")
                dx = 1

            edges = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
            return cv2.convertScaleAbs(edges)
            
        elif edge_type == "Prewitt":
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            return cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
            
        elif edge_type == "Laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            return cv2.convertScaleAbs(edges)
            
        elif edge_type == "Canny":
            low = st.sidebar.slider("Low Threshold", 0, 255, 50)
            high = st.sidebar.slider("High Threshold", 0, 255, 150)
            return cv2.Canny(gray, low, high)

    elif operation == "Thresholding":
        thresh_type = st.sidebar.selectbox(
            "Thresholding Method",
            ["Global", "Otsu's"]
        )
        
        if thresh_type == "Global":
            threshold = st.sidebar.slider("Threshold Value", 0, 255, 127)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
        elif thresh_type == "Otsu's":
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)

    elif operation == "Region Growing":
        seed = (
            st.sidebar.slider("Seed X", 0, gray.shape[1]-1, gray.shape[1]//2),
            st.sidebar.slider("Seed Y", 0, gray.shape[0]-1, gray.shape[0]//2)
        )
        tolerance = st.sidebar.slider("Tolerance", 1, 100, 20)
        
        
        segmented = flood_fill(
            gray, 
            seed_point=(seed[1], seed[0]),  
            new_value=255,
            tolerance=tolerance
        )
        return cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)


    elif operation == "Hough Transform":
        hough_type = st.sidebar.selectbox(
            "Hough Transform Type",
            ["Lines", "Circles"]
        )
        
        if hough_type == "Lines":
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=st.sidebar.slider("Threshold", 1, 200, 100)
            )
            
            output = img.copy()
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(output, (x1,y1), (x2,y2), (0,0,255), 2)
            return output
            
        elif hough_type == "Circles":
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=st.sidebar.slider("DP", 1.0, 2.0, 1.2),
                minDist=st.sidebar.slider("Min Distance", 1, 100, 20),
                param1=50,
                param2=st.sidebar.slider("Accumulator Threshold", 1, 100, 30),
                minRadius=0,
                maxRadius=0
            )
            
            output = img.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv2.circle(output, (i[0],i[1]), i[2], (0,255,0), 2)
                    cv2.circle(output, (i[0],i[1]), 2, (0,0,255), 3)
            return output

    return img
