# Interactive Toolkit for Digital Image Processing

Welcome to the Interactive Toolkit for Digital Image Processing project!  
This project is a web-based toolkit built using Python and Streamlit to perform essential digital image processing operations.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This toolkit allows users to upload images and perform a variety of basic and advanced image processing operations like grayscale conversion, image smoothing, sharpening, noise addition, histogram equalization, and more.  
It serves as an educational platform to understand the theory and effects of common digital image processing techniques in a practical, hands-on way.

## Features

- **Grayscale Conversion**
  - Converts a colored image into shades of gray.
  - **Theory**: Removes color information and keeps only intensity. Each pixel’s RGB values are combined into a single luminance value using a weighted average (e.g., 0.299R + 0.587G + 0.114B).

- **Image Smoothing (Blurring)**
  - Reduces image noise and detail.
  - **Theory**: Applies filters like average (mean) filter or Gaussian filter to smooth out rapid changes in intensity between neighboring pixels, useful for noise reduction and softening.

- **Image Sharpening**
  - Enhances edges and fine details in the image.
  - **Theory**: Uses kernels like the Laplacian or high-pass filters. These emphasize areas where there is a sudden change in intensity, making edges and lines more prominent.

- **Noise Addition (Salt and Pepper / Gaussian Noise)**
  - Adds artificial noise to the image for testing.
  - **Theory**:
    - Salt and Pepper Noise: Random pixels are set to black or white.
    - Gaussian Noise: Adds random variations based on a Gaussian (normal) distribution to pixel intensities.
  - Useful to simulate real-world imperfect imaging.

- **Histogram Equalization**
  - Improves image contrast.
  - **Theory**: Redistributes the intensity distribution of the image. It spreads out frequent intensity values, making the histogram more uniform and enhancing contrast especially in images that are too dark or too bright.

- **Edge Detection (Sobel / Canny)**
  - Identifies boundaries within images.
  - **Theory**:
    - Sobel Operator: Calculates gradient magnitude at each pixel for both horizontal and vertical directions.
    - Canny Edge Detector: Multi-step algorithm involving smoothing, finding gradients, non-maximum suppression, and hysteresis thresholding to produce thin, clean edges.

- **Image Resizing**
  - Changes the dimensions of an image.
  - **Theory**: Interpolation methods (nearest neighbor, bilinear, bicubic) are used to estimate pixel values for the resized image. Helps in scaling up or down while maintaining quality.

- **Image Rotation**
  - Rotates the image by a specified angle.
  - **Theory**: Rotational transformation matrices are applied to shift the pixel locations appropriately, often requiring interpolation to fill in gaps.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/dhanush-730/A-Interactive-ToolKit-for-Digital-Image-Processing.git
   cd A-Interactive-ToolKit-for-Digital-Image-Processing
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/macOS
   venv\Scripts\activate      # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

   You should see something like:
   ```
   Local URL: http://localhost:8501
   Network URL: http://your-network-ip:8501
   ```

## Usage

Once the server is running, open your browser and navigate to:

```
http://localhost:8501
```

Upload an image and explore the available digital image processing operations interactively.

