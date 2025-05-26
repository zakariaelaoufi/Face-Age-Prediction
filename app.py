import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from image_processing import preprocess_image

uploaded_file = st.file_uploader(label="Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    # Method 1: Using PIL to open the image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    enhanced = preprocess_image(image_array)

    # Display with matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # First subplot
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Uploaded Image")
    axes[0].axis('off')

    # Second subplot
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title(f"Enhanced")
    axes[1].axis('off')

    plt.tight_layout()
    st.pyplot(plt)

    resized_image = cv2.resize(image_array, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    enhanced2 = preprocess_image(resized_image)

    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 5))

    # First subplot
    axes2[0].imshow(resized_image, cmap='gray')
    axes2[0].set_title(f"Resized")
    axes2[0].axis('off')

    # Second subplot
    axes2[1].imshow(enhanced2, cmap='gray')
    axes2[1].set_title(f"Enhanced")
    axes2[1].axis('off')

    plt.tight_layout()
    st.pyplot(plt)