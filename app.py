import streamlit as st
from PIL import Image
import numpy as np
import torch

from image_processing import preprocess_image, resize_image
from fapnet import FAPNet, face_age_prediction

# Page config
st.set_page_config(page_title="Face Age Prediction", page_icon="🧠", layout="centered")

# Title & Description
st.title("🧠 Face Age Prediction App")
st.markdown("Upload a face image and let the model estimate the age. Works best with clear, frontal face shots.")

# Model loading (cache to avoid reloading on each run)
@st.cache_resource
def load_model(path="./models/human_age_prediction_challenger.pt"):
    return torch.load(path, map_location='cpu', weights_only=False)

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_array = np.array(image)

    st.subheader("🖼️ Original vs Enhanced Image")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        enhanced = preprocess_image(image_array)
        st.image(enhanced, caption="Enhanced Image", use_container_width=True, channels="BGR")

    st.divider()

    st.subheader("📏 Resized & Enhanced (Model Input)")
    resized = resize_image(image_array)
    enhanced_resized = preprocess_image(resized)

    col3, col4 = st.columns(2)
    with col3:
        st.image(resized, caption="Resized (200x200)", use_container_width=True)
    with col4:
        st.image(enhanced_resized, caption="Enhanced Input", use_container_width=True, channels="BGR")

    st.divider()

    st.subheader("🔮 Age Prediction")
    output = face_age_prediction(model, enhanced_resized)
    st.success(f"**Estimated Age:** {np.round(output)} years")