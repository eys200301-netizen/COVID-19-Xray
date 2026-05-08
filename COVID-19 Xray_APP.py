# ============================================
# Simple Chest X-Ray Classifier
# No switches, no settings - Just upload & analyze
# ============================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page title
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🩻")
st.title("Chest X-Ray Classifier")
st.write("Upload a chest X-ray image to check if it's **Normal** or **Pneumonia**")

# Load model
@st.cache_resource
def load_model_cached():
    if os.path.exists('inceptionv3_pneumonia_model.h5'):
        return load_model('inceptionv3_pneumonia_model.h5', compile=False)
    
    else:
        st.error("❌ Model file not found!")
        return None

model = load_model_cached()

if model is None:
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, width=300)
    
    # Analyze button
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            # Preprocess
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)
            probability = float(prediction[0][0])
            
            # Show result
            st.markdown("---")
            st.subheader("📋 Result:")
            
            # Simple logic - AFTER FIX (swapped)
            if probability > 0.5:
                st.error("⚠️ **PNEUMONIA DETECTED**")
            else:
                st.success("✅ **NORMAL**")