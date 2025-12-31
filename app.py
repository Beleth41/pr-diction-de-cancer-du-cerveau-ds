import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Configuration
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningoima', 'notumor', 'pituitary']

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model('med_ai_new.h5')

# Preprocess image
def preprocess_image(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# App
st.title("brain tumor Classification")

model = load_model()

uploaded_file = st.file_uploader("Upload X-Ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Predict"):
        preprocessed = preprocess_image(image)
        predictions = model.predict(preprocessed, verbose=0)
        predicted_class = CLASSES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        st.write("**All Probabilities:**")
        for i, cls in enumerate(CLASSES):
            st.write(f"{cls}: {predictions[0][i]*100:.2f}%")