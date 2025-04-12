import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained MNIST model
model = load_model('digit_recognition_model.h5')

# App title
st.title("Handwritten Digit Recognition")

# File uploader widget
uploaded_file = st.file_uploader("Upload a digit image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match MNIST dimensions
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for prediction
    image_array = np.array(image) / 255.0      # Normalize the pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model
    
    # Predict and display the result
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    st.write(f"Predicted Digit: **{predicted_digit}**")
