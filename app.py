import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST format
    image = np.array(image)

    # Invert colors if the background is black
    if np.mean(image) < 128:
        image = 255 - image
    
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # CNN input shape
    return image

# Streamlit UI
st.title("📝 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image, and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Show prediction result
    st.success(f"**Predicted Digit:** {predicted_digit}")
    st.write("Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"Digit {i}: {score:.4f}")
