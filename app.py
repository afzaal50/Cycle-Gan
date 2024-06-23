import streamlit as st
from PIL import Image
import numpy as np
import requests
import tensorflow as tf
import os

# Ensure TensorFlow and Keras availability
try:
    tf_version = tf.__version__
    keras_available = hasattr(tf, 'keras')
except Exception as e:
    st.error(f"Error checking TensorFlow or Keras: {e}")
    st.stop()

st.text(f"TensorFlow version: {tf_version}")
st.text(f"Keras module available: {keras_available}")

# URLs of the large files in the release
url_generator_f = "https://github.com/afzaal50/Cycle-Gan/releases/download/v1.0/generator_f.2.h5"
url_generator_g = "https://github.com/afzaal50/Cycle-Gan/releases/download/v1.0/generator_g.1.h5"

def download_file(url, local_filename):
    # Download the file from `url` and save it locally under `local_filename`
    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTPError: {e}")
            st.error(f"Status code: {r.status_code}")
            st.error(f"Response text: {r.text}")
            raise
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Download the large files if they don't already exist locally
if not os.path.exists("generator_f.2.h5"):
    st.text("Downloading generator_f.2.h5...")
    download_file(url_generator_f, "generator_f.2.h5")

if not os.path.exists("generator_g.1.h5"):
    st.text("Downloading generator_g.1.h5...")
    download_file(url_generator_g, "generator_g.1.h5")

# Verify that the files exist
if os.path.exists("generator_f.2.h5") and os.path.exists("generator_g.1.h5"):
    st.text("Model files downloaded successfully.")
else:
    st.error("Model files were not downloaded successfully. Please check the URLs and try again.")
    st.stop()

# Load the models
try:
    generator_f = tf.keras.models.load_model("generator_f.2.h5")
    generator_g = tf.keras.models.load_model("generator_g.1.h5")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Function to process images with CycleGAN
def process_with_cyclegan(image, generator):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    processed_image = generator.predict(image)  # Run the generator
    processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
    return processed_image

st.title("CycleGAN implementation for Image to Image Translation")

st.header("Upload Ultrasound Image")
ultrasound_file = st.file_uploader("Drag and drop file here", type=["png", "jpg", "jpeg"], key="ultrasound")
if ultrasound_file:
    ultrasound_image = Image.open(ultrasound_file)
    ultrasound_image = np.array(ultrasound_image.resize((256, 256)))  # Resize for CycleGAN
    st.image(ultrasound_image, caption="Uploaded Ultrasound Image", use_column_width=True)

st.header("Upload Chicken Image")
chicken_file = st.file_uploader("Drag and drop file here", type=["png", "jpg", "jpeg"], key="chicken")
if chicken_file:
    chicken_image = Image.open(chicken_file)
    chicken_image = np.array(chicken_image.resize((256, 256)))  # Resize for CycleGAN
    st.image(chicken_image, caption="Uploaded Chicken Image", use_column_width=True)

if ultrasound_file and chicken_file:
    st.subheader("Processed Result Image")
    
    # Process images with CycleGAN
    processed_image = process_with_cyclegan(ultrasound_image, generator_g)  # Assuming ultrasound to chicken transformation
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert to uint8 for display
    
    st.image(processed_image, caption="Processed Result Image", use_column_width=True)
