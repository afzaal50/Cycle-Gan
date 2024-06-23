import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the CycleGAN model weights
generator_g = tf.keras.models.load_model("C:\Users\muham\Downloads\Streamlit App\generator_g (1).h5")
generator_f = tf.keras.models.load_model("C:\Users\muham\Downloads\Streamlit App\generator_f (2).h5")

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
