import streamlit as st
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')  # Make sure 'best.pt' is in the same directory or provide the correct path

st.title("YOLOv8 Object Detection Web App")

st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and resize the image to a standard size (e.g., 640x640 pixels)
    image = Image.open(uploaded_file)
    standard_size = (640, 640)
    resized_image = image.resize(standard_size, Image.Resampling.LANCZOS)
    
    # Convert resized image to numpy array for model input
    img_np = np.array(resized_image)
    
    # Perform inference
    results = model.predict(img_np)
    
    # Draw results (the ultralytics library handles plotting)
    annotated_img = results[0].plot()  # Results are annotated by YOLOv8
    
    # Display the resized image and detection results
    st.image(resized_image, caption='Resized Image', use_column_width=True)
    st.image(annotated_img, caption='Detected Objects', use_column_width=True)
