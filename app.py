import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')  # Use the relative path since the model is in the same directory

st.title("YOLOv8 Object Detection Web App")

st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prepare the image for prediction
    img_np = np.array(image)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Perform inference
    results = model.predict(img_tensor)

    # Draw results
    annotated_img = results[0].plot()
    st.image(annotated_img, caption='Detected Objects', use_column_width=True)
