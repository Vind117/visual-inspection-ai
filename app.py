import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from detector import DefectDetector

st.set_page_config(page_title="Vision Inspection AI", page_icon="🔍", layout="wide")

st.title("🔍 Vision Inspection AI — Quality Control Dashboard")
st.markdown("Automated industrial defect detection, benchmark performance, and log inspection metrics.")

# Sidebar Configuration
st.sidebar.header("Model Settings")
model_path = "runs/detect/runs/train/industrial_defect_model/weights/best.pt"

if not os.path.exists(model_path):
    # Fallback path check
    model_path = "runs/train/industrial_defect_model/weights/best.pt"

if not os.path.exists(model_path):
    st.sidebar.warning("Trained model not found. Defaulting to pre-trained YOLOv8n.")
    model_path = "yolov8n.pt"

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

@st.cache_resource
def load_detector(path):
    return DefectDetector(path)

detector = load_detector(model_path)

# File Uploader
uploaded_file = st.file_uploader("Upload product inspection image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Product Image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        annotated_img, detections, latency = detector.predict(image_np, conf_threshold)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("AI Inspection Output")
        st.image(annotated_img_rgb, use_container_width=True)

    # Metrics Row
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Inference Latency", f"{latency:.2f} ms")
    m2.metric("Defects Detected", len(detections))
    m3.metric("Confidence Threshold", f"{conf_threshold:.2f}")

    # Results Table
    if detections:
        st.subheader("📋 Detected Defect Log")
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Inspection Report (CSV)",
            data=csv,
            file_name='defect_report.csv',
            mime='text/csv'
        )
    else:
        st.success("✨ Product passed inspection! No defects detected above threshold.")
else:
    st.info("👆 Upload an image from `dataset/images/val/` to test your trained inspection model!")