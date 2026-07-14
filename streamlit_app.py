"""
streamlit_app.py

Production Streamlit Dashboard

Author: Argha Sarkar Project
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ASL Vision",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------

st.title("🤟 ASL Vision")

st.markdown("---")

st.markdown(
    """
Welcome to **ASL Vision**.

A production-grade Deep Learning application for
American Sign Language Recognition.
"""
)

# ---------------------------------------------------------

st.sidebar.title("Navigation")

st.sidebar.success("Select a page.")

# ---------------------------------------------------------

st.subheader("Project Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Classes",
        "24",
    )

with col2:

    st.metric(
        "Framework",
        "TensorFlow",
    )

with col3:

    st.metric(
        "Deployment",
        "FastAPI",
    )

# ---------------------------------------------------------

st.markdown("---")

st.subheader("Features")

st.markdown(
    """
- Image Prediction
- Webcam Prediction
- Grad-CAM Visualization
- Feature Maps
- Confidence Analysis
- Misclassified Samples
- Runtime Benchmark
- Model Comparison
- Optimization Reports
- ONNX Runtime
- TensorFlow Lite
- TensorRT
"""
)

# ---------------------------------------------------------

st.markdown("---")

st.subheader("Project Structure")

st.code(
    """
pages/
components/
utils/
src/
models/
reports/
exports/
"""
)

# ---------------------------------------------------------

st.markdown("---")

st.subheader("Model Information")

model_path = Path("models/best_model.keras")

if model_path.exists():

    size = model_path.stat().st_size

    size = size / 1024 / 1024

    st.success(f"Model Found ({size:.2f} MB)")

else:

    st.error("Model not found.")

# ---------------------------------------------------------

st.markdown("---")

st.subheader("Optimization")

onnx = Path("exports/onnx/asl_vision.onnx")

tflite = Path("exports/tflite")

tensorrt = Path("exports/tensorrt")

col1, col2, col3 = st.columns(3)

with col1:

    st.write("ONNX")

    st.success("Available" if onnx.exists() else "Missing")

with col2:

    st.write("TensorFlow Lite")

    st.success("Available" if tflite.exists() else "Missing")

with col3:

    st.write("TensorRT")

    st.success("Available" if tensorrt.exists() else "Missing")

# ---------------------------------------------------------

st.markdown("---")

st.info("Select a page from the left sidebar.")
