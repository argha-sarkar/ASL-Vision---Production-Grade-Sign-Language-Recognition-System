"""
5_GradCAM.py

Grad-CAM Visualization Dashboard

Author: Argha Sarkar Project
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.api.predictor import APIPredictor
from src.explainability.gradcam import GradCAM

st.set_page_config(
    page_title="Grad-CAM",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Grad-CAM Visualization")

st.markdown("---")


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------


@st.cache_resource
def load_objects():

    predictor = APIPredictor()

    gradcam = GradCAM(predictor.model)

    return predictor, gradcam


predictor, gradcam = load_objects()


# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    result = predictor.predict(image_np)

    prediction = result["prediction"]

    confidence = result["confidence"]

    temp_dir = Path("reports/explainability")

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = temp_dir / "gradcam_streamlit.png"

    gradcam.save(
        image=image_np,
        filename=str(output_file.absolute()),
    )

    heatmap = cv2.imread(str(output_file))

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_column_width=True,
        )

    with col2:

        st.subheader("Grad-CAM")

        st.image(
            heatmap,
            use_column_width=True,
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction",
            prediction,
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%",
        )

    st.markdown("---")

    st.subheader("Prediction Probabilities")

    probabilities = np.array(result["probabilities"])

    st.bar_chart(probabilities)

    st.markdown("---")

    with open(
        output_file,
        "rb",
    ) as file:

        st.download_button(
            label="Download Grad-CAM",
            data=file,
            file_name="gradcam.png",
            mime="image/png",
        )

else:

    st.info("Upload an image to visualize Grad-CAM.")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Grad-CAM")

st.sidebar.info(
    """
Grad-CAM highlights the regions
used by the CNN to make its prediction.

Brighter regions contribute more
to the predicted class.
"""
)
