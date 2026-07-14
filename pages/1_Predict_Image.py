"""
1_Predict_Image.py

ASL Vision Image Prediction

Author: Argha Sarkar Project
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.api.predictor import APIPredictor

st.set_page_config(
    page_title="Image Prediction",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ ASL Image Prediction")

st.markdown("---")

# ---------------------------------------------------------
# Load Predictor
# ---------------------------------------------------------


@st.cache_resource
def load_predictor():

    return APIPredictor()


predictor = load_predictor()

# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an ASL Image",
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

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image_np = np.array(image)

    col1, col2 = st.columns(2)

    # -----------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_column_width=True,
        )

    # -----------------------------------------------------

    with col2:

        with st.spinner("Predicting..."):

            result = predictor.predict(image_np)

        st.subheader("Prediction")

        st.success(result["prediction"])

        st.metric(
            "Confidence",
            f"{result['confidence']*100:.2f}%",
        )

        probabilities = np.array(result["probabilities"])

        dataframe = pd.DataFrame(
            {
                "Class": np.arange(len(probabilities)),
                "Probability": probabilities,
            }
        )

        dataframe = dataframe.sort_values(
            "Probability",
            ascending=False,
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Probability Chart
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Probability Distribution")

    chart = dataframe.set_index("Class")

    st.bar_chart(chart)

    # -----------------------------------------------------
    # Top 5
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("Top 5 Predictions")

    top5 = dataframe.head(5)

    st.table(top5)

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    st.markdown("---")

    csv = top5.to_csv(index=False)

    st.download_button(
        label="Download Prediction",
        data=csv,
        file_name="prediction.csv",
        mime="text/csv",
    )

else:

    st.info("Please upload an image.")

# ---------------------------------------------------------

st.sidebar.success("Image Prediction")
