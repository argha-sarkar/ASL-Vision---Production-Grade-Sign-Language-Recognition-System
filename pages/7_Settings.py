"""
7_Settings.py

Application Settings

Author: Argha Sarkar Project
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Application Settings")

st.markdown("---")


# ==========================================================
# Configuration
# ==========================================================

CONFIG_DIR = Path("configs")

CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONFIG_FILE = CONFIG_DIR / "settings.json"


# ==========================================================
# Default Settings
# ==========================================================

DEFAULT_SETTINGS = {
    "confidence_threshold": 0.70,
    "default_runtime": "TensorFlow",
    "image_size": 224,
    "theme": "Light",
    "show_probability_chart": True,
    "show_top5": True,
    "save_predictions": False,
    "auto_refresh": False,
    "enable_gradcam": True,
    "enable_gpu": True,
}


# ==========================================================
# Load Settings
# ==========================================================


def load_settings():

    if CONFIG_FILE.exists():

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    return DEFAULT_SETTINGS.copy()


settings = load_settings()


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Settings")

st.sidebar.info("Modify application settings.")


# ==========================================================
# Model Settings
# ==========================================================

st.header("Model Settings")

confidence = st.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    float(settings["confidence_threshold"]),
)

runtime = st.selectbox(
    "Inference Runtime",
    [
        "TensorFlow",
        "ONNX",
        "TensorFlow Lite",
        "TensorRT",
    ],
    index=[
        "TensorFlow",
        "ONNX",
        "TensorFlow Lite",
        "TensorRT",
    ].index(settings["default_runtime"]),
)

image_size = st.selectbox(
    "Input Image Size",
    [
        28,
        64,
        128,
        224,
        256,
    ],
    index=[
        28,
        64,
        128,
        224,
        256,
    ].index(settings["image_size"]),
)


# ==========================================================
# Interface
# ==========================================================

st.markdown("---")

st.header("Interface")

theme = st.radio(
    "Theme",
    [
        "Light",
        "Dark",
    ],
    index=0 if settings["theme"] == "Light" else 1,
)

show_probability = st.checkbox(
    "Show Probability Chart",
    value=settings["show_probability_chart"],
)

show_top5 = st.checkbox(
    "Show Top-5 Predictions",
    value=settings["show_top5"],
)

enable_gradcam = st.checkbox(
    "Enable Grad-CAM",
    value=settings["enable_gradcam"],
)


# ==========================================================
# Runtime
# ==========================================================

st.markdown("---")

st.header("Runtime")

enable_gpu = st.checkbox(
    "Enable GPU",
    value=settings["enable_gpu"],
)

auto_refresh = st.checkbox(
    "Auto Refresh",
    value=settings["auto_refresh"],
)

save_predictions = st.checkbox(
    "Save Predictions",
    value=settings["save_predictions"],
)


# ==========================================================
# Save
# ==========================================================

st.markdown("---")

if st.button(
    "💾 Save Settings",
    use_container_width=True,
):

    config = {
        "confidence_threshold": confidence,
        "default_runtime": runtime,
        "image_size": image_size,
        "theme": theme,
        "show_probability_chart": show_probability,
        "show_top5": show_top5,
        "save_predictions": save_predictions,
        "auto_refresh": auto_refresh,
        "enable_gradcam": enable_gradcam,
        "enable_gpu": enable_gpu,
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    st.success("Settings saved successfully.")


# ==========================================================
# Reset
# ==========================================================

if st.button(
    "🔄 Reset to Default",
    use_container_width=True,
):

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            DEFAULT_SETTINGS,
            file,
            indent=4,
        )

    st.success("Default settings restored.")


# ==========================================================
# Configuration Viewer
# ==========================================================

st.markdown("---")

st.header("Current Configuration")

st.json(load_settings())


# ==========================================================
# About
# ==========================================================

st.markdown("---")

st.header("About")

st.info(
    """
ASL Vision

Production-Grade Deep Learning Project

Features

• TensorFlow
• ONNX Runtime
• TensorFlow Lite
• TensorRT
• FastAPI
• MLflow
• Docker
• GitHub Actions
• Streamlit
• Grad-CAM
• Production Deployment
"""
)
