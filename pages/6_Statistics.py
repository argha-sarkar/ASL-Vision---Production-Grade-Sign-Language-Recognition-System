"""
6_Statistics.py

Dataset & Model Statistics Dashboard

Author: Argha Sarkar Project
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Statistics",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Statistics Dashboard")

st.markdown("---")

# ==========================================================
# Dataset Statistics
# ==========================================================

st.header("Dataset Statistics")

train_csv = Path("data/raw/sign_mnist_train.csv")

test_csv = Path("data/raw/sign_mnist_test.csv")

if train_csv.exists() and test_csv.exists():

    train = pd.read_csv(train_csv)

    test = pd.read_csv(test_csv)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Training Samples",
            len(train),
        )

    with col2:

        st.metric(
            "Testing Samples",
            len(test),
        )

    with col3:

        st.metric(
            "Classes",
            train["label"].nunique(),
        )

    st.markdown("---")

    st.subheader("Dataset Shape")

    shape = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Testing",
            ],
            "Rows": [
                train.shape[0],
                test.shape[0],
            ],
            "Columns": [
                train.shape[1],
                test.shape[1],
            ],
        }
    )

    st.dataframe(
        shape,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader("Class Distribution")

    distribution = train["label"].value_counts()

    st.bar_chart(distribution)

    st.markdown("---")

    st.subheader("Pixel Statistics")

    pixels = train.drop(columns=["label"])

    statistics = pd.DataFrame(
        {
            "Metric": [
                "Mean",
                "Standard Deviation",
                "Minimum",
                "Maximum",
            ],
            "Value": [
                pixels.values.mean(),
                pixels.values.std(),
                pixels.values.min(),
                pixels.values.max(),
            ],
        }
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.warning("Dataset not found.")

# ==========================================================
# Model Statistics
# ==========================================================

st.markdown("---")

st.header("Model Statistics")

model_path = Path("models/best_model.keras")

if model_path.exists():

    import keras

    model = keras.models.load_model(
        str(model_path)
    )

    params = model.count_params()

    trainable = int(sum(
        w.numpy().size
        for w in model.trainable_weights
    ))

    non_trainable = int(sum(
        w.numpy().size
        for w in model.non_trainable_weights
    ))

    size = model_path.stat().st_size / 1024 / 1024

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Parameters",
            f"{params:,}",
        )

    with c2:

        st.metric(
            "Trainable",
            f"{trainable:,}",
        )

    with c3:

        st.metric(
            "Non-Trainable",
            f"{non_trainable:,}",
        )

    with c4:

        st.metric(
            "Model Size",
            f"{size:.2f} MB",
        )

    st.markdown("---")

    st.subheader("Model Summary")

    summary = []

    model.summary(print_fn=lambda x: summary.append(x))

    st.code("\n".join(summary))

else:

    st.warning("Model not found.")

# ==========================================================
# Runtime Statistics
# ==========================================================

st.markdown("---")

st.header("Runtime Statistics")

benchmark = Path("reports/optimization/benchmark.csv")

if benchmark.exists():

    benchmark_df = pd.read_csv(
        benchmark,
        index_col=0,
    )

    st.dataframe(
        benchmark_df,
        use_container_width=True,
    )

    st.bar_chart(benchmark_df["Latency(ms)"])

    st.bar_chart(benchmark_df["FPS"])

    st.bar_chart(benchmark_df["Model Size(MB)"])

else:

    st.info("Run optimization benchmark first.")

# ==========================================================
# TensorFlow Information
# ==========================================================

st.markdown("---")

st.header("Environment")

try:

    import tensorflow as tf

    gpu = tf.config.list_physical_devices("GPU")

    info = pd.DataFrame(
        {
            "Property": [
                "TensorFlow Version",
                "GPU Available",
                "GPU Count",
            ],
            "Value": [
                tf.__version__,
                len(gpu) > 0,
                len(gpu),
            ],
        }
    )

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True,
    )

except Exception:

    st.warning("Unable to retrieve TensorFlow information.")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Statistics")

st.sidebar.success("Dataset • Model • Runtime")
