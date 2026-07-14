"""
4_Model_Comparison.py

Model Comparison Dashboard

Author: Argha Sarkar Project
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Comparison Dashboard")

st.markdown("---")

# ---------------------------------------------------------
# Load Benchmark
# ---------------------------------------------------------

benchmark_path = Path("reports/optimization/benchmark.csv")

if not benchmark_path.exists():

    st.error("Benchmark file not found.\n" "Run optimization benchmark first.")

    st.stop()

benchmark = pd.read_csv(
    benchmark_path,
    index_col=0,
)

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

st.subheader("Overview")

st.dataframe(
    benchmark,
    use_container_width=True,
)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

latency = benchmark["Latency(ms)"]

fps = benchmark["FPS"]

size = benchmark["Model Size(MB)"]

best_latency = latency.idxmin()

best_fps = fps.idxmax()

smallest_model = size.idxmin()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Fastest Runtime",
        best_latency,
        f"{latency.min():.3f} ms",
    )

with col2:

    st.metric(
        "Highest FPS",
        best_fps,
        f"{fps.max():.2f}",
    )

with col3:

    st.metric(
        "Smallest Model",
        smallest_model,
        f"{size.min():.2f} MB",
    )

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "Latency",
        "FPS",
        "Model Size",
    ]
)

with tab1:

    st.bar_chart(latency)

with tab2:

    st.bar_chart(fps)

with tab3:

    st.bar_chart(size)

# ---------------------------------------------------------
# Ranking
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Runtime Ranking")

ranking = benchmark.copy()

ranking["Latency Rank"] = ranking["Latency(ms)"].rank(ascending=True).astype(int)

ranking["FPS Rank"] = ranking["FPS"].rank(ascending=False).astype(int)

ranking["Size Rank"] = ranking["Model Size(MB)"].rank(ascending=True).astype(int)

ranking["Overall Score"] = (
    ranking["Latency Rank"] + ranking["FPS Rank"] + ranking["Size Rank"]
)

ranking = ranking.sort_values("Overall Score")

st.dataframe(
    ranking,
    use_container_width=True,
)

# ---------------------------------------------------------
# Best Runtime
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Best Runtime")

winner = ranking.index[0]

st.success(
    f"""
🏆 Recommended Runtime

Runtime : {winner}

Latency : {benchmark.loc[winner,'Latency(ms)']:.3f} ms

FPS : {benchmark.loc[winner,'FPS']:.2f}

Model Size : {benchmark.loc[winner,'Model Size(MB)']:.2f} MB
"""
)

# ---------------------------------------------------------
# Runtime Details
# ---------------------------------------------------------

st.markdown("---")

runtime = st.selectbox(
    "Select Runtime",
    benchmark.index.tolist(),
)

st.subheader(runtime)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Latency",
        f"{benchmark.loc[runtime,'Latency(ms)']:.3f} ms",
    )

with c2:

    st.metric(
        "FPS",
        f"{benchmark.loc[runtime,'FPS']:.2f}",
    )

with c3:

    st.metric(
        "Model Size",
        f"{benchmark.loc[runtime,'Model Size(MB)']:.2f} MB",
    )

# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

st.markdown("---")

st.download_button(
    "Download Benchmark CSV",
    benchmark.to_csv(),
    file_name="runtime_comparison.csv",
    mime="text/csv",
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Model Comparison")

st.sidebar.info(
    """
Compare

• TensorFlow

• ONNX Runtime

• TensorFlow Lite

• TensorRT
"""
)
