"""
3_Benchmark.py

Runtime Benchmark Dashboard

Author: Argha Sarkar Project
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Benchmark",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Runtime Benchmark")

st.markdown("---")

# ---------------------------------------------------------
# Load Benchmark
# ---------------------------------------------------------

benchmark_file = Path("reports/optimization/benchmark.csv")

if not benchmark_file.exists():

    st.error(
        "Benchmark report not found.\n\n"
        "Run:\n"
        "python src/optimization/benchmark.py"
    )

    st.stop()

benchmark = pd.read_csv(
    benchmark_file,
    index_col=0,
)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

st.subheader("Summary")

col1, col2, col3 = st.columns(3)

best_latency = benchmark["Latency(ms)"].idxmin()

best_fps = benchmark["FPS"].idxmax()

smallest = benchmark["Model Size(MB)"].idxmin()

with col1:

    st.metric(
        "Fastest Runtime",
        best_latency,
        f"{benchmark.loc[best_latency,'Latency(ms)']:.2f} ms",
    )

with col2:

    st.metric(
        "Highest FPS",
        best_fps,
        f"{benchmark.loc[best_fps,'FPS']:.2f}",
    )

with col3:

    st.metric(
        "Smallest Model",
        smallest,
        f"{benchmark.loc[smallest,'Model Size(MB)']:.2f} MB",
    )

# ---------------------------------------------------------
# Table
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Benchmark Table")

st.dataframe(
    benchmark,
    use_container_width=True,
)

# ---------------------------------------------------------
# Latency
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Latency Comparison")

st.bar_chart(benchmark["Latency(ms)"])

# ---------------------------------------------------------
# FPS
# ---------------------------------------------------------

st.markdown("---")

st.subheader("FPS Comparison")

st.bar_chart(benchmark["FPS"])

# ---------------------------------------------------------
# Model Size
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Model Size Comparison")

st.bar_chart(benchmark["Model Size(MB)"])

# ---------------------------------------------------------
# Ranking
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Runtime Ranking")

ranking = benchmark.sort_values(by="Latency(ms)")

ranking = ranking.reset_index()

ranking.columns = [
    "Runtime",
    "Latency(ms)",
    "FPS",
    "Model Size(MB)",
]

ranking.insert(
    0,
    "Rank",
    range(
        1,
        len(ranking) + 1,
    ),
)

st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Recommendation")

st.success(
    f"""
Best Runtime

• Fastest Runtime : {best_latency}

• Highest FPS : {best_fps}

• Smallest Model : {smallest}
"""
)

# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

st.markdown("---")

csv = benchmark.to_csv()

st.download_button(
    "Download Benchmark CSV",
    csv,
    file_name="benchmark.csv",
    mime="text/csv",
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Benchmark")

st.sidebar.success("Runtime Performance")
