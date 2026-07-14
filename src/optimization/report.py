"""
report.py

Production Optimization Report Generator

Author: Argha Sarkar Project
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


class OptimizationReport:
    """
    Generate optimization report for

    TensorFlow
    ONNX Runtime
    TensorFlow Lite
    TensorRT
    """

    def __init__(
        self,
        benchmark_file="reports/optimization/benchmark.csv",
        output_dir="reports/optimization",
    ):

        self.benchmark_file = Path(benchmark_file)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data = self.load()

    # ---------------------------------------------------------

    def load(self):

        if not self.benchmark_file.exists():

            raise FileNotFoundError(self.benchmark_file)

        return pd.read_csv(
            self.benchmark_file,
            index_col=0,
        )

    # ---------------------------------------------------------

    def best_latency(self):

        latency = self.data["Latency(ms)"].dropna()

        return (
            latency.idxmin(),
            latency.min(),
        )

    # ---------------------------------------------------------

    def best_fps(self):

        fps = self.data["FPS"].dropna()

        return (
            fps.idxmax(),
            fps.max(),
        )

    # ---------------------------------------------------------

    def smallest_model(self):

        size = self.data["Model Size(MB)"]

        return (
            size.idxmin(),
            size.min(),
        )

    # ---------------------------------------------------------

    def save_markdown(self):

        latency_runtime, latency = self.best_latency()

        fps_runtime, fps = self.best_fps()

        model_runtime, model_size = self.smallest_model()

        report = f"""# ASL Vision Optimization Report

Generated

{datetime.now()}

---

## Runtime Comparison

{self.data.to_markdown()}

---

## Best Runtime

**Runtime**

{latency_runtime}

**Average Latency**

{latency:.4f} ms

---

## Highest Throughput

**Runtime**

{fps_runtime}

**FPS**

{fps:.2f}

---

## Smallest Model

**Runtime**

{model_runtime}

**Model Size**

{model_size:.2f} MB

---

## Recommendation

"""

        if latency_runtime == fps_runtime:

            report += f"- Use **{latency_runtime}** " "for production deployment.\n"

        else:

            report += f"- Fastest runtime: **{latency_runtime}**\n"

            report += f"- Highest FPS: **{fps_runtime}**\n"

            report += f"- Smallest model: **{model_runtime}**\n"

        report += """

---

## Generated Files

- benchmark.csv
- benchmark.xlsx
- latency.png
- fps.png
- size.png
- comparison.md
- optimization_report.md

"""

        output = self.output_dir / "optimization_report.md"

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

        return output

    # ---------------------------------------------------------

    def save_excel(self):

        output = self.output_dir / "optimization_summary.xlsx"

        self.data.to_excel(
            output,
            index=True,
        )

        return output

    # ---------------------------------------------------------

    def print_summary(self):

        latency_runtime, latency = self.best_latency()

        fps_runtime, fps = self.best_fps()

        model_runtime, model_size = self.smallest_model()

        print()

        print("=" * 70)

        print("OPTIMIZATION SUMMARY")

        print("=" * 70)

        print(f"Best Runtime      : {latency_runtime}")

        print(f"Latency           : {latency:.4f} ms")

        print(f"Highest FPS       : {fps_runtime}")

        print(f"FPS               : {fps:.2f}")

        print(f"Smallest Model    : {model_runtime}")

        print(f"Model Size        : {model_size:.2f} MB")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        markdown = self.save_markdown()

        excel = self.save_excel()

        self.print_summary()

        print()

        print("Generated Reports")

        print("----------------------------")

        print(markdown)

        print(excel)


if __name__ == "__main__":

    report = OptimizationReport()

    report.run()
