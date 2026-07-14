"""
comparison.py

Production Runtime Comparison

Compare TensorFlow, ONNX Runtime,
TensorFlow Lite and TensorRT.

Author: Argha Sarkar Project
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class RuntimeComparison:
    """
    Compare optimized runtime performance.
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

        dataframe = pd.read_csv(
            self.benchmark_file,
            index_col=0,
        )

        return dataframe

    # ---------------------------------------------------------

    def latency_plot(self):

        plt.figure(figsize=(10, 6))

        self.data["Latency(ms)"].plot(kind="bar")

        plt.ylabel("Milliseconds")

        plt.title("Inference Latency")

        plt.tight_layout()

        output = self.output_dir / "latency.png"

        plt.savefig(
            output,
            dpi=300,
        )

        plt.close()

        return output

    # ---------------------------------------------------------

    def fps_plot(self):

        plt.figure(figsize=(10, 6))

        self.data["FPS"].plot(kind="bar")

        plt.ylabel("Frames Per Second")

        plt.title("Runtime FPS")

        plt.tight_layout()

        output = self.output_dir / "fps.png"

        plt.savefig(
            output,
            dpi=300,
        )

        plt.close()

        return output

    # ---------------------------------------------------------

    def size_plot(self):

        plt.figure(figsize=(10, 6))

        self.data["Model Size(MB)"].plot(kind="bar")

        plt.ylabel("MB")

        plt.title("Model Size")

        plt.tight_layout()

        output = self.output_dir / "size.png"

        plt.savefig(
            output,
            dpi=300,
        )

        plt.close()

        return output

    # ---------------------------------------------------------

    def best_runtime(self):

        latency = self.data["Latency(ms)"].dropna()

        return latency.idxmin()

    # ---------------------------------------------------------

    def best_fps(self):

        fps = self.data["FPS"].dropna()

        return fps.idxmax()

    # ---------------------------------------------------------

    def smallest_model(self):

        size = self.data["Model Size(MB)"]

        return size.idxmin()

    # ---------------------------------------------------------

    def summary(self):

        report = []

        report.append("# Runtime Comparison\n")

        report.append(f"Best Runtime : {self.best_runtime()}\n")

        report.append(f"Highest FPS : {self.best_fps()}\n")

        report.append(f"Smallest Model : {self.smallest_model()}\n")

        report.append("\n")

        report.append(self.data.to_markdown())

        output = self.output_dir / "comparison.md"

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as file:

            file.write("\n".join(report))

        return output

    # ---------------------------------------------------------

    def print_summary(self):

        print()

        print("=" * 70)

        print("MODEL COMPARISON")

        print("=" * 70)

        print(f"Best Runtime : {self.best_runtime()}")

        print(f"Highest FPS  : {self.best_fps()}")

        print(f"Smallest Model : {self.smallest_model()}")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        self.latency_plot()

        self.fps_plot()

        self.size_plot()

        self.summary()

        self.print_summary()


if __name__ == "__main__":

    comparison = RuntimeComparison()

    comparison.run()
