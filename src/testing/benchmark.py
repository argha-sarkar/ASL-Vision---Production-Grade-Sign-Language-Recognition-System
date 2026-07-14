"""
benchmark.py

Production Model Benchmark

Author: Argha Sarkar Project
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


class Benchmark:
    """
    Benchmark trained TensorFlow models.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        input_shape=(28, 28, 1),
        iterations=100,
    ):

        self.model_path = Path(model_path)

        self.input_shape = input_shape

        self.iterations = iterations

        self.model = self.load_model()

    # ---------------------------------------------------------

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(f"{self.model_path} not found.")

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def dummy_input(self):

        return np.random.rand(
            1,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ).astype(np.float32)

    # ---------------------------------------------------------

    def warmup(self):

        image = self.dummy_input()

        for _ in range(10):

            self.model.predict(
                image,
                verbose=0,
            )

    # ---------------------------------------------------------

    def benchmark_prediction(self):

        image = self.dummy_input()

        times = []

        for _ in range(self.iterations):

            start = time.perf_counter()

            self.model.predict(
                image,
                verbose=0,
            )

            end = time.perf_counter()

            times.append((end - start) * 1000)

        return times

    # ---------------------------------------------------------

    def benchmark_batch(
        self,
        batch_size=32,
    ):

        images = np.random.rand(
            batch_size,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
        ).astype(np.float32)

        start = time.perf_counter()

        self.model.predict(
            images,
            verbose=0,
        )

        end = time.perf_counter()

        return (end - start) * 1000

    # ---------------------------------------------------------

    def memory_usage(self):

        parameters = self.model.count_params()

        size_mb = (parameters * 4) / (1024**2)

        return size_mb

    # ---------------------------------------------------------

    def throughput(
        self,
        batch_size=32,
    ):

        elapsed = self.benchmark_batch(batch_size)

        return batch_size / (elapsed / 1000)

    # ---------------------------------------------------------

    def generate_report(self):

        self.warmup()

        prediction_times = self.benchmark_prediction()

        batch_time = self.benchmark_batch()

        report = {
            "Average Latency (ms)": np.mean(prediction_times),
            "Minimum Latency (ms)": np.min(prediction_times),
            "Maximum Latency (ms)": np.max(prediction_times),
            "Median Latency (ms)": np.median(prediction_times),
            "Std Dev (ms)": np.std(prediction_times),
            "Batch Time (32 Images)": batch_time,
            "Throughput (Images/sec)": self.throughput(),
            "Parameters": self.model.count_params(),
            "Estimated Model Size (MB)": self.memory_usage(),
        }

        return report

    # ---------------------------------------------------------

    def save_report(
        self,
        output="reports/testing/benchmark.csv",
    ):

        report = self.generate_report()

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = pd.DataFrame(
            report.items(),
            columns=[
                "Metric",
                "Value",
            ],
        )

        dataframe.to_csv(
            output,
            index=False,
        )

        return output

    # ---------------------------------------------------------

    def print_report(self):

        report = self.generate_report()

        print()

        print("=" * 70)

        print("MODEL BENCHMARK")

        print("=" * 70)

        for key, value in report.items():

            if isinstance(
                value,
                float,
            ):

                print(f"{key:<35}: {value:.4f}")

            else:

                print(f"{key:<35}: {value}")

        print("=" * 70)


if __name__ == "__main__":

    benchmark = Benchmark()

    benchmark.print_report()

    benchmark.save_report()
