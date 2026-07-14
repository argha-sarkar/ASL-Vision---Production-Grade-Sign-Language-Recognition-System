"""
benchmark.py

Production Runtime Benchmark

Compare TensorFlow, ONNX Runtime,
TensorFlow Lite and TensorRT.

Author: Argha Sarkar Project
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class RuntimeBenchmark:

    def __init__(
        self,
        keras_model="models/best_model.keras",
        onnx_model="exports/onnx/asl_vision.onnx",
        tflite_model="exports/tflite/asl_vision_fp32.tflite",
        iterations=200,
    ):

        self.keras_model = Path(keras_model)

        self.onnx_model = Path(onnx_model)

        self.tflite_model = Path(tflite_model)

        self.iterations = iterations

        self.results = {}

    # ---------------------------------------------------------

    def sample(self):

        return np.random.rand(
            1,
            28,
            28,
            1,
        ).astype(np.float32)

    # ---------------------------------------------------------

    def benchmark_tensorflow(self):

        if not self.keras_model.exists():

            return

        model = tf.keras.models.load_model(self.keras_model)

        image = self.sample()

        model.predict(
            image,
            verbose=0,
        )

        latency = []

        for _ in range(self.iterations):

            start = time.perf_counter()

            model.predict(
                image,
                verbose=0,
            )

            end = time.perf_counter()

            latency.append((end - start) * 1000)

        self.results["TensorFlow"] = {
            "Latency(ms)": np.mean(latency),
            "FPS": 1000 / np.mean(latency),
            "Model Size(MB)": self.keras_model.stat().st_size / 1024 / 1024,
        }

    # ---------------------------------------------------------

    def benchmark_onnx(self):

        if ort is None:

            return

        if not self.onnx_model.exists():

            return

        session = ort.InferenceSession(
            str(self.onnx_model),
            providers=["CPUExecutionProvider"],
        )

        image = self.sample()

        input_name = session.get_inputs()[0].name

        output_name = session.get_outputs()[0].name

        session.run(
            [output_name],
            {input_name: image},
        )

        latency = []

        for _ in range(self.iterations):

            start = time.perf_counter()

            session.run(
                [output_name],
                {input_name: image},
            )

            end = time.perf_counter()

            latency.append((end - start) * 1000)

        self.results["ONNX Runtime"] = {
            "Latency(ms)": np.mean(latency),
            "FPS": 1000 / np.mean(latency),
            "Model Size(MB)": self.onnx_model.stat().st_size / 1024 / 1024,
        }

    # ---------------------------------------------------------

    def benchmark_tflite(self):

        if not self.tflite_model.exists():

            return

        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model))

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        output_details = interpreter.get_output_details()

        image = self.sample()

        interpreter.set_tensor(
            input_details[0]["index"],
            image,
        )

        interpreter.invoke()

        latency = []

        for _ in range(self.iterations):

            start = time.perf_counter()

            interpreter.set_tensor(
                input_details[0]["index"],
                image,
            )

            interpreter.invoke()

            interpreter.get_tensor(output_details[0]["index"])

            end = time.perf_counter()

            latency.append((end - start) * 1000)

        self.results["TensorFlow Lite"] = {
            "Latency(ms)": np.mean(latency),
            "FPS": 1000 / np.mean(latency),
            "Model Size(MB)": self.tflite_model.stat().st_size / 1024 / 1024,
        }

    # ---------------------------------------------------------

    def benchmark_tensorrt(self):

        engine = Path("exports/tensorrt/asl_vision_fp16.engine")

        if not engine.exists():

            return

        self.results["TensorRT"] = {
            "Latency(ms)": None,
            "FPS": None,
            "Model Size(MB)": engine.stat().st_size / 1024 / 1024,
        }

    # ---------------------------------------------------------

    def save(self):

        output = Path("reports/optimization")

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = pd.DataFrame(self.results).T

        dataframe.to_csv(output / "benchmark.csv")

        dataframe.to_excel(output / "benchmark.xlsx")

        return dataframe

    # ---------------------------------------------------------

    def print(self):

        dataframe = pd.DataFrame(self.results).T

        print()

        print("=" * 80)

        print("RUNTIME BENCHMARK")

        print("=" * 80)

        print(dataframe)

        print("=" * 80)

    # ---------------------------------------------------------

    def run(self):

        print()

        print("=" * 80)

        print("STARTING RUNTIME BENCHMARK")

        print("=" * 80)

        self.benchmark_tensorflow()

        self.benchmark_onnx()

        self.benchmark_tflite()

        self.benchmark_tensorrt()

        self.save()

        self.print()


if __name__ == "__main__":

    benchmark = RuntimeBenchmark()

    benchmark.run()
