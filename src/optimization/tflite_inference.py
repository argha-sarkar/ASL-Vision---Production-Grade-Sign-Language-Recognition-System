"""
tflite_inference.py

Production TensorFlow Lite Inference Engine

Author: Argha Sarkar Project
"""

import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class TFLiteInference:
    """
    TensorFlow Lite Inference Engine.
    """

    def __init__(
        self,
        model_path="exports/tflite/asl_vision_fp32.tflite",
    ):

        self.model_path = Path(model_path)

        self.interpreter = self.load_model()

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

    # ---------------------------------------------------------

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        interpreter = tf.lite.Interpreter(model_path=str(self.model_path))

        interpreter.allocate_tensors()

        return interpreter

    # ---------------------------------------------------------

    def preprocess(
        self,
        image,
    ):

        if isinstance(
            image,
            (str, Path),
        ):

            image = cv2.imread(
                str(image),
                cv2.IMREAD_GRAYSCALE,
            )

        if image is None:

            raise ValueError("Invalid image.")

        if image.ndim == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        image = cv2.resize(
            image,
            (28, 28),
        )

        image = image.astype(np.float32)

        image /= 255.0

        image = np.expand_dims(
            image,
            axis=-1,
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        input_dtype = self.input_details[0]["dtype"]

        image = image.astype(input_dtype)

        return image

    # ---------------------------------------------------------

    def predict(
        self,
        image,
    ):

        image = self.preprocess(image)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            image,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        prediction = int(np.argmax(output))

        confidence = float(output[prediction])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": output.tolist(),
        }

    # ---------------------------------------------------------

    def benchmark(
        self,
        iterations=100,
    ):

        image = np.random.rand(
            28,
            28,
        ).astype(np.float32)

        timings = []

        for _ in range(iterations):

            start = time.perf_counter()

            self.predict(image)

            end = time.perf_counter()

            timings.append((end - start) * 1000)

        report = {
            "average_ms": float(np.mean(timings)),
            "minimum_ms": float(np.min(timings)),
            "maximum_ms": float(np.max(timings)),
            "median_ms": float(np.median(timings)),
            "fps": float(1000 / np.mean(timings)),
        }

        return report

    # ---------------------------------------------------------

    def information(self):

        print()

        print("=" * 70)

        print("TFLITE MODEL")

        print("=" * 70)

        print(
            "Input Shape :",
            self.input_details[0]["shape"],
        )

        print(
            "Output Shape:",
            self.output_details[0]["shape"],
        )

        print(
            "Input Type  :",
            self.input_details[0]["dtype"],
        )

        print(
            "Output Type :",
            self.output_details[0]["dtype"],
        )

        print("=" * 70)

    # ---------------------------------------------------------

    def benchmark_report(self):

        report = self.benchmark()

        print()

        print("=" * 70)

        print("TFLITE BENCHMARK")

        print("=" * 70)

        for key, value in report.items():

            print(f"{key:<15}: {value:.4f}")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(
        self,
        image,
    ):

        self.information()

        result = self.predict(image)

        print()

        print(result)

        self.benchmark_report()

        return result


if __name__ == "__main__":

    inference = TFLiteInference()

    sample = np.random.randint(
        0,
        255,
        (28, 28),
        dtype=np.uint8,
    )

    inference.run(sample)
