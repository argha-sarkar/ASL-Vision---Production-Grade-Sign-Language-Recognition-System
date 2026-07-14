"""
onnx_inference.py

Production ONNX Runtime Inference

Author: Argha Sarkar Project
"""

import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXInference:
    """
    ONNX Runtime Inference Engine.
    """

    def __init__(
        self,
        model_path="exports/onnx/asl_vision.onnx",
    ):

        self.model_path = Path(model_path)

        self.session = self.load_model()

        self.input_name = self.session.get_inputs()[0].name

        self.output_name = self.session.get_outputs()[0].name

    # ---------------------------------------------------------

    def load_model(self):

        if ort is None:

            raise ImportError("Please install onnxruntime.")

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        providers = []

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:

            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        return session

    # ---------------------------------------------------------

    def preprocess(
        self,
        image,
    ):

        if isinstance(
            image,
            (str, Path),
        ):

            image = cv2.imread(str(image))

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

        return image

    # ---------------------------------------------------------

    def predict(
        self,
        image,
    ):

        image = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: image,
            },
        )

        probabilities = outputs[0][0]

        prediction = int(np.argmax(probabilities))

        confidence = float(probabilities[prediction])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
        }

    # ---------------------------------------------------------

    def benchmark(
        self,
        iterations=100,
    ):

        sample = np.random.rand(
            28,
            28,
        ).astype(np.float32)

        latencies = []

        for _ in range(iterations):

            start = time.perf_counter()

            self.predict(sample)

            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        report = {
            "average_ms": float(np.mean(latencies)),
            "minimum_ms": float(np.min(latencies)),
            "maximum_ms": float(np.max(latencies)),
            "median_ms": float(np.median(latencies)),
            "fps": float(1000 / np.mean(latencies)),
        }

        return report

    # ---------------------------------------------------------

    def information(self):

        print()

        print("=" * 70)

        print("ONNX RUNTIME")

        print("=" * 70)

        print(
            "Providers :",
            self.session.get_providers(),
        )

        print("Input Name :", self.input_name)

        print("Output Name:", self.output_name)

        print("=" * 70)

    # ---------------------------------------------------------

    def benchmark_report(
        self,
    ):

        report = self.benchmark()

        print()

        print("=" * 70)

        print("ONNX BENCHMARK")

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

    inference = ONNXInference()

    sample = np.random.randint(
        0,
        255,
        (28, 28),
        dtype=np.uint8,
    )

    inference.run(sample)
