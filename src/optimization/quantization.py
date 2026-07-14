"""
quantization.py

Production TensorFlow Model Quantization

Author: Argha Sarkar Project
"""

from pathlib import Path

import numpy as np
import tensorflow as tf


class ModelQuantizer:
    """
    TensorFlow Lite Quantization Pipeline.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        output_dir="exports/quantized",
    ):

        self.model_path = Path(model_path)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = self.load_model()

    # ---------------------------------------------------------

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def representative_dataset(self):

        for _ in range(100):

            sample = np.random.rand(
                1,
                28,
                28,
                1,
            ).astype(np.float32)

            yield [sample]

    # ---------------------------------------------------------

    def export_dynamic_range(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        model = converter.convert()

        output = self.output_dir / "dynamic_range.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(model)

        return output

    # ---------------------------------------------------------

    def export_float16(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.target_spec.supported_types = [tf.float16]

        model = converter.convert()

        output = self.output_dir / "float16.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(model)

        return output

    # ---------------------------------------------------------

    def export_int8(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.representative_dataset = self.representative_dataset

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        converter.inference_input_type = tf.uint8

        converter.inference_output_type = tf.uint8

        model = converter.convert()

        output = self.output_dir / "int8.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(model)

        return output

    # ---------------------------------------------------------

    def benchmark_size(self):

        print()

        print("=" * 70)

        print("MODEL SIZE COMPARISON")

        print("=" * 70)

        original = self.model_path.stat().st_size / 1024 / 1024

        print(f"{'Original':<25}" f"{original:.2f} MB")

        for file in sorted(self.output_dir.glob("*.tflite")):

            size = file.stat().st_size / 1024 / 1024

            reduction = (1 - size / original) * 100

            print(f"{file.name:<25}" f"{size:.2f} MB" f"   ({reduction:.2f}% smaller)")

        print("=" * 70)

    # ---------------------------------------------------------

    def verify(self):

        print()

        print("=" * 70)

        print("VERIFYING TFLITE MODELS")

        print("=" * 70)

        for model_file in sorted(self.output_dir.glob("*.tflite")):

            interpreter = tf.lite.Interpreter(model_path=str(model_file))

            interpreter.allocate_tensors()

            print(f"{model_file.name:<25} OK")

        print("=" * 70)

    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("QUANTIZATION COMPLETED")

        print("=" * 70)

        print("Generated Models")

        print("------------------------------")

        for file in sorted(self.output_dir.glob("*.tflite")):

            print(file.name)

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        print()

        print("=" * 70)

        print("STARTING MODEL QUANTIZATION")

        print("=" * 70)

        self.export_dynamic_range()

        self.export_float16()

        self.export_int8()

        self.verify()

        self.benchmark_size()

        self.summary()


if __name__ == "__main__":

    quantizer = ModelQuantizer()

    quantizer.run()
