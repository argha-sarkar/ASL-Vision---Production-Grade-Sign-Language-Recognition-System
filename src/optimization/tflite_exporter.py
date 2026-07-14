"""
tflite_exporter.py

Production TensorFlow Lite Exporter

Author: Argha Sarkar Project
"""

from pathlib import Path

import tensorflow as tf


class TFLiteExporter:
    """
    Export TensorFlow model to TensorFlow Lite.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        output_dir="exports/tflite",
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

        print("=" * 70)
        print("Loading TensorFlow Model")
        print("=" * 70)

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def export_fp32(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        tflite_model = converter.convert()

        output = self.output_dir / "asl_vision_fp32.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        print()

        print("FP32 Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_dynamic(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        output = self.output_dir / "asl_vision_dynamic.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        print()

        print("Dynamic Quantization Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_float16(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        output = self.output_dir / "asl_vision_fp16.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        print()

        print("Float16 Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def representative_dataset(self):

        for _ in range(100):

            sample = tf.random.uniform(
                (
                    1,
                    28,
                    28,
                    1,
                ),
                dtype=tf.float32,
            )

            yield [sample]

    # ---------------------------------------------------------

    def export_int8(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.representative_dataset = self.representative_dataset

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        converter.inference_input_type = tf.uint8

        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()

        output = self.output_dir / "asl_vision_int8.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        print()

        print("INT8 Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def verify(self):

        print()

        print("=" * 70)

        print("GENERATED TFLITE MODELS")

        print("=" * 70)

        for model in sorted(self.output_dir.glob("*.tflite")):

            size = model.stat().st_size / 1024 / 1024

            print(f"{model.name:<35}" f"{size:.2f} MB")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        self.export_fp32()

        self.export_dynamic()

        self.export_float16()

        self.export_int8()

        self.verify()


if __name__ == "__main__":

    exporter = TFLiteExporter()

    exporter.run()
