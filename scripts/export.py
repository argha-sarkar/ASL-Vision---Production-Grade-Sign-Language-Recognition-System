"""
export.py

Production Model Export Script

Author: Argha Sarkar Project
"""

import argparse
from pathlib import Path

import tensorflow as tf

try:
    import tf2onnx
except ImportError:
    tf2onnx = None

try:
    import onnx
except ImportError:
    onnx = None


class ModelExporter:
    """
    Export trained TensorFlow models into
    multiple production formats.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        export_dir="exports",
    ):

        self.model_path = Path(model_path)

        self.export_dir = Path(export_dir)

        self.export_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = self.load_model()

    # ---------------------------------------------------------

    def load_model(self):

        print("=" * 70)
        print("Loading Model")
        print("=" * 70)

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def export_saved_model(self):

        output = self.export_dir / "saved_model"

        tf.saved_model.save(
            self.model,
            str(output),
        )

        print()

        print("SavedModel Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_keras(self):

        output = self.export_dir / "model.keras"

        self.model.save(output)

        print()

        print("Keras Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_h5(self):

        output = self.export_dir / "model.h5"

        self.model.save(output)

        print()

        print("H5 Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_tflite(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        tflite_model = converter.convert()

        output = self.export_dir / "model.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        print()

        print("TensorFlow Lite Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_quantized_tflite(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        quantized = converter.convert()

        output = self.export_dir / "model_int8.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(quantized)

        print()

        print("Quantized TFLite Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def export_onnx(self):

        if tf2onnx is None:

            print()

            print("tf2onnx is not installed.")

            return None

        output = self.export_dir / "model.onnx"

        spec = (
            tf.TensorSpec(
                (
                    None,
                    28,
                    28,
                    1,
                ),
                tf.float32,
                name="input",
            ),
        )

        tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            output_path=str(output),
        )

        print()

        print("ONNX Model Exported")

        print(output)

        return output

    # ---------------------------------------------------------

    def verify(self):

        print("=" * 70)
        print("VERIFY EXPORTS")
        print("=" * 70)

        for file in sorted(self.export_dir.glob("*")):

            print(f"{file.name:<30}" f"{file.stat().st_size / 1024 / 1024:.2f} MB")

    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("EXPORTED FORMATS")

        print("=" * 70)

        print("TensorFlow SavedModel")

        print("Keras")

        print("HDF5")

        print("TensorFlow Lite")

        print("Quantized TensorFlow Lite")

        print("ONNX")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        self.export_saved_model()

        self.export_keras()

        self.export_h5()

        self.export_tflite()

        self.export_quantized_tflite()

        self.export_onnx()

        self.verify()

        self.summary()


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/best_model.keras",
        type=str,
    )

    parser.add_argument(
        "--output",
        default="exports",
        type=str,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    args = arguments()

    exporter = ModelExporter(
        model_path=args.model,
        export_dir=args.output,
    )

    exporter.run()
