"""
onnx_exporter.py

Production ONNX Exporter

Author: Argha Sarkar Project
"""

from pathlib import Path

import tensorflow as tf

try:
    import tf2onnx
except ImportError:
    tf2onnx = None


class ONNXExporter:
    """
    Export TensorFlow model to ONNX.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        output_dir="exports/onnx",
        opset=17,
    ):

        self.model_path = Path(model_path)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.opset = opset

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

    def export(self):

        if tf2onnx is None:

            raise ImportError("Please install tf2onnx")

        output = self.output_dir / "asl_vision.onnx"

        signature = (
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
            input_signature=signature,
            opset=self.opset,
            output_path=str(output),
        )

        print()

        print("ONNX Model Saved")

        print(output)

        return output

    # ---------------------------------------------------------

    def verify(self, model_path):

        try:

            import onnx

        except ImportError:

            print("onnx package not installed.")

            return False

        model = onnx.load(str(model_path))

        onnx.checker.check_model(model)

        print()

        print("ONNX Validation Successful")

        return True

    # ---------------------------------------------------------

    def information(self, model_path):

        size = model_path.stat().st_size / 1024 / 1024

        print()

        print("=" * 70)

        print("ONNX MODEL INFORMATION")

        print("=" * 70)

        print(f"Path      : {model_path}")

        print(f"Size (MB) : {size:.2f}")

        print(f"Opset     : {self.opset}")

        print("=" * 70)

    # ---------------------------------------------------------

    def run(self):

        model = self.export()

        self.verify(model)

        self.information(model)


if __name__ == "__main__":

    exporter = ONNXExporter()

    exporter.run()
