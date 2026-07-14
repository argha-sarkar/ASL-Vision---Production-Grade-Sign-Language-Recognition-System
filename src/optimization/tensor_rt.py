"""
tensor_rt.py

Production TensorRT Optimization

Author: Argha Sarkar Project
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import tensorflow as tf


class TensorRTOptimizer:
    """
    TensorRT Optimization Pipeline.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        output_dir="exports/tensorrt",
        precision="FP16",
    ):

        self.model_path = Path(model_path)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.precision = precision.upper()

        self.model = self.load_model()

    # ---------------------------------------------------------

    def load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def export_saved_model(self):

        saved_model_dir = self.output_dir / "saved_model"

        if saved_model_dir.exists():

            shutil.rmtree(saved_model_dir)

        tf.saved_model.save(
            self.model,
            str(saved_model_dir),
        )

        return saved_model_dir

    # ---------------------------------------------------------

    def build_engine(
        self,
        saved_model_dir,
    ):

        trtexec = shutil.which("trtexec")

        if trtexec is None:

            raise RuntimeError("TensorRT (trtexec) not found in PATH.")

        engine_path = self.output_dir / f"asl_vision_{self.precision.lower()}.engine"

        command = [
            trtexec,
            f"--savedModel={saved_model_dir}",
            f"--saveEngine={engine_path}",
            "--workspace=4096",
        ]

        if self.precision == "FP16":

            command.append("--fp16")

        elif self.precision == "INT8":

            command.append("--int8")

        print()

        print("Running TensorRT Engine Builder")

        subprocess.run(
            command,
            check=True,
        )

        return engine_path

    # ---------------------------------------------------------

    def benchmark(
        self,
        engine_path,
    ):

        trtexec = shutil.which("trtexec")

        report_file = self.output_dir / "benchmark.txt"

        command = [
            trtexec,
            f"--loadEngine={engine_path}",
            "--shapes=input:1x28x28x1",
            "--iterations=100",
            "--avgRuns=50",
        ]

        with open(
            report_file,
            "w",
        ) as file:

            subprocess.run(
                command,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        return report_file

    # ---------------------------------------------------------

    def verify(
        self,
        engine_path,
    ):

        if not engine_path.exists():

            raise FileNotFoundError(engine_path)

        size = engine_path.stat().st_size / 1024 / 1024

        print()

        print("=" * 70)

        print("TensorRT Engine")

        print("=" * 70)

        print(f"Engine : {engine_path.name}")

        print(f"Precision : {self.precision}")

        print(f"Size : {size:.2f} MB")

        print("=" * 70)

    # ---------------------------------------------------------

    def export_onnx(
        self,
    ):

        try:

            import tf2onnx

        except ImportError:

            return None

        output = self.output_dir / "model.onnx"

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

        return output

    # ---------------------------------------------------------

    def cleanup(self):

        temp = self.output_dir / "saved_model"

        if temp.exists():

            shutil.rmtree(temp)

    # ---------------------------------------------------------

    def run(self):

        print()

        print("=" * 70)

        print("TensorRT Optimization")

        print("=" * 70)

        saved_model = self.export_saved_model()

        self.export_onnx()

        engine = self.build_engine(saved_model)

        benchmark = self.benchmark(engine)

        self.verify(engine)

        self.cleanup()

        print()

        print("Benchmark Report")

        print(benchmark)

        print()

        print("=" * 70)

        print("TensorRT Optimization Completed")

        print("=" * 70)


if __name__ == "__main__":

    optimizer = TensorRTOptimizer(precision="FP16")

    optimizer.run()
