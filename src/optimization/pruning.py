"""
pruning.py

Production TensorFlow Model Pruning

Author: Argha Sarkar Project
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    tfmot = None


class ModelPruner:
    """
    TensorFlow Model Pruning Pipeline.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
        output_dir="exports/pruned",
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

        if tfmot is None:

            raise ImportError("Please install tensorflow-model-optimization")

        if not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        return tf.keras.models.load_model(self.model_path)

    # ---------------------------------------------------------

    def prune_model(
        self,
        initial_sparsity=0.30,
        final_sparsity=0.80,
        begin_step=0,
        end_step=1000,
    ):

        schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        )

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            pruning_schedule=schedule,
        )

        pruned_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
            ],
        )

        return pruned_model

    # ---------------------------------------------------------

    def strip_pruning(
        self,
        model,
    ):

        return tfmot.sparsity.keras.strip_pruning(model)

    # ---------------------------------------------------------

    def save_model(
        self,
        model,
    ):

        output = self.output_dir / "pruned_model.keras"

        model.save(output)

        return output

    # ---------------------------------------------------------

    def export_tflite(
        self,
        model,
    ):

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        output = self.output_dir / "pruned_model.tflite"

        with open(
            output,
            "wb",
        ) as file:

            file.write(tflite_model)

        return output

    # ---------------------------------------------------------

    def benchmark(
        self,
        original,
        pruned,
    ):

        original_size = Path(original).stat().st_size / 1024 / 1024

        pruned_size = Path(pruned).stat().st_size / 1024 / 1024

        reduction = ((original_size - pruned_size) / original_size) * 100

        print()

        print("=" * 70)

        print("PRUNING SUMMARY")

        print("=" * 70)

        print(f"{'Original':<20}" f"{original_size:.2f} MB")

        print(f"{'Pruned':<20}" f"{pruned_size:.2f} MB")

        print(f"{'Reduction':<20}" f"{reduction:.2f}%")

        print("=" * 70)

    # ---------------------------------------------------------

    def verify(
        self,
        model_path,
    ):

        model = tf.keras.models.load_model(model_path)

        sample = np.random.rand(
            1,
            28,
            28,
            1,
        ).astype(np.float32)

        prediction = model.predict(
            sample,
            verbose=0,
        )

        print()

        print("Verification Successful")

        print(
            "Prediction Shape:",
            prediction.shape,
        )

    # ---------------------------------------------------------

    def run(self):

        print()

        print("=" * 70)

        print("STARTING MODEL PRUNING")

        print("=" * 70)

        pruned = self.prune_model()

        stripped = self.strip_pruning(pruned)

        keras_path = self.save_model(stripped)

        self.export_tflite(stripped)

        self.verify(keras_path)

        self.benchmark(
            self.model_path,
            keras_path,
        )

        print()

        print("Pruning Completed Successfully")


if __name__ == "__main__":

    pruner = ModelPruner()

    pruner.run()
