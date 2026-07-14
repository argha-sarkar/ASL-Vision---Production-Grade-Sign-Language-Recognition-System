"""
profiler.py

Production Model Profiler

Author: Argha Sarkar Project
"""

from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd
import tensorflow as tf


class ModelProfiler:
    """
    Profile TensorFlow model performance.
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

            raise FileNotFoundError(

                f"{self.model_path} not found."

            )

        return tf.keras.models.load_model(

            self.model_path

        )

    # ---------------------------------------------------------

    def sample(self):

        return np.random.rand(

            1,

            self.input_shape[0],

            self.input_shape[1],

            self.input_shape[2],

        ).astype(np.float32)

    # ---------------------------------------------------------

    def warmup(self):

        image = self.sample()

        for _ in range(10):

            self.model.predict(

                image,

                verbose=0,

            )

    # ---------------------------------------------------------

    def cpu_profile(self):

        image = self.sample()

        execution = []

        for _ in range(self.iterations):

            start = time.perf_counter()

            self.model.predict(

                image,

                verbose=0,

            )

            end = time.perf_counter()

            execution.append(

                end - start

            )

        return {

            "average_ms":

                np.mean(execution) * 1000,

            "minimum_ms":

                np.min(execution) * 1000,

            "maximum_ms":

                np.max(execution) * 1000,

            "std_ms":

                np.std(execution) * 1000,

        }

    # ---------------------------------------------------------

    def memory_profile(self):

        image = self.sample()

        tracemalloc.start()

        self.model.predict(

            image,

            verbose=0,

        )

        current, peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        return {

            "current_mb":

                current / 1024 / 1024,

            "peak_mb":

                peak / 1024 / 1024,

        }

    # ---------------------------------------------------------

    def gpu_information(self):

        gpus = tf.config.list_physical_devices(

            "GPU"

        )

        return {

            "gpu_available":

                len(gpus) > 0,

            "gpu_count":

                len(gpus),

            "devices":

                [gpu.name for gpu in gpus],

        }

    # ---------------------------------------------------------

    def model_information(self):

        trainable = np.sum(

            [

                tf.keras.backend.count_params(

                    variable

                )

                for variable in self.model.trainable_weights

            ]

        )

        non_trainable = np.sum(

            [

                tf.keras.backend.count_params(

                    variable

                )

                for variable in self.model.non_trainable_weights

            ]

        )

        return {

            "total_parameters":

                self.model.count_params(),

            "trainable_parameters":

                int(trainable),

            "non_trainable_parameters":

                int(non_trainable),

        }

    # ---------------------------------------------------------

    def tensorflow_information(self):

        return {

            "tensorflow":

                tf.__version__,

            "eager_execution":

                tf.executing_eagerly(),

            "built_with_cuda":

                tf.test.is_built_with_cuda(),

        }

    # ---------------------------------------------------------

    def full_profile(self):

        self.warmup()

        profile = {}

        profile.update(

            self.cpu_profile()

        )

        profile.update(

            self.memory_profile()

        )

        profile.update(

            self.model_information()

        )

        profile.update(

            self.gpu_information()

        )

        profile.update(

            self.tensorflow_information()

        )

        return profile

    # ---------------------------------------------------------

    def save(

        self,

        output="reports/testing/profile.csv",

    ):

        output = Path(output)

        output.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        profile = self.full_profile()

        dataframe = pd.DataFrame(

            profile.items(),

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

        profile = self.full_profile()

        print()

        print("=" * 70)

        print("MODEL PROFILE")

        print("=" * 70)

        for key, value in profile.items():

            print(

                f"{key:<30}: {value}"

            )

        print("=" * 70)


if __name__ == "__main__":

    profiler = ModelProfiler()

    profiler.print_report()

    profiler.save()