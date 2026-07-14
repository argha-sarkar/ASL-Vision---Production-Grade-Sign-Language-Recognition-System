"""
utils.py

Production MLflow Utility Functions

Author: Argha Sarkar Project
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf


class MLflowUtils:
    """
    Utility functions for MLflow.
    """

    # ---------------------------------------------------------

    @staticmethod
    def create_directory(
        directory,
    ):

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    # ---------------------------------------------------------

    @staticmethod
    def current_timestamp():

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------

    @staticmethod
    def save_json(
        data,
        filepath,
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )

    # ---------------------------------------------------------

    @staticmethod
    def load_json(
        filepath,
    ):

        filepath = Path(filepath)

        if not filepath.exists():

            return None

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------

    @staticmethod
    def set_random_seed(
        seed=42,
    ):

        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)

    # ---------------------------------------------------------

    @staticmethod
    def log_environment():

        mlflow.log_param(
            "python_version",
            os.sys.version,
        )

        mlflow.log_param(
            "tensorflow_version",
            tf.__version__,
        )

        mlflow.log_param(
            "numpy_version",
            np.__version__,
        )

    # ---------------------------------------------------------

    @staticmethod
    def log_gpu_information():

        gpus = tf.config.list_physical_devices("GPU")

        mlflow.log_param(
            "gpu_available",
            len(gpus) > 0,
        )

        mlflow.log_param(
            "gpu_count",
            len(gpus),
        )

    # ---------------------------------------------------------

    @staticmethod
    def log_system_information():

        import platform

        mlflow.log_param(
            "platform",
            platform.platform(),
        )

        mlflow.log_param(
            "processor",
            platform.processor(),
        )

        mlflow.log_param(
            "machine",
            platform.machine(),
        )

    # ---------------------------------------------------------

    @staticmethod
    def log_training_configuration(
        config,
    ):

        for key, value in config.items():

            mlflow.log_param(
                key,
                value,
            )

    # ---------------------------------------------------------

    @staticmethod
    def save_model_summary(
        model,
        filepath,
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            model.summary(print_fn=lambda x: file.write(x + "\n"))

    # ---------------------------------------------------------

    @staticmethod
    def log_model_summary(
        model,
    ):

        summary_path = Path("reports/mlflow/model_summary.txt")

        MLflowUtils.save_model_summary(
            model,
            summary_path,
        )

        mlflow.log_artifact(str(summary_path))

    # ---------------------------------------------------------

    @staticmethod
    def experiment_exists(
        experiment_name,
    ):

        experiment = mlflow.get_experiment_by_name(experiment_name)

        return experiment is not None

    # ---------------------------------------------------------

    @staticmethod
    def active_run_id():

        run = mlflow.active_run()

        if run is None:

            return None

        return run.info.run_id

    # ---------------------------------------------------------

    @staticmethod
    def print_banner():

        print("\n")

        print("=" * 70)

        print("MLFLOW PIPELINE")

        print("=" * 70)

        print(f"Started : {MLflowUtils.current_timestamp()}")

        print("=" * 70)


if __name__ == "__main__":

    MLflowUtils.print_banner()

    MLflowUtils.set_random_seed()

    print(MLflowUtils.current_timestamp())
