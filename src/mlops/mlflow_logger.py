"""
mlflow_logger.py

Production MLflow Logger

Author: Argha Sarkar Project
"""

from pathlib import Path

import mlflow
import mlflow.tensorflow


class MLflowLogger:
    """
    Production MLflow Logger.
    """

    def __init__(
        self,
        experiment_name="ASL-Vision",
        tracking_uri="mlruns",
    ):

        self.experiment_name = experiment_name

        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    # ---------------------------------------------------------

    def start_run(
        self,
        run_name=None,
    ):

        mlflow.start_run(run_name=run_name)

    # ---------------------------------------------------------

    def end_run(self):

        mlflow.end_run()

    # ---------------------------------------------------------

    def log_param(
        self,
        key,
        value,
    ):

        mlflow.log_param(
            key,
            value,
        )

    # ---------------------------------------------------------

    def log_params(
        self,
        params,
    ):

        mlflow.log_params(params)

    # ---------------------------------------------------------

    def log_metric(
        self,
        key,
        value,
        step=None,
    ):

        if step is None:

            mlflow.log_metric(
                key,
                value,
            )

        else:

            mlflow.log_metric(
                key,
                value,
                step=step,
            )

    # ---------------------------------------------------------

    def log_metrics(
        self,
        metrics,
        step=None,
    ):

        for key, value in metrics.items():

            self.log_metric(
                key,
                value,
                step,
            )

    # ---------------------------------------------------------

    def log_model(
        self,
        model,
        artifact_path="model",
    ):

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path=artifact_path,
        )

    # ---------------------------------------------------------

    def log_artifact(
        self,
        filepath,
    ):

        filepath = Path(filepath)

        if filepath.exists():

            mlflow.log_artifact(str(filepath))

    # ---------------------------------------------------------

    def log_artifacts(
        self,
        directory,
    ):

        directory = Path(directory)

        if directory.exists():

            mlflow.log_artifacts(str(directory))

    # ---------------------------------------------------------

    def log_figure(
        self,
        figure,
        filename,
    ):

        mlflow.log_figure(
            figure,
            filename,
        )

    # ---------------------------------------------------------

    def log_text(
        self,
        text,
        filename,
    ):

        mlflow.log_text(
            text,
            filename,
        )

    # ---------------------------------------------------------

    def log_dict(
        self,
        dictionary,
        filename,
    ):

        mlflow.log_dict(
            dictionary,
            filename,
        )

    # ---------------------------------------------------------

    def log_tags(
        self,
        tags,
    ):

        mlflow.set_tags(tags)

    # ---------------------------------------------------------

    def log_tag(
        self,
        key,
        value,
    ):

        mlflow.set_tag(
            key,
            value,
        )

    # ---------------------------------------------------------

    def log_history(
        self,
        history,
    ):

        for epoch in range(len(history.history["loss"])):

            for metric, values in history.history.items():

                self.log_metric(
                    metric,
                    values[epoch],
                    epoch,
                )

    # ---------------------------------------------------------

    def log_system_information(
        self,
    ):

        import platform

        import tensorflow as tf

        self.log_tags(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
            }
        )

    # ---------------------------------------------------------

    def log_training_results(
        self,
        history,
        model,
        parameters,
        metrics,
    ):

        self.start_run()

        self.log_system_information()

        self.log_params(parameters)

        self.log_metrics(metrics)

        self.log_history(history)

        self.log_model(model)

        self.end_run()

    # ---------------------------------------------------------

    def log_directory(
        self,
        directory,
    ):

        directory = Path(directory)

        if not directory.exists():

            return

        for file in directory.rglob("*"):

            if file.is_file():

                mlflow.log_artifact(str(file))

    # ---------------------------------------------------------

    def active_run(self):

        return mlflow.active_run()

    # ---------------------------------------------------------

    def get_run_id(self):

        run = mlflow.active_run()

        if run is None:

            return None

        return run.info.run_id

    # ---------------------------------------------------------

    def get_experiment(self):

        return mlflow.get_experiment_by_name(self.experiment_name)


if __name__ == "__main__":

    logger = MLflowLogger()

    logger.start_run("test_run")

    logger.log_param(
        "learning_rate",
        0.001,
    )

    logger.log_metric(
        "accuracy",
        0.98,
    )

    logger.end_run()
