"""
artifact_logger.py

Production MLflow Artifact Logger

Author: Argha Sarkar Project
"""

from pathlib import Path

import mlflow


class ArtifactLogger:
    """
    Log artifacts to MLflow.
    """

    def __init__(
        self,
        tracking_uri="mlruns",
    ):

        mlflow.set_tracking_uri(
            tracking_uri
        )

    # ---------------------------------------------------------

    def log_file(
        self,
        filepath,
    ):

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(filepath)

        mlflow.log_artifact(
            str(filepath)
        )

    # ---------------------------------------------------------

    def log_directory(
        self,
        directory,
    ):

        directory = Path(directory)

        if not directory.exists():

            raise FileNotFoundError(directory)

        mlflow.log_artifacts(
            str(directory)
        )

    # ---------------------------------------------------------

    def log_reports(self):

        folders = [

            "reports",

            "logs",

            "figures",

        ]

        for folder in folders:

            folder = Path(folder)

            if folder.exists():

                mlflow.log_artifacts(
                    str(folder)
                )

    # ---------------------------------------------------------

    def log_tensorboard(self):

        directory = Path(
            "logs/tensorboard"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="tensorboard",

            )

    # ---------------------------------------------------------

    def log_models(self):

        directory = Path(
            "models"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="models",

            )

    # ---------------------------------------------------------

    def log_evaluation(self):

        directory = Path(
            "reports/evaluation"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="evaluation",

            )

    # ---------------------------------------------------------

    def log_explainability(self):

        directory = Path(
            "reports/explainability"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="explainability",

            )

    # ---------------------------------------------------------

    def log_optimization(self):

        directory = Path(
            "reports/optimization"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="optimization",

            )

    # ---------------------------------------------------------

    def log_transfer_learning(self):

        directory = Path(
            "reports/transfer_learning"
        )

        if directory.exists():

            mlflow.log_artifacts(

                str(directory),

                artifact_path="transfer_learning",

            )

    # ---------------------------------------------------------

    def log_images(
        self,
        image_directory,
    ):

        image_directory = Path(
            image_directory
        )

        if not image_directory.exists():

            return

        for image in image_directory.glob("*"):

            if image.suffix.lower() in [

                ".png",

                ".jpg",

                ".jpeg",

                ".svg",

            ]:

                mlflow.log_artifact(
                    str(image)
                )

    # ---------------------------------------------------------

    def log_everything(self):

        self.log_reports()

        self.log_models()

        self.log_tensorboard()

        self.log_evaluation()

        self.log_explainability()

        self.log_optimization()

        self.log_transfer_learning()

    # ---------------------------------------------------------

    def log_project_structure(
        self,
    ):

        root = Path(".")

        tree_file = Path(
            "project_structure.txt"
        )

        with open(
            tree_file,
            "w",
            encoding="utf-8",
        ) as file:

            for path in sorted(
                root.rglob("*")
            ):

                file.write(
                    f"{path}\n"
                )

        mlflow.log_artifact(
            str(tree_file)
        )

        tree_file.unlink(
            missing_ok=True
        )


if __name__ == "__main__":

    logger = ArtifactLogger()

    logger.log_everything()