"""
model_loader.py

Production MLflow Model Loader

Author: Argha Sarkar Project
"""

from pathlib import Path

import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient


class ModelLoader:
    """
    Load models from MLflow.
    """

    def __init__(
        self,
        tracking_uri="mlruns",
    ):

        mlflow.set_tracking_uri(
            tracking_uri
        )

        self.client = MlflowClient(
            tracking_uri=tracking_uri
        )

    # ---------------------------------------------------------

    def load_model(
        self,
        model_uri,
    ):

        return mlflow.tensorflow.load_model(
            model_uri
        )

    # ---------------------------------------------------------

    def load_registered_model(
        self,
        model_name,
        stage="Production",
    ):

        model_uri = (
            f"models:/{model_name}/{stage}"
        )

        return self.load_model(
            model_uri
        )

    # ---------------------------------------------------------

    def load_model_version(
        self,
        model_name,
        version,
    ):

        model_uri = (
            f"models:/{model_name}/{version}"
        )

        return self.load_model(
            model_uri
        )

    # ---------------------------------------------------------

    def latest_version(
        self,
        model_name,
    ):

        versions = self.client.get_latest_versions(
            model_name
        )

        if len(versions) == 0:

            return None

        return versions[0]

    # ---------------------------------------------------------

    def latest_model(
        self,
        model_name,
    ):

        version = self.latest_version(
            model_name
        )

        if version is None:

            return None

        return self.load_model_version(

            model_name,

            version.version,

        )

    # ---------------------------------------------------------

    def download_artifacts(
        self,
        run_id,
        destination="downloads",
    ):

        destination = Path(
            destination
        )

        destination.mkdir(

            parents=True,

            exist_ok=True,

        )

        return self.client.download_artifacts(

            run_id,

            "",

            str(destination),

        )

    # ---------------------------------------------------------

    def list_registered_models(
        self,
    ):

        return self.client.search_registered_models()

    # ---------------------------------------------------------

    def list_versions(
        self,
        model_name,
    ):

        return self.client.search_model_versions(

            f"name='{model_name}'"

        )

    # ---------------------------------------------------------

    def model_exists(
        self,
        model_name,
    ):

        try:

            self.client.get_registered_model(
                model_name
            )

            return True

        except Exception:

            return False

    # ---------------------------------------------------------

    def print_registered_models(
        self,
    ):

        print("\n" + "=" * 70)
        print("REGISTERED MODELS")
        print("=" * 70)

        models = self.list_registered_models()

        if len(models) == 0:

            print("No registered models.")

            return

        for model in models:

            print(
                f"Name : {model.name}"
            )

            print(
                f"Description : {model.description}"
            )

            print("-" * 70)

    # ---------------------------------------------------------

    def print_versions(
        self,
        model_name,
    ):

        versions = self.list_versions(
            model_name
        )

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        for version in versions:

            print(
                f"Version : {version.version}"
            )

            print(
                f"Stage   : {version.current_stage}"
            )

            print(
                f"Run ID  : {version.run_id}"
            )

            print("-" * 70)


if __name__ == "__main__":

    loader = ModelLoader()

    loader.print_registered_models()