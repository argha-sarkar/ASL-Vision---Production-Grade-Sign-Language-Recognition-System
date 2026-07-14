"""
model_registry.py

Production MLflow Model Registry

Author: Argha Sarkar Project
"""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """
    MLflow Model Registry Manager.
    """

    def __init__(
        self,
        tracking_uri="mlruns",
    ):

        mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient(tracking_uri=tracking_uri)

    # ---------------------------------------------------------

    def register_model(
        self,
        model_uri,
        model_name,
    ):

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        return result

    # ---------------------------------------------------------

    def create_registered_model(
        self,
        model_name,
    ):

        try:

            return self.client.create_registered_model(model_name)

        except Exception:

            return self.client.get_registered_model(model_name)

    # ---------------------------------------------------------

    def get_registered_model(
        self,
        model_name,
    ):

        return self.client.get_registered_model(model_name)

    # ---------------------------------------------------------

    def list_registered_models(
        self,
    ):

        return self.client.search_registered_models()

    # ---------------------------------------------------------

    def get_latest_versions(
        self,
        model_name,
    ):

        return self.client.get_latest_versions(model_name)

    # ---------------------------------------------------------

    def transition_stage(
        self,
        model_name,
        version,
        stage,
    ):

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )

    # ---------------------------------------------------------

    def delete_model_version(
        self,
        model_name,
        version,
    ):

        self.client.delete_model_version(
            model_name,
            version,
        )

    # ---------------------------------------------------------

    def delete_registered_model(
        self,
        model_name,
    ):

        self.client.delete_registered_model(model_name)

    # ---------------------------------------------------------

    def download_model(
        self,
        model_uri,
        output_dir="downloaded_models",
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(output_dir),
        )

    # ---------------------------------------------------------

    def load_model(
        self,
        model_uri,
    ):

        return mlflow.tensorflow.load_model(model_uri)

    # ---------------------------------------------------------

    def model_summary(
        self,
        model_name,
    ):

        model = self.client.get_registered_model(model_name)

        print("\n" + "=" * 70)
        print("REGISTERED MODEL")
        print("=" * 70)

        print(f"Name        : {model.name}")

        print(f"Description : {model.description}")

        print(f"Created     : {model.creation_timestamp}")

        print(f"Updated     : {model.last_updated_timestamp}")

        print()

        versions = self.client.search_model_versions(f"name='{model_name}'")

        print("=" * 70)
        print("MODEL VERSIONS")
        print("=" * 70)

        for version in versions:

            print(f"Version : {version.version}")

            print(f"Stage   : {version.current_stage}")

            print(f"Run ID  : {version.run_id}")

            print("-" * 70)

    # ---------------------------------------------------------

    def promote_to_production(
        self,
        model_name,
        version,
    ):

        self.transition_stage(
            model_name=model_name,
            version=version,
            stage="Production",
        )

    # ---------------------------------------------------------

    def promote_to_staging(
        self,
        model_name,
        version,
    ):

        self.transition_stage(
            model_name=model_name,
            version=version,
            stage="Staging",
        )

    # ---------------------------------------------------------

    def archive_model(
        self,
        model_name,
        version,
    ):

        self.transition_stage(
            model_name=model_name,
            version=version,
            stage="Archived",
        )


if __name__ == "__main__":

    registry = ModelRegistry()

    models = registry.list_registered_models()

    print("\nRegistered Models\n")

    for model in models:

        print(model.name)
