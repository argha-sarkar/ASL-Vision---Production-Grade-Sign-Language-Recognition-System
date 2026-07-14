"""
experiment_manager.py

Production MLflow Experiment Manager

Author: Argha Sarkar Project
"""

from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


class ExperimentManager:
    """
    MLflow Experiment Manager.
    """

    def __init__(
        self,
        experiment_name="ASL-Vision",
        tracking_uri="mlruns",
    ):

        self.experiment_name = experiment_name

        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(
            tracking_uri
        )

        self.client = MlflowClient(
            tracking_uri=tracking_uri
        )

        self.experiment_id = (
            self._create_or_get_experiment()
        )

    # ---------------------------------------------------------

    def _create_or_get_experiment(
        self,
    ):

        experiment = mlflow.get_experiment_by_name(
            self.experiment_name
        )

        if experiment is not None:

            return experiment.experiment_id

        experiment_id = mlflow.create_experiment(
            self.experiment_name
        )

        return experiment_id

    # ---------------------------------------------------------

    def set_experiment(self):

        mlflow.set_experiment(
            self.experiment_name
        )

    # ---------------------------------------------------------

    def experiment(self):

        return self.client.get_experiment(
            self.experiment_id
        )

    # ---------------------------------------------------------

    def list_experiments(self):

        return self.client.search_experiments()

    # ---------------------------------------------------------

    def create_run(
        self,
        run_name=None,
    ):

        if run_name is None:

            run_name = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        return self.client.create_run(

            experiment_id=self.experiment_id,

            tags={
                "mlflow.runName": run_name
            },

        )

    # ---------------------------------------------------------

    def delete_run(
        self,
        run_id,
    ):

        self.client.delete_run(
            run_id
        )

    # ---------------------------------------------------------

    def get_run(
        self,
        run_id,
    ):

        return self.client.get_run(
            run_id
        )

    # ---------------------------------------------------------

    def search_runs(
        self,
        order_by=None,
        max_results=100,
    ):

        if order_by is None:

            order_by = [
                "metrics.accuracy DESC"
            ]

        return self.client.search_runs(

            experiment_ids=[
                self.experiment_id
            ],

            order_by=order_by,

            max_results=max_results,

        )

    # ---------------------------------------------------------

    def best_run(
        self,
        metric="accuracy",
    ):

        runs = self.client.search_runs(

            experiment_ids=[
                self.experiment_id
            ],

            order_by=[
                f"metrics.{metric} DESC"
            ],

            max_results=1,

        )

        if len(runs) == 0:

            return None

        return runs[0]

    # ---------------------------------------------------------

    def latest_run(self):

        runs = self.client.search_runs(

            experiment_ids=[
                self.experiment_id
            ],

            order_by=[
                "attributes.start_time DESC"
            ],

            max_results=1,

        )

        if len(runs) == 0:

            return None

        return runs[0]

    # ---------------------------------------------------------

    def list_run_ids(self):

        runs = self.search_runs()

        return [

            run.info.run_id

            for run in runs

        ]

    # ---------------------------------------------------------

    def download_artifacts(
        self,
        run_id,
        artifact_path="",
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

            artifact_path,

            str(destination),

        )

    # ---------------------------------------------------------

    def delete_experiment(self):

        self.client.delete_experiment(

            self.experiment_id

        )

    # ---------------------------------------------------------

    def restore_experiment(self):

        self.client.restore_experiment(

            self.experiment_id

        )

    # ---------------------------------------------------------

    def experiment_summary(self):

        experiment = self.experiment()

        print("\n" + "=" * 70)
        print("MLFLOW EXPERIMENT")
        print("=" * 70)

        print(
            f"Name          : {experiment.name}"
        )

        print(
            f"Experiment ID : {experiment.experiment_id}"
        )

        print(
            f"Artifact URI  : {experiment.artifact_location}"
        )

        print(
            f"Lifecycle     : {experiment.lifecycle_stage}"
        )

    # ---------------------------------------------------------

    def run_summary(
        self,
        run,
    ):

        print("\n" + "=" * 70)
        print("RUN SUMMARY")
        print("=" * 70)

        print(
            f"Run ID : {run.info.run_id}"
        )

        print(
            f"Status : {run.info.status}"
        )

        print(
            f"Start  : {run.info.start_time}"
        )

        print(
            f"End    : {run.info.end_time}"
        )

        print("\nParameters")

        for key, value in run.data.params.items():

            print(
                f"{key:<20}: {value}"
            )

        print("\nMetrics")

        for key, value in run.data.metrics.items():

            print(
                f"{key:<20}: {value}"
            )

    # ---------------------------------------------------------

    def compare_runs(
        self,
        metric="accuracy",
        top_k=5,
    ):

        runs = self.client.search_runs(

            experiment_ids=[
                self.experiment_id
            ],

            order_by=[
                f"metrics.{metric} DESC"
            ],

            max_results=top_k,

        )

        print("\n" + "=" * 90)
        print("TOP RUNS")
        print("=" * 90)

        for i, run in enumerate(runs, start=1):

            value = run.data.metrics.get(
                metric,
                0,
            )

            print(
                f"{i}. "
                f"{run.info.run_id} "
                f"-> {metric}: {value:.5f}"
            )


if __name__ == "__main__":

    manager = ExperimentManager()

    manager.experiment_summary()

    best = manager.best_run()

    if best is not None:

        manager.run_summary(best)