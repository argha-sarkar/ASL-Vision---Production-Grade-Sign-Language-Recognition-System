"""
report.py

Production MLflow Report Generator

Author: Argha Sarkar Project
"""

from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


class MLflowReport:
    """
    Generate MLflow experiment reports.
    """

    def __init__(
        self,
        experiment_name="ASL-Vision",
        tracking_uri="mlruns",
        output_dir="reports/mlflow",
    ):

        mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient(tracking_uri=tracking_uri)

        self.experiment_name = experiment_name

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:

            raise ValueError(f"Experiment '{experiment_name}' not found.")

        self.experiment_id = experiment.experiment_id

    # ---------------------------------------------------------

    def load_runs(self):

        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=10000,
        )

        data = []

        for run in runs:

            row = {
                "Run ID": run.info.run_id,
                "Status": run.info.status,
                "Start Time": run.info.start_time,
                "End Time": run.info.end_time,
            }

            row.update(run.data.params)

            row.update(run.data.metrics)

            data.append(row)

        return pd.DataFrame(data)

    # ---------------------------------------------------------

    def save_csv(
        self,
        dataframe,
    ):

        path = self.output_dir / "experiment_runs.csv"

        dataframe.to_csv(
            path,
            index=False,
        )

        return path

    # ---------------------------------------------------------

    def save_excel(
        self,
        dataframe,
    ):

        path = self.output_dir / "experiment_runs.xlsx"

        dataframe.to_excel(
            path,
            index=False,
        )

        return path

    # ---------------------------------------------------------

    def best_run(
        self,
        dataframe,
        metric="accuracy",
    ):

        if metric not in dataframe.columns:

            return None

        return dataframe.loc[dataframe[metric].idxmax()]

    # ---------------------------------------------------------

    def markdown_report(
        self,
        dataframe,
    ):

        report_path = self.output_dir / "mlflow_report.md"

        best = self.best_run(dataframe)

        markdown = f"""# MLflow Experiment Report

Generated:

{datetime.now()}

---

## Experiment

{self.experiment_name}

---

## Total Runs

{len(dataframe)}

---

"""

        if best is not None:

            markdown += f"""## Best Run

Run ID:

{best['Run ID']}

Accuracy:

{best['accuracy']:.5f}

---

"""

        markdown += """## Runs

"""

        markdown += dataframe.to_markdown(index=False)

        markdown += """

---

## Files

- experiment_runs.csv
- experiment_runs.xlsx
- mlflow_report.md

"""

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

        return report_path

    # ---------------------------------------------------------

    def summary(
        self,
        dataframe,
    ):

        print("\n" + "=" * 70)
        print("MLFLOW REPORT")
        print("=" * 70)

        print(f"Experiment : {self.experiment_name}")

        print(f"Runs       : {len(dataframe)}")

        best = self.best_run(dataframe)

        if best is not None:

            print(f"Best Accuracy : {best['accuracy']:.5f}")

            print(f"Best Run ID   : {best['Run ID']}")

    # ---------------------------------------------------------

    def generate(self):

        dataframe = self.load_runs()

        csv_path = self.save_csv(dataframe)

        excel_path = self.save_excel(dataframe)

        markdown_path = self.markdown_report(dataframe)

        self.summary(dataframe)

        print("\nGenerated Files\n")

        print(csv_path)

        print(excel_path)

        print(markdown_path)

        return {
            "dataframe": dataframe,
            "csv": csv_path,
            "excel": excel_path,
            "markdown": markdown_path,
        }


if __name__ == "__main__":

    report = MLflowReport()

    report.generate()
