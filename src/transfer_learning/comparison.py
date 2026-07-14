"""
comparison.py

Compare multiple Deep Learning models.

Author: Argha Sarkar Project
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


class ModelComparison:
    """
    Compare multiple trained models.
    """

    def __init__(
        self,
        output_dir="reports/transfer_learning",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results = []

    # ---------------------------------------------------------

    def add_result(
        self,
        model_name,
        metrics,
        training_time=None,
        inference_time=None,
        parameters=None,
    ):

        row = {

            "Model": model_name,

            "Accuracy": metrics.get(
                "accuracy",
                0,
            ),

            "Precision": metrics.get(
                "precision",
                0,
            ),

            "Recall": metrics.get(
                "recall",
                0,
            ),

            "F1 Score": metrics.get(
                "f1_score",
                0,
            ),

            "Training Time": training_time,

            "Inference Time": inference_time,

            "Parameters": parameters,

        }

        self.results.append(row)

    # ---------------------------------------------------------

    def dataframe(self):

        return pd.DataFrame(
            self.results
        )

    # ---------------------------------------------------------

    def save_csv(self):

        df = self.dataframe()

        save_path = (
            self.output_dir /
            "model_comparison.csv"
        )

        df.to_csv(
            save_path,
            index=False,
        )

        return save_path

    # ---------------------------------------------------------

    def save_json(self):

        save_path = (
            self.output_dir /
            "model_comparison.json"
        )

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.results,
                file,
                indent=4,
            )

        return save_path

    # ---------------------------------------------------------

    def plot_accuracy(self):

        df = self.dataframe()

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(

            df["Model"],

            df["Accuracy"],

        )

        plt.ylabel("Accuracy")

        plt.xlabel("Model")

        plt.title(
            "Model Accuracy Comparison"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        save_path = (
            self.output_dir /
            "accuracy_comparison.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()

        return save_path

    # ---------------------------------------------------------

    def plot_f1(self):

        df = self.dataframe()

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(

            df["Model"],

            df["F1 Score"],

        )

        plt.ylabel("F1 Score")

        plt.xlabel("Model")

        plt.title(
            "Model F1 Score Comparison"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        save_path = (
            self.output_dir /
            "f1_comparison.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()

        return save_path

    # ---------------------------------------------------------

    def best_model(self):

        df = self.dataframe()

        index = df[
            "Accuracy"
        ].idxmax()

        return df.iloc[index]

    # ---------------------------------------------------------

    def print_summary(self):

        df = self.dataframe()

        print("\n")
        print("=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(df)

        print("\n")

        best = self.best_model()

        print(
            f"Best Model : {best['Model']}"
        )

        print(
            f"Accuracy   : {best['Accuracy']:.4f}"
        )

        print(
            f"F1 Score   : {best['F1 Score']:.4f}"
        )

    # ---------------------------------------------------------

    def generate(self):

        csv_path = self.save_csv()

        json_path = self.save_json()

        accuracy_plot = (
            self.plot_accuracy()
        )

        f1_plot = (
            self.plot_f1()
        )

        self.print_summary()

        return {

            "csv": csv_path,

            "json": json_path,

            "accuracy_plot": accuracy_plot,

            "f1_plot": f1_plot,

        }


if __name__ == "__main__":

    comparison = ModelComparison()

    comparison.add_result(

        "Baseline CNN",

        {

            "accuracy": 0.962,

            "precision": 0.963,

            "recall": 0.962,

            "f1_score": 0.962,

        },

        training_time=120,

        inference_time=8,

        parameters=245000,

    )

    comparison.add_result(

        "EfficientNetB0",

        {

            "accuracy": 0.988,

            "precision": 0.988,

            "recall": 0.988,

            "f1_score": 0.988,

        },

        training_time=185,

        inference_time=12,

        parameters=5288548,

    )

    comparison.generate()