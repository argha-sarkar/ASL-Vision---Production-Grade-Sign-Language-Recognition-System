"""
evaluator.py

Production Evaluation Pipeline for Transfer Learning Models.

Author: Argha Sarkar Project
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.explainability.confidence import ConfidenceAnalyzer
from src.explainability.gradcam import GradCAM
from src.explainability.misclassified import MisclassifiedAnalyzer


class TransferLearningEvaluator:
    """
    Evaluate Transfer Learning Models.
    """

    def __init__(
        self,
        model,
        class_names=None,
        report_dir="reports/transfer_learning",
    ):

        self.model = model

        self.class_names = class_names

        self.report_dir = Path(report_dir)

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.confusion = ConfusionMatrixGenerator()

        self.confidence = ConfidenceAnalyzer()

        self.misclassified = MisclassifiedAnalyzer()

        self.gradcam = GradCAM(model)

    # ---------------------------------------------------------

    def predict(
        self,
        dataset,
    ):

        probabilities = self.model.predict(
            dataset,
            verbose=1,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        labels = []

        images = []

        for image_batch, label_batch in dataset:

            images.extend(image_batch.numpy())

            labels.extend(label_batch.numpy())

        labels = np.array(labels)

        images = np.array(images)

        return (
            images,
            labels,
            predictions,
            probabilities,
        )

    # ---------------------------------------------------------

    def metrics(
        self,
        y_true,
        y_pred,
    ):

        results = {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                average="weighted",
            ),
        }

        return results

    # ---------------------------------------------------------

    def save_metrics(
        self,
        metrics,
    ):

        save_path = self.report_dir / "metrics.json"

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

    # ---------------------------------------------------------

    def save_classification_report(
        self,
        y_true,
        y_pred,
    ):

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
        )

        save_path = self.report_dir / "classification_report.txt"

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

    # ---------------------------------------------------------

    def evaluate(
        self,
        dataset,
    ):

        print("\n" + "=" * 70)
        print("Transfer Learning Evaluation")
        print("=" * 70)

        (
            images,
            labels,
            predictions,
            probabilities,
        ) = self.predict(dataset)

        metrics = self.metrics(
            labels,
            predictions,
        )

        self.save_metrics(metrics)

        self.save_classification_report(
            labels,
            predictions,
        )

        confusion_path = self.confusion.generate(
            labels,
            predictions,
            self.class_names,
        )

        confidence_summary = self.confidence.confidence_summary(probabilities)

        confidence_plot = self.confidence.confidence_histogram(probabilities)

        misclassified = self.misclassified.save_gallery(
            images,
            labels,
            predictions,
            probabilities,
        )

        total = min(
            5,
            len(images),
        )

        for i in range(total):

            self.gradcam.save(
                image=images[i],
                filename=f"gradcam_{i}.png",
            )

        print("\n" + "=" * 70)
        print("Evaluation Metrics")
        print("=" * 70)

        for key, value in metrics.items():

            print(f"{key:<15}: {value:.4f}")

        print("\n" + "=" * 70)
        print("Confidence Summary")
        print("=" * 70)

        for key, value in confidence_summary.items():

            print(f"{key:<25}: {value:.4f}")

        print("\n" + "=" * 70)
        print("Generated Files")
        print("=" * 70)

        print(f"Metrics              : {self.report_dir / 'metrics.json'}")

        print(f"Classification       : {self.report_dir / 'classification_report.txt'}")

        print(f"Confusion Matrix     : {confusion_path}")

        print(f"Confidence Plot      : {confidence_plot}")

        print(f"Misclassified Images : {misclassified}")

        print("\nEvaluation Completed Successfully.\n")

        return {
            "metrics": metrics,
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
            "confidence": confidence_summary,
            "confusion_matrix": confusion_path,
            "misclassified": misclassified,
        }


if __name__ == "__main__":

    print("TransferLearningEvaluator module loaded successfully.")
