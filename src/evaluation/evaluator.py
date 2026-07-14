"""
evaluator.py

Complete evaluation pipeline.
"""

from src.evaluation.predictor import Predictor
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.confusion_matrix import (
    ConfusionMatrixGenerator,
)
from src.evaluation.report import EvaluationReport


class ModelEvaluator:

    def __init__(
        self,
        model_path="models/best_model.keras",
    ):

        self.predictor = Predictor(model_path)

        self.report = EvaluationReport()

        self.confusion = (
            ConfusionMatrixGenerator()
        )

    def evaluate(
        self,
        images,
        labels,
        class_names=None,
    ):

        predictions, probabilities = (
            self.predictor.predict(images)
        )

        metrics = (
            EvaluationMetrics.calculate(
                labels,
                predictions,
            )
        )

        classification = (
            EvaluationMetrics.classification(
                labels,
                predictions,
            )
        )

        self.report.save_metrics(metrics)

        self.report.save_classification_report(
            classification
        )

        self.report.save_summary(metrics)

        self.confusion.generate(
            labels,
            predictions,
            class_names,
        )

        print()

        print("=" * 70)
        print("Evaluation Complete")
        print("=" * 70)

        print()

        for key, value in metrics.items():

            print(
                f"{key.capitalize():<12}: {value:.4f}"
            )

        print()

        print("Reports saved to:")

        print("reports/evaluation/")

        return (
            metrics,
            predictions,
            probabilities,
        )