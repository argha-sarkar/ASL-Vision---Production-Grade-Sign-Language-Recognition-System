"""
evaluator.py

Complete production-grade evaluation pipeline.

Responsibilities
----------------
1. Load trained model
2. Predict validation/test images
3. Calculate evaluation metrics
4. Generate confusion matrix
5. Generate confidence analysis
6. Save reports
"""

from src.evaluation.predictor import Predictor
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.confusion_matrix import (
    ConfusionMatrixGenerator,
)
from src.evaluation.report import EvaluationReport
from src.explainability.confidence import (
    ConfidenceAnalyzer,
)


class ModelEvaluator:
    """
    Complete model evaluation pipeline.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
    ):

        self.predictor = Predictor(model_path)

        self.metrics = EvaluationMetrics()

        self.report = EvaluationReport()

        self.confusion = ConfusionMatrixGenerator()

        self.confidence = ConfidenceAnalyzer()

    def evaluate(
        self,
        images,
        labels,
        class_names=None,
    ):

        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        print("\nGenerating Predictions...\n")

        predictions, probabilities = self.predictor.predict(
            images
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = self.metrics.calculate(
            labels,
            predictions,
        )

        classification_report = (
            self.metrics.classification(
                labels,
                predictions,
            )
        )

        # -------------------------------------------------
        # Save Reports
        # -------------------------------------------------

        self.report.save_metrics(metrics)

        self.report.save_classification_report(
            classification_report
        )

        self.report.save_summary(metrics)

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        confusion_path = self.confusion.generate(
            labels,
            predictions,
            class_names,
        )

        # -------------------------------------------------
        # Confidence Analysis
        # -------------------------------------------------

        confidence_summary = (
            self.confidence.confidence_summary(
                probabilities
            )
        )

        confidence_plot = (
            self.confidence.confidence_histogram(
                probabilities
            )
        )

        low_confidence = (
            self.confidence.low_confidence_indices(
                probabilities,
                threshold=0.70,
            )
        )

        # -------------------------------------------------
        # Print Metrics
        # -------------------------------------------------

        print("=" * 70)
        print("Evaluation Metrics")
        print("=" * 70)

        for key, value in metrics.items():

            print(f"{key:<15}: {value:.4f}")

        # -------------------------------------------------
        # Confidence Summary
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print("Confidence Summary")
        print("=" * 70)

        for key, value in confidence_summary.items():

            print(f"{key:<25}: {value:.4f}")

        print()

        print(
            f"Low Confidence Predictions (<70%) : {len(low_confidence)}"
        )

        # -------------------------------------------------
        # Files Saved
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print("Generated Reports")
        print("=" * 70)

        print(f"Confusion Matrix : {confusion_path}")

        print(f"Confidence Plot  : {confidence_plot}")

        print("Metrics          : reports/evaluation/metrics.json")

        print("Classification   : reports/evaluation/classification_report.txt")

        print("Summary          : reports/evaluation/summary.md")

        print()

        print("=" * 70)
        print("Evaluation Completed Successfully")
        print("=" * 70)

        return {

            "metrics": metrics,

            "predictions": predictions,

            "probabilities": probabilities,

            "confidence_summary": confidence_summary,

            "low_confidence_indices": low_confidence,

        }