"""
evaluator.py

Production-grade evaluation pipeline.

Responsibilities
----------------
1. Generate predictions
2. Compute metrics
3. Save reports
4. Generate confusion matrix
5. Perform confidence analysis
6. Save misclassified gallery
7. Generate Grad-CAM
8. Generate feature maps
9. Generate activation maps
10. Generate explainability report

Author: Argha Sarkar Project
"""

import logging
from typing import Dict

from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.predictor import Predictor
from src.evaluation.report import EvaluationReport
from src.explainability.activation_maps import ActivationMapVisualizer
from src.explainability.confidence import ConfidenceAnalyzer
from src.explainability.feature_maps import FeatureMapVisualizer
from src.explainability.gradcam import GradCAM
from src.explainability.misclassified import MisclassifiedAnalyzer
from src.explainability.report import ExplainabilityReport

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Complete evaluation pipeline.
    """

    def __init__(
        self,
        model_path="models/best_model.keras",
    ):

        logger.info("Initializing evaluator...")

        self.predictor = Predictor(model_path)

        self.metrics = EvaluationMetrics()

        self.confusion = ConfusionMatrixGenerator()

        self.report = EvaluationReport()

        self.confidence = ConfidenceAnalyzer()

        self.misclassified = MisclassifiedAnalyzer()

        self.gradcam = GradCAM(self.predictor.model)

        self.feature_maps = FeatureMapVisualizer(self.predictor.model)

        self.activation_maps = ActivationMapVisualizer(self.predictor.model)

        self.explainability = ExplainabilityReport()

    def evaluate(
        self,
        images,
        labels,
        class_names=None,
    ) -> Dict:

        logger.info("Starting evaluation...")

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        predictions, probabilities = self.predictor.predict(images)

        # --------------------------------------------
        # Metrics
        # --------------------------------------------

        metrics = self.metrics.calculate(
            labels,
            predictions,
        )

        classification = self.metrics.classification(
            labels,
            predictions,
        )

        # --------------------------------------------
        # Save Evaluation Reports
        # --------------------------------------------

        self.report.save_metrics(metrics)

        self.report.save_classification_report(classification)

        self.report.save_summary(metrics)

        # --------------------------------------------
        # Confusion Matrix
        # --------------------------------------------

        confusion_path = self.confusion.generate(
            labels,
            predictions,
            class_names,
        )

        # --------------------------------------------
        # Confidence Analysis
        # --------------------------------------------

        confidence_summary = self.confidence.confidence_summary(probabilities)

        confidence_plot = self.confidence.confidence_histogram(probabilities)

        low_confidence = self.confidence.low_confidence_indices(
            probabilities,
            threshold=0.70,
        )

        # --------------------------------------------
        # Misclassified Images
        # --------------------------------------------

        gallery = self.misclassified.save_gallery(
            images,
            labels,
            predictions,
            probabilities,
        )

        # --------------------------------------------
        # Explainability
        # --------------------------------------------

        total = min(
            5,
            len(images),
        )

        for index in range(total):

            self.gradcam.save(
                image=images[index],
                filename=f"gradcam_{index}.png",
            )

        self.feature_maps.generate(images[0])

        self.activation_maps.generate(images[0])

        explainability_report = self.explainability.generate(
            metrics=metrics,
            confidence=confidence_summary,
            model_name="ASL CNN Baseline",
        )

        logger.info("Evaluation complete.")

        return {
            "metrics": metrics,
            "predictions": predictions,
            "probabilities": probabilities,
            "classification_report": classification,
            "confusion_matrix": confusion_path,
            "confidence_summary": confidence_summary,
            "confidence_plot": confidence_plot,
            "low_confidence": low_confidence,
            "misclassified_gallery": gallery,
            "explainability_report": explainability_report,
        }
