"""
metrics.py

Evaluation metrics.
"""

from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)


class EvaluationMetrics:

    @staticmethod
    def calculate(y_true, y_pred):

        return {
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
            "f1": f1_score(
                y_true,
                y_pred,
                average="weighted",
            ),
        }

    @staticmethod
    def classification(y_true, y_pred):

        return classification_report(
            y_true,
            y_pred,
        )
