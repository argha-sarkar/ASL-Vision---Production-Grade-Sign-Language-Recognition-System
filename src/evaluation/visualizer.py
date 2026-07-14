"""
visualizer.py

Evaluation plots.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class EvaluationPlots:

    @staticmethod
    def confusion(y_true, y_pred):

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        display = ConfusionMatrixDisplay(confusion_matrix=cm)

        plt.figure(figsize=(10, 10))

        display.plot(
            cmap="Blues",
            values_format="d",
        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.show()
