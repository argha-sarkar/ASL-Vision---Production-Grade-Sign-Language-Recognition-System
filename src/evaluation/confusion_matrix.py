"""
confusion_matrix.py

Generate and save confusion matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


class ConfusionMatrixGenerator:

    def __init__(self):

        self.output_dir = Path("reports/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        y_true,
        y_pred,
        class_names=None,
    ):

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        fig, ax = plt.subplots(
            figsize=(12, 12)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        display.plot(
            cmap="Blues",
            xticks_rotation=90,
            ax=ax,
            colorbar=True,
        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        save_path = self.output_dir / "confusion_matrix.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return save_path