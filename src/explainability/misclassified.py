"""
misclassified.py

Analyze and visualize misclassified images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MisclassifiedAnalyzer:

    def __init__(self):

        self.output_dir = Path("reports/explainability")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get_misclassified(
        self,
        images,
        y_true,
        y_pred,
        probabilities,
    ):

        incorrect = np.where(y_true != y_pred)[0]

        return (
            images[incorrect],
            y_true[incorrect],
            y_pred[incorrect],
            probabilities[incorrect],
        )

    def save_gallery(
        self,
        images,
        y_true,
        y_pred,
        probabilities,
        max_images=16,
    ):

        (
            wrong_images,
            wrong_true,
            wrong_pred,
            wrong_prob,
        ) = self.get_misclassified(
            images,
            y_true,
            y_pred,
            probabilities,
        )

        total = min(
            len(wrong_images),
            max_images,
        )

        if total == 0:

            print("No misclassified images found.")

            return

        rows = 4
        cols = 4

        plt.figure(figsize=(12, 12))

        for i in range(total):

            plt.subplot(
                rows,
                cols,
                i + 1,
            )

            plt.imshow(
                wrong_images[i].squeeze(),
                cmap="gray",
            )

            confidence = np.max(wrong_prob[i])

            plt.title(
                f"T:{wrong_true[i]}\n" f"P:{wrong_pred[i]}\n" f"C:{confidence:.2f}"
            )

            plt.axis("off")

        plt.tight_layout()

        save_path = self.output_dir / "misclassified_images.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return save_path
