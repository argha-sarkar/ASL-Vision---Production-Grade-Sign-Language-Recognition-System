"""
confidence.py

Analyze prediction confidence.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ConfidenceAnalyzer:

    def __init__(self):

        self.output_dir = Path("reports/explainability")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def confidence_scores(
        self,
        probabilities,
    ):

        return np.max(
            probabilities,
            axis=1,
        )

    def confidence_histogram(
        self,
        probabilities,
    ):

        confidence = self.confidence_scores(probabilities)

        plt.figure(figsize=(10, 5))

        plt.hist(
            confidence,
            bins=25,
            edgecolor="black",
        )

        plt.title("Prediction Confidence Distribution")

        plt.xlabel("Confidence")

        plt.ylabel("Number of Images")

        plt.grid(True)

        save_path = self.output_dir / "confidence_distribution.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return save_path

    def confidence_summary(
        self,
        probabilities,
    ):

        confidence = self.confidence_scores(probabilities)

        return {
            "Mean Confidence": float(np.mean(confidence)),
            "Minimum Confidence": float(np.min(confidence)),
            "Maximum Confidence": float(np.max(confidence)),
            "Median Confidence": float(np.median(confidence)),
        }

    def low_confidence_indices(
        self,
        probabilities,
        threshold=0.70,
    ):

        confidence = self.confidence_scores(probabilities)

        return np.where(confidence < threshold)[0]
