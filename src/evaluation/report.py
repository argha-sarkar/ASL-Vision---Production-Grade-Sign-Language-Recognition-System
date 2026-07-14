"""
report.py

Generate a complete Explainability Report
for the trained CNN.

Author: Argha Sarkar Project
"""

from pathlib import Path
from datetime import datetime
import json


class EvaluationReport:
    """
    Saves evaluation outputs:
    - metrics.json
    - classification_report.txt
    - summary.txt
    """

    def __init__(self):
        self.output_dir = Path("reports/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: dict):
        """Save metrics dictionary as JSON."""
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(
                {k: float(v) if hasattr(v, "item") else v
                 for k, v in metrics.items()},
                f,
                indent=4,
            )
        print(f"Metrics saved to:\n{path}")
        return path

    def save_classification_report(self, report: str):
        """Save the sklearn classification report as text."""
        path = self.output_dir / "classification_report.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Classification report saved to:\n{path}")
        return path

    def save_summary(self, metrics: dict):
        """Save a human-readable summary."""
        path = self.output_dir / "summary.txt"
        lines = [
            "=" * 50,
            "Evaluation Summary",
            f"Generated: {datetime.now()}",
            "=" * 50,
        ]
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = float(v)
            lines.append(f"{k:<20}: {v}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Summary saved to:\n{path}")
        return path


class ExplainabilityReport:

    """
    Generates a Markdown report summarizing
    the explainability outputs.
    """

    def __init__(self):

        self.output_dir = Path(
            "reports/explainability"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate(

        self,

        metrics: dict,

        confidence: dict,

        model_name="ASL CNN Baseline",

    ):

        report_path = (
            self.output_dir /
            "explainability_report.md"
        )

        report = f"""# Explainability Report

Generated on:

{datetime.now()}

---

# Model

{model_name}

---

# Evaluation Metrics

| Metric | Value |
|---------|------:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 Score | {metrics["f1"]:.4f} |

---

# Confidence Analysis

| Metric | Value |
|---------|------:|
| Mean Confidence | {confidence["Mean Confidence"]:.4f} |
| Minimum Confidence | {confidence["Minimum Confidence"]:.4f} |
| Maximum Confidence | {confidence["Maximum Confidence"]:.4f} |
| Median Confidence | {confidence["Median Confidence"]:.4f} |

---

# Generated Artifacts

## Evaluation

- metrics.json

- classification_report.txt

- confusion_matrix.png

---

## Explainability

- confidence_distribution.png

- misclassified_images.png

- gradcam/

- activation_maps/

- feature_maps/

---

# Interpretation

## Model Performance

The model achieved high overall performance
according to the evaluation metrics.

## Confidence

Review low-confidence samples before deployment.

## Misclassified Images

Inspect recurring failure patterns.

## Grad-CAM

Verify the CNN focuses on the hand
rather than the background.

## Feature Maps

Ensure filters learn meaningful features.

## Activation Maps

Check feature extraction across layers.

---

# Deployment Recommendation

Before production deployment:

- Review low-confidence samples.

- Review misclassified images.

- Verify Grad-CAM outputs.

- Compare against previous model versions.

- Archive this report.

"""

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

        return report_path