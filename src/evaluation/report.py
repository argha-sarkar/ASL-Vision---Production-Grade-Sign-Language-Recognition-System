"""
report.py

Generate evaluation reports.
"""

from pathlib import Path
import json


class EvaluationReport:

    def __init__(self):

        self.output_dir = Path("reports/evaluation")
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_metrics(
        self,
        metrics: dict,
    ):

        metrics_path = self.output_dir / "metrics.json"

        with open(
            metrics_path,
            "w",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

        return metrics_path

    def save_classification_report(
        self,
        report: str,
    ):

        report_path = (
            self.output_dir
            / "classification_report.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

        return report_path

    def save_summary(
        self,
        metrics: dict,
    ):

        summary_path = (
            self.output_dir
            / "summary.md"
        )

        markdown = f"""
# Model Evaluation Summary

## Accuracy

{metrics["accuracy"]:.4f}

## Precision

{metrics["precision"]:.4f}

## Recall

{metrics["recall"]:.4f}

## F1 Score

{metrics["f1"]:.4f}
"""

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

        return summary_path