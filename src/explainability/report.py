"""
report.py

Generates a structured explainability report
summarizing model evaluation metrics and confidence.
"""

import json
from pathlib import Path
from typing import Dict


class ExplainabilityReport:
    """
    Saves a human-readable explainability report.
    """

    def __init__(self):
        self.output_dir = Path("reports/explainability")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: Dict,
        confidence: Dict,
        model_name: str = "ASL CNN",
    ) -> Path:
        """
        Generate and save the explainability report.

        Returns
        -------
        Path to the saved report.
        """

        report = {
            "model_name": model_name,
            "metrics": {
                k: float(v) if hasattr(v, "item") else v
                for k, v in metrics.items()
            },
            "confidence_summary": confidence,
        }

        report_path = self.output_dir / "explainability_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, default=str)

        print(f"Explainability report saved to:\n{report_path}")

        return report_path
