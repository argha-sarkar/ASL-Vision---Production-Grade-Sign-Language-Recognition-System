"""
report.py

Generate a complete Transfer Learning Report.

Author: Argha Sarkar Project
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


class TransferLearningReport:
    """
    Generate Markdown report for all Transfer Learning models.
    """

    def __init__(
        self,
        report_dir="reports/transfer_learning",
    ):

        self.report_dir = Path(report_dir)

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------

    def load_results(self):

        csv_path = self.report_dir / "model_comparison.csv"

        if not csv_path.exists():

            raise FileNotFoundError(f"{csv_path} not found.")

        return pd.read_csv(csv_path)

    # ---------------------------------------------------------

    def best_model(self, dataframe):

        return dataframe.loc[dataframe["Accuracy"].idxmax()]

    # ---------------------------------------------------------

    def create_table(self, dataframe):

        markdown = dataframe.to_markdown(index=False)

        return markdown

    # ---------------------------------------------------------

    def recommendations(
        self,
        best_model,
    ):

        text = f"""
## Recommendation

Based on the experimental results,
**{best_model['Model']}**
achieved the highest validation accuracy.

Recommended for production deployment.

Further improvements:

- Fine-tuning
- Hyperparameter Optimization
- Test-Time Augmentation
- Model Quantization
- ONNX Export
- TensorRT Optimization

"""

        return text

    # ---------------------------------------------------------

    def generated_files(self):

        return """
## Generated Files

- model_comparison.csv
- model_comparison.json
- accuracy_comparison.png
- f1_comparison.png
- metrics.json
- classification_report.txt
- confusion_matrix.png
- GradCAM Images
- Misclassified Images
"""

    # ---------------------------------------------------------

    def generate(self):

        dataframe = self.load_results()

        best = self.best_model(dataframe)

        markdown = f"""# Transfer Learning Report

Generated:

{datetime.now()}

---

# Project

American Sign Language Recognition

---

# Compared Models

{self.create_table(dataframe)}

---

# Best Model

| Item | Value |
|------|-------|
| Model | {best['Model']} |
| Accuracy | {best['Accuracy']:.4f} |
| Precision | {best['Precision']:.4f} |
| Recall | {best['Recall']:.4f} |
| F1 Score | {best['F1 Score']:.4f} |

---

# Model Ranking

"""

        ranking = dataframe.sort_values(
            by="Accuracy",
            ascending=False,
        )

        for i, row in ranking.iterrows():

            markdown += f"{i+1}. " f"{row['Model']} " f"({row['Accuracy']:.4f})\n"

        markdown += "\n---\n"

        markdown += self.generated_files()

        markdown += "\n---\n"

        markdown += self.recommendations(best)

        report_path = self.report_dir / "transfer_learning_report.md"

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

        print("\n" + "=" * 70)
        print("TRANSFER LEARNING REPORT")
        print("=" * 70)
        print(f"Saved : {report_path}")

        return report_path


if __name__ == "__main__":

    report = TransferLearningReport()

    report.generate()
