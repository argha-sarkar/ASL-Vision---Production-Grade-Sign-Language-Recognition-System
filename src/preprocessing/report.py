"""
report.py

Generate reports and visualizations for Optuna optimization.

Responsibilities
----------------
1. Optimization History Plot
2. Parameter Importance Plot
3. Parallel Coordinate Plot
4. Slice Plot
5. Contour Plot
6. Save Best Parameters
7. Generate Markdown Report

Author:
Argha Sarkar Project
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (plot_contour,
                                             plot_optimization_history,
                                             plot_parallel_coordinate,
                                             plot_param_importances,
                                             plot_slice)


class OptimizationReport:
    """
    Generate Optuna reports.
    """

    def __init__(self):

        self.output_dir = Path("reports/optimization")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Save Best Parameters
    # ---------------------------------------------------------

    def save_best_parameters(
        self,
        study,
    ):

        save_path = self.output_dir / "best_parameters.json"

        data = {
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "parameters": study.best_params,
        }

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )

    # ---------------------------------------------------------
    # Optimization History
    # ---------------------------------------------------------

    def optimization_history(
        self,
        study,
    ):

        ax = plot_optimization_history(study)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "optimization_history.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------------------
    # Parameter Importance
    # ---------------------------------------------------------

    def parameter_importance(
        self,
        study,
    ):

        ax = plot_param_importances(study)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "parameter_importance.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------------------
    # Parallel Coordinate
    # ---------------------------------------------------------

    def parallel_coordinate(
        self,
        study,
    ):

        ax = plot_parallel_coordinate(study)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "parallel_coordinate.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------------------
    # Slice Plot
    # ---------------------------------------------------------

    def slice_plot(
        self,
        study,
    ):

        ax = plot_slice(study)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "slice_plot.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ---------------------------------------------------------
    # Contour Plot
    # ---------------------------------------------------------

    def contour_plot(
        self,
        study,
    ):

        try:

            ax = plot_contour(study)

            plt.tight_layout()

            plt.savefig(
                self.output_dir / "contour_plot.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

        except Exception:

            print("Contour plot requires at least two optimized parameters.")

    # ---------------------------------------------------------
    # Markdown Report
    # ---------------------------------------------------------

    def markdown_report(
        self,
        study,
    ):

        report_path = self.output_dir / "optimization_report.md"

        markdown = f"""# Hyperparameter Optimization Report

Generated:

{datetime.now()}

---

## Study Name

{study.study_name}

---

## Best Trial

{study.best_trial.number}

---

## Best Validation Score

{study.best_value:.5f}

---

## Best Hyperparameters

"""

        for key, value in study.best_params.items():

            markdown += f"- **{key}** : {value}\n"

        markdown += """

---

## Generated Files

- best_parameters.json

- optimization_history.png

- parameter_importance.png

- parallel_coordinate.png

- slice_plot.png

- contour_plot.png

---

## Conclusion

The best hyperparameter configuration has been saved
and can now be used for production model training.

"""

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)

    # ---------------------------------------------------------
    # Complete Report
    # ---------------------------------------------------------

    def generate(
        self,
        study,
    ):

        print("\nGenerating Optimization Report...\n")

        self.save_best_parameters(study)

        self.optimization_history(study)

        self.parameter_importance(study)

        self.parallel_coordinate(study)

        self.slice_plot(study)

        self.contour_plot(study)

        self.markdown_report(study)

        print("Optimization Report Generated Successfully.")
