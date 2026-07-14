"""
optimizer.py

Production-grade Optuna optimizer for ASL Vision.

Responsibilities
----------------
1. Create Optuna study
2. Optimize hyperparameters
3. Save best model parameters
4. Save optimization study
5. Display optimization summary

Author:
Argha Sarkar Project
"""

import json
from pathlib import Path

import optuna

from src.optimization.objective import Objective
from src.optimization.report import OptimizationReport


class HyperparameterOptimizer:
    """
    Runs Optuna hyperparameter optimization.
    """

    def __init__(
        self,
        train_dataset,
        validation_dataset,
        input_shape,
        num_classes,
        epochs=20,
        n_trials=30,
        direction="maximize",
        study_name="asl_cnn_optimization",
        storage=None,
    ):

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.epochs = epochs
        self.n_trials = n_trials

        self.direction = direction
        self.study_name = study_name
        self.storage = storage

        self.output_dir = Path("reports/optimization")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_study(self):
        """
        Create or load Optuna study.
        """

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True,
        )

        return study

    def optimize(self):
        """
        Run optimization.
        """

        study = self.create_study()

        objective = Objective(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            epochs=self.epochs,
        )

        print("\n" + "=" * 70)
        print("Starting Hyperparameter Optimization")
        print("=" * 70)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        report = OptimizationReport()

        report.generate(study)

        self.save_best_parameters(study)

        self.save_trials(study)

        self.print_summary(study)

        self.save_trials(study)

        self.print_summary(study)

        return study

    def save_best_parameters(
        self,
        study,
    ):
        """
        Save best hyperparameters.
        """

        save_path = self.output_dir / "best_parameters.json"

        result = {
            "best_score": study.best_value,
            "best_trial": study.best_trial.number,
            "parameters": study.best_params,
        }

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
            )

    def save_trials(
        self,
        study,
    ):
        """
        Save all trials.
        """

        dataframe = study.trials_dataframe()

        csv_path = self.output_dir / "trials.csv"

        dataframe.to_csv(
            csv_path,
            index=False,
        )

    def print_summary(
        self,
        study,
    ):

        print()

        print("=" * 70)
        print("Optimization Completed")
        print("=" * 70)

        print()

        print(f"Best Trial : {study.best_trial.number}")

        print(f"Best Score : {study.best_value:.5f}")

        print()

        print("Best Parameters")

        print("-" * 70)

        for key, value in study.best_params.items():

            print(f"{key:<20}: {value}")

        print()

        print(f"Reports Saved : {self.output_dir}")
