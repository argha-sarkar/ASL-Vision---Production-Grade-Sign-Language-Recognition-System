"""
evaluate.py

Production Evaluation Script

Author: Argha Sarkar Project
"""

import argparse

from src.data.loader import DataLoader
from src.data.pipeline import DataPipeline
from src.evaluation.evaluator import ModelEvaluator
from src.mlops.mlflow_logger import MLflowLogger


class EvaluationPipeline:
    """
    End-to-End Evaluation Pipeline.
    """

    def __init__(self, args):

        self.args = args

        self.mlflow = MLflowLogger(experiment_name="ASL-Vision")

    # ---------------------------------------------------------

    def load_dataset(self):

        print("=" * 70)
        print("Loading Dataset")
        print("=" * 70)

        loader = DataLoader(
            train_csv=self.args.train_csv,
            test_csv=self.args.test_csv,
        )

        train_df, test_df = loader.load()

        pipeline = DataPipeline()

        (
            train_dataset,
            validation_dataset,
            test_dataset,
        ) = pipeline.prepare(
            train_df,
            test_df,
        )

        return test_dataset

    # ---------------------------------------------------------

    def evaluate(self):

        dataset = self.load_dataset()

        evaluator = ModelEvaluator(
            model_path=self.args.model,
        )

        self.mlflow.start_run(run_name="evaluation")

        results = evaluator.evaluate(
            dataset,
        )

        self.mlflow.log_metrics(results["metrics"])

        self.mlflow.log_artifact("reports/evaluation")

        self.mlflow.end_run()

        print()

        print("=" * 70)
        print("EVALUATION COMPLETED")
        print("=" * 70)

        return results


# ---------------------------------------------------------
# Command Line Arguments
# ---------------------------------------------------------


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/best_model.keras",
        type=str,
    )

    parser.add_argument(
        "--train_csv",
        default="data/raw/sign_mnist_train.csv",
        type=str,
    )

    parser.add_argument(
        "--test_csv",
        default="data/raw/sign_mnist_test.csv",
        type=str,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    args = parse_arguments()

    pipeline = EvaluationPipeline(args)

    pipeline.evaluate()
