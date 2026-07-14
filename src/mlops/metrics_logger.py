"""
metrics_logger.py

Production MLflow Metrics Logger

Author: Argha Sarkar Project
"""

import mlflow


class MetricsLogger:
    """
    Log training and evaluation metrics to MLflow.
    """

    def __init__(
        self,
        tracking_uri="mlruns",
    ):

        mlflow.set_tracking_uri(
            tracking_uri
        )

    # ---------------------------------------------------------

    def log_metric(
        self,
        name,
        value,
        step=None,
    ):

        if step is None:

            mlflow.log_metric(
                name,
                float(value),
            )

        else:

            mlflow.log_metric(

                name,

                float(value),

                step=step,

            )

    # ---------------------------------------------------------

    def log_metrics(
        self,
        metrics,
        step=None,
    ):

        for key, value in metrics.items():

            self.log_metric(

                key,

                value,

                step,

            )

    # ---------------------------------------------------------

    def log_training_history(
        self,
        history,
    ):

        history_dict = history.history

        epochs = len(
            history_dict[
                "loss"
            ]
        )

        for epoch in range(
            epochs
        ):

            for metric, values in history_dict.items():

                mlflow.log_metric(

                    metric,

                    float(values[epoch]),

                    step=epoch,

                )

    # ---------------------------------------------------------

    def log_evaluation_metrics(
        self,
        accuracy,
        precision,
        recall,
        f1_score,
        loss=None,
    ):

        metrics = {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1_score,

        }

        if loss is not None:

            metrics["loss"] = loss

        self.log_metrics(
            metrics
        )

    # ---------------------------------------------------------

    def log_confusion_statistics(
        self,
        tp,
        tn,
        fp,
        fn,
    ):

        mlflow.log_metric(
            "true_positive",
            tp,
        )

        mlflow.log_metric(
            "true_negative",
            tn,
        )

        mlflow.log_metric(
            "false_positive",
            fp,
        )

        mlflow.log_metric(
            "false_negative",
            fn,
        )

    # ---------------------------------------------------------

    def log_prediction_statistics(
        self,
        total_predictions,
        correct_predictions,
        incorrect_predictions,
    ):

        accuracy = (
            correct_predictions /
            total_predictions
        )

        mlflow.log_metric(
            "total_predictions",
            total_predictions,
        )

        mlflow.log_metric(
            "correct_predictions",
            correct_predictions,
        )

        mlflow.log_metric(
            "incorrect_predictions",
            incorrect_predictions,
        )

        mlflow.log_metric(
            "prediction_accuracy",
            accuracy,
        )

    # ---------------------------------------------------------

    def log_confidence_statistics(
        self,
        confidence_scores,
    ):

        confidence_scores = list(
            confidence_scores
        )

        if len(
            confidence_scores
        ) == 0:

            return

        mlflow.log_metric(

            "confidence_mean",

            sum(confidence_scores)
            / len(confidence_scores),

        )

        mlflow.log_metric(

            "confidence_min",

            min(confidence_scores),

        )

        mlflow.log_metric(

            "confidence_max",

            max(confidence_scores),

        )

    # ---------------------------------------------------------

    def log_learning_rate(
        self,
        learning_rate,
        epoch,
    ):

        mlflow.log_metric(

            "learning_rate",

            learning_rate,

            step=epoch,

        )

    # ---------------------------------------------------------

    def log_epoch_summary(
        self,
        epoch,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy,
    ):

        mlflow.log_metric(

            "train_loss",

            train_loss,

            step=epoch,

        )

        mlflow.log_metric(

            "train_accuracy",

            train_accuracy,

            step=epoch,

        )

        mlflow.log_metric(

            "validation_loss",

            validation_loss,

            step=epoch,

        )

        mlflow.log_metric(

            "validation_accuracy",

            validation_accuracy,

            step=epoch,

        )

    # ---------------------------------------------------------

    def log_test_metrics(
        self,
        metrics,
    ):

        for key, value in metrics.items():

            mlflow.log_metric(

                f"test_{key}",

                float(value),

            )


if __name__ == "__main__":

    logger = MetricsLogger()

    logger.log_metric(
        "accuracy",
        0.985,
    )

    logger.log_metrics(

        {

            "precision": 0.981,

            "recall": 0.982,

            "f1_score": 0.981,

        }

    )