"""
train.py

Production Training Script

Author: Argha Sarkar Project
"""

from pathlib import Path
import argparse

import tensorflow as tf

from src.data.loader import DataLoader
from src.data.pipeline import DataPipeline

from src.models.cnn import CNNModel
from src.models.trainer import ModelTrainer

from src.mlops.mlflow_logger import MLflowLogger


class TrainingPipeline:
    """
    End-to-End Training Pipeline.
    """

    def __init__(self, args):

        self.args = args

        self.mlflow = MLflowLogger(
            experiment_name="ASL-Vision"
        )

    # -----------------------------------------------------

    def load_data(self):

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

        return (

            train_dataset,

            validation_dataset,

            test_dataset,

        )

    # -----------------------------------------------------

    def build_model(self):

        print("=" * 70)
        print("Building Model")
        print("=" * 70)

        builder = CNNModel(

            input_shape=(28, 28, 1),

            num_classes=24,

        )

        model = builder.build()

        model.compile(

            optimizer=tf.keras.optimizers.Adam(

                learning_rate=self.args.learning_rate,

            ),

            loss="sparse_categorical_crossentropy",

            metrics=[

                "accuracy",

            ],

        )

        return model

    # -----------------------------------------------------

    def train(

        self,

        model,

        train_dataset,

        validation_dataset,

    ):

        trainer = ModelTrainer(

            model=model,

            train_dataset=train_dataset,

            validation_dataset=validation_dataset,

            epochs=self.args.epochs,

        )

        history = trainer.train()

        trainer.save_model()

        return history

    # -----------------------------------------------------

    def run(self):

        self.mlflow.start_run(

            run_name="training_pipeline"

        )

        (

            train_dataset,

            validation_dataset,

            test_dataset,

        ) = self.load_data()

        model = self.build_model()

        history = self.train(

            model,

            train_dataset,

            validation_dataset,

        )

        self.mlflow.log_params(

            {

                "epochs": self.args.epochs,

                "batch_size": self.args.batch_size,

                "learning_rate": self.args.learning_rate,

            }

        )

        self.mlflow.log_history(

            history

        )

        self.mlflow.log_model(

            model

        )

        self.mlflow.end_run()

        print()

        print("=" * 70)

        print("TRAINING COMPLETED")

        print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--train_csv",

        type=str,

        default="data/raw/sign_mnist_train.csv",

    )

    parser.add_argument(

        "--test_csv",

        type=str,

        default="data/raw/sign_mnist_test.csv",

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=30,

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=64,

    )

    parser.add_argument(

        "--learning_rate",

        type=float,

        default=0.001,

    )

    return parser.parse_args()


if __name__ == "__main__":

    arguments = parse_arguments()

    pipeline = TrainingPipeline(arguments)

    pipeline.run()