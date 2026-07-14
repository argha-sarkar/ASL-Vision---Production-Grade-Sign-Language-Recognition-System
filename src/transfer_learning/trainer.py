"""
trainer.py

Production Trainer for Transfer Learning Models.

Author: Argha Sarkar Project
"""

from pathlib import Path

import tensorflow as tf

from src.models.callbacks import CallbackManager


class TransferLearningTrainer:
    """
    Trainer for all transfer learning models.
    """

    def __init__(
        self,
        model,
        train_dataset,
        validation_dataset,
        epochs=20,
        model_name="transfer_model",
    ):

        self.model = model

        self.train_dataset = train_dataset

        self.validation_dataset = validation_dataset

        self.epochs = epochs

        self.model_name = model_name

        self.callbacks = CallbackManager().build()

        self.model_dir = Path("models")

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history = None

    # -----------------------------------------------------

    def train(self):

        print("\n" + "=" * 70)
        print(f"Training : {self.model_name}")
        print("=" * 70)

        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1,
        )

        return self.history

    # -----------------------------------------------------

    def fine_tune(
        self,
        base_model,
        fine_tune_at=100,
        learning_rate=1e-5,
        epochs=10,
    ):

        print("\n" + "=" * 70)
        print("Fine Tuning")
        print("=" * 70)

        base_model.trainable = True

        for layer in base_model.layers[:fine_tune_at]:

            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
            ],
        )

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1,
        )

        return history

    # -----------------------------------------------------

    def save_model(self):

        save_path = self.model_dir / f"{self.model_name}.keras"

        self.model.save(save_path)

        print()

        print(f"Model Saved : {save_path}")

        return save_path

    # -----------------------------------------------------

    def run(self):

        history = self.train()

        model_path = self.save_model()

        return {
            "history": history,
            "model_path": model_path,
        }


if __name__ == "__main__":

    print("TransferLearningTrainer module loaded successfully.")
