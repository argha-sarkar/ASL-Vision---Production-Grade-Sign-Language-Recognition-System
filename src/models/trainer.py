"""
trainer.py

Production-grade training engine for
American Sign Language Recognition.

Responsibilities
----------------
1. Build the CNN model
2. Compile the model
3. Load callbacks
4. Train the model
5. Save training history
"""

from tensorflow.python.platform import self_check
from pathlib import Path
import json
from typing import Dict

import tensorflow as tf

from src.models.cnn import CNNModel
from src.models.compiler import ModelCompiler
from src.models.callbacks import CallbackManager


class ModelTrainer:
    """
    Handles complete model training.
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=24,
        learning_rate=0.001,
        epochs=50,
    ):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.history_dir = Path("reports")
        self.history_dir.mkdir(exist_ok=True)

    def build_model(self):

        model = CNNModel(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        ).build()

        model = ModelCompiler(
            learning_rate=self.learning_rate
        ).compile(model)

        return model

    def train(
        self,
        train_dataset,
        validation_dataset,
    ):

        print("=" * 70)
        print("Building CNN Model")
        print("=" * 70)

        model = self.build_model()

        print()
        model.summary()

        print()
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)

        callbacks = CallbackManager().build()

        history = model.fit(

            train_dataset,

            validation_data=validation_dataset,

            epochs=self.epochs,

            callbacks=callbacks,

            verbose=1,

        )

        self.save_history(history.history)

        print()
        print("=" * 70)
        print("Training Completed Successfully")
        print("=" * 70)

        return model, history

    def save_history(
        self,
        history: Dict,
    ):

        history_path = self.history_dir / "training_history.json"

        with open(history_path, "w") as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        print(f"\nTraining history saved to:\n{history_path}")