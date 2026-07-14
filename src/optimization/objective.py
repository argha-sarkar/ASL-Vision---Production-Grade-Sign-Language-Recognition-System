"""
objective.py

Objective function for Optuna.

Each trial:

1. Sample hyperparameters
2. Build CNN
3. Compile model
4. Train model
5. Return validation accuracy

Author:
Argha Sarkar Project
"""

from typing import Tuple

import optuna

from src.models.cnn import CNNModel
from src.models.compiler import ModelCompiler
from src.models.callbacks import CallbackManager

from src.optimization.search_space import SearchSpace


class Objective:

    """
    Objective function used by Optuna.
    """

    def __init__(

        self,

        train_dataset,

        validation_dataset,

        input_shape,

        num_classes,

        epochs=20,

    ):

        self.train_dataset = train_dataset

        self.validation_dataset = validation_dataset

        self.input_shape = input_shape

        self.num_classes = num_classes

        self.epochs = epochs

    def __call__(
        self,
        trial: optuna.Trial,
    ) -> float:

        params = SearchSpace.build(
            trial
        )

        model = CNNModel(

            input_shape=self.input_shape,

            num_classes=self.num_classes,

            filters=params.filters,

            dropout_rate=params.dropout_rate,

            dense_units=params.dense_units,

        ).build()

        model = ModelCompiler(

            learning_rate=params.learning_rate,

            optimizer_name=params.optimizer,

        ).compile(model)

        callbacks = CallbackManager().build()

        history = model.fit(

            self.train_dataset,

            validation_data=self.validation_dataset,

            epochs=self.epochs,

            callbacks=callbacks,

            verbose=0,

        )

        validation_accuracy = max(

            history.history["val_accuracy"]

        )

        return validation_accuracy