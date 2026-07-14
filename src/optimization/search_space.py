"""
search_space.py

Defines the Optuna search space for
ASL Vision.

Author:
Argha Sarkar Project
"""

from dataclasses import dataclass

import optuna


@dataclass
class HyperParameters:
    """
    Container for all hyperparameters.
    """

    learning_rate: float

    dropout_rate: float

    filters: int

    dense_units: int

    optimizer: str

    batch_size: int


class SearchSpace:
    """
    Defines all Optuna search parameters.
    """

    @staticmethod
    def build(
        trial: optuna.Trial,
    ) -> HyperParameters:

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-2,
            log=True,
        )

        dropout_rate = trial.suggest_float(
            "dropout_rate",
            0.20,
            0.60,
        )

        filters = trial.suggest_categorical(
            "filters",
            [
                32,
                64,
                128,
            ],
        )

        dense_units = trial.suggest_categorical(
            "dense_units",
            [
                64,
                128,
                256,
                512,
            ],
        )

        optimizer = trial.suggest_categorical(
            "optimizer",
            [
                "Adam",
                "RMSprop",
            ],
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            [
                32,
                64,
                128,
            ],
        )

        return HyperParameters(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            filters=filters,
            dense_units=dense_units,
            optimizer=optimizer,
            batch_size=batch_size,
        )
