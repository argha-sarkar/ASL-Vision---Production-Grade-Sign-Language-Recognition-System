"""
compiler.py

Model compilation utilities.
"""

import tensorflow as tf


class ModelCompiler:
    """
    Compile TensorFlow models using
    production-ready defaults.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
    ):

        self.learning_rate = learning_rate

    def compile(self, model):

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model
