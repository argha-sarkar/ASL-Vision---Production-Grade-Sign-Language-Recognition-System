"""
mobilenet.py

MobileNetV3 Small Transfer Learning Model.

Author: Argha Sarkar Project
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small

from src.transfer_learning.base_model import BaseTransferModel


class MobileNetModel(BaseTransferModel):
    """
    MobileNetV3 Small Transfer Learning Model.
    """

    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=24,
        weights="imagenet",
        trainable=False,
        dense_units=256,
        dropout_rate=0.30,
        learning_rate=1e-3,
        optimizer="adam",
        l2_weight=1e-4,
    ):

        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            weights=weights,
            trainable=trainable,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            l2_weight=l2_weight,
        )

    # ---------------------------------------------------------

    def get_backbone(self):

        backbone = MobileNetV3Small(

            include_top=False,

            weights=self.weights,

            input_shape=self.input_shape,

            pooling=None,

        )

        return backbone


if __name__ == "__main__":

    model_builder = MobileNetModel(

        input_shape=(224, 224, 3),

        num_classes=24,

        trainable=False,

    )

    model = model_builder.build()

    model = model_builder.compile(model)

    model.summary()