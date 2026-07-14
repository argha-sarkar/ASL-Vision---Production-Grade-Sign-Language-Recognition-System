"""
cnn.py

Production-grade CNN architecture for
American Sign Language Recognition.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
)


class CNNModel:
    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=24,
        filters=None,
        dropout_rate=0.25,
        dense_units=128,
        learning_rate=0.001,
        optimizer="adam",
        batch_size=64,
    ):
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self) -> Model:
        """
        Build and return the CNN model.
        """

        inputs = Input(
            shape=self.input_shape,
            name="input_image"
        )

        # ----------------------------------------
        # Block 1
        # ----------------------------------------

        x = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv1"
        )(inputs)

        x = BatchNormalization(
            name="bn1"
        )(x)

        x = Activation(
            "relu",
            name="relu1"
        )(x)

        x = MaxPooling2D(
            pool_size=(2, 2),
            name="pool1"
        )(x)

        x = Dropout(
            0.25,
            name="dropout1"
        )(x)

        # ----------------------------------------
        # Block 2
        # ----------------------------------------

        x = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2"
        )(x)

        x = BatchNormalization(
            name="bn2"
        )(x)

        x = Activation(
            "relu",
            name="relu2"
        )(x)

        x = MaxPooling2D(
            pool_size=(2, 2),
            name="pool2"
        )(x)

        x = Dropout(
            0.30,
            name="dropout2"
        )(x)

        # ----------------------------------------
        # Block 3
        # ----------------------------------------

        x = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv3"
        )(x)

        x = BatchNormalization(
            name="bn3"
        )(x)

        x = Activation(
            "relu",
            name="relu3"
        )(x)

        x = GlobalAveragePooling2D(
            name="gap"
        )(x)

        # ----------------------------------------
        # Classification Head
        # ----------------------------------------

        x = Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense1"
        )(x)

        x = Dropout(
            0.40,
            name="dropout3"
        )(x)

        outputs = Dense(
            self.num_classes,
            activation="softmax",
            name="predictions"
        )(x)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name="ASL_CNN_Baseline"
        )

        return model
