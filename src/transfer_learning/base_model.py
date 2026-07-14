"""
base_model.py

Base class for Transfer Learning models.

Author: Argha Sarkar Project
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import (
    Adam,
    RMSprop,
    SGD,
)


class BaseTransferModel(ABC):
    """
    Base class for all transfer learning models.
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

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights

        self.trainable = trainable

        self.dense_units = dense_units

        self.dropout_rate = dropout_rate

        self.learning_rate = learning_rate

        self.optimizer_name = optimizer.lower()

        self.l2_weight = l2_weight

    # ---------------------------------------------------------

    @abstractmethod
    def get_backbone(self):
        """
        Return pretrained backbone.
        """
        pass

    # ---------------------------------------------------------

    def freeze_backbone(self, backbone):

        backbone.trainable = self.trainable

        return backbone

    # ---------------------------------------------------------

    def classifier(self, x):

        x = GlobalAveragePooling2D()(x)

        x = BatchNormalization()(x)

        x = Dense(
            self.dense_units,
            activation="relu",
            kernel_regularizer=l2(
                self.l2_weight
            ),
        )(x)

        x = Dropout(
            self.dropout_rate
        )(x)

        outputs = Dense(
            self.num_classes,
            activation="softmax",
        )(x)

        return outputs

    # ---------------------------------------------------------

    def build(self):

        backbone = self.get_backbone()

        backbone = self.freeze_backbone(
            backbone
        )

        inputs = tf.keras.Input(
            shape=self.input_shape
        )

        x = backbone(
            inputs,
            training=False,
        )

        outputs = self.classifier(x)

        model = Model(

            inputs,

            outputs,

            name=self.__class__.__name__,

        )

        return model

    # ---------------------------------------------------------

    def get_optimizer(self):

        if self.optimizer_name == "adam":

            return Adam(
                learning_rate=self.learning_rate
            )

        elif self.optimizer_name == "rmsprop":

            return RMSprop(
                learning_rate=self.learning_rate
            )

        elif self.optimizer_name == "sgd":

            return SGD(
                learning_rate=self.learning_rate,
                momentum=0.9,
            )

        raise ValueError(
            f"Unsupported optimizer: {self.optimizer_name}"
        )

    # ---------------------------------------------------------

    def compile(self, model):

        model.compile(

            optimizer=self.get_optimizer(),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy",
            ],

        )

        return model

    # ---------------------------------------------------------

    def summary(self):

        model = self.build()

        model.summary()

        return model

    # ---------------------------------------------------------

    def unfreeze_top_layers(
        self,
        model,
        num_layers=20,
    ):
        """
        Unfreeze top layers for fine tuning.
        """

        backbone = model.layers[1]

        backbone.trainable = True

        if num_layers < len(backbone.layers):

            for layer in backbone.layers[:-num_layers]:

                layer.trainable = False

        return model

    # ---------------------------------------------------------

    def fine_tune(
        self,
        model,
        learning_rate=1e-5,
    ):

        optimizer = Adam(
            learning_rate=learning_rate
        )

        model.compile(

            optimizer=optimizer,

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy",
            ],

        )

        return model

    # ---------------------------------------------------------

    @staticmethod
    def preprocess_input(images):

        images = tf.cast(
            images,
            tf.float32,
        )

        images /= 255.0

        return images