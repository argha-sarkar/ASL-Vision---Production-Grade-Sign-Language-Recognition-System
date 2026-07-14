"""
predictor.py

Load trained model and generate predictions.
"""

import tensorflow as tf
import numpy as np


class Predictor:

    def __init__(self, model_path="models/best_model.keras"):

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, images):

        probabilities = self.model.predict(
            images,
            verbose=1,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        return predictions, probabilities