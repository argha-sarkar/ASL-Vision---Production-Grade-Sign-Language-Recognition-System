"""
predictor.py

Production-grade inference engine for ASL Vision.

Responsibilities
----------------
1. Load trained model
2. Predict a single image
3. Predict multiple images
4. Return confidence scores
5. Return Top-K predictions
6. Validate inputs

Author: Argha Sarkar Project
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


class Predictor:
    """
    Production inference engine.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.keras",
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(

                f"Model not found:\n{self.model_path}"

            )

        print("\nLoading trained model...")

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        print("Model loaded successfully.\n")

    def _validate_input(
        self,
        images: np.ndarray,
    ) -> np.ndarray:
        """
        Validate image dimensions.

        Expected:

        Batch:
        (N,28,28,1)

        Single:
        (28,28,1)
        """

        if not isinstance(images, np.ndarray):

            raise TypeError(

                "Input must be a NumPy array."

            )

        if images.ndim == 3:

            images = np.expand_dims(

                images,

                axis=0,

            )

        if images.ndim != 4:

            raise ValueError(

                f"Expected 4 dimensions.\n"

                f"Received {images.shape}"

            )

        return images

    def predict(
        self,
        images: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict one or more images.

        Returns
        -------

        predictions

        probabilities
        """

        images = self._validate_input(
            images
        )

        probabilities = self.model.predict(

            images,

            verbose=0,

        )

        predictions = np.argmax(

            probabilities,

            axis=1,

        )

        return predictions, probabilities

    def confidence(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Maximum probability for each image.
        """

        return np.max(

            probabilities,

            axis=1,

        )

    def top_k(
        self,
        probabilities: np.ndarray,
        k: int = 5,
    ):
        """
        Return Top-K predictions.
        """

        top_indices = np.argsort(

            probabilities,

            axis=1,

        )[:, -k:]

        top_scores = np.take_along_axis(

            probabilities,

            top_indices,

            axis=1,

        )

        return top_indices, top_scores

    def predict_single(
        self,
        image: np.ndarray,
    ):

        predictions, probabilities = self.predict(
            image
        )

        confidence = self.confidence(
            probabilities
        )[0]

        return {

            "prediction": int(
                predictions[0]
            ),

            "confidence": float(
                confidence
            ),

            "probabilities": probabilities[0],

        }