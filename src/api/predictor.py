"""
predictor.py

FastAPI Prediction Engine

Author: Argha Sarkar Project
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.api.config import settings


class APIPredictor:
    """
    Prediction Engine for FastAPI.
    """

    def __init__(self):

        self.model = self._load_model()

        self.labels = self._load_labels()

    # ---------------------------------------------------------

    def _load_model(self):

        model_path = Path(settings.MODEL_PATH)

        if not model_path.exists():

            raise FileNotFoundError(f"Model not found: {model_path}")

        return tf.keras.models.load_model(str(model_path))

    # ---------------------------------------------------------

    def _load_labels(self):

        label_path = Path(settings.LABELS_PATH)

        if not label_path.exists():

            return None

        with open(
            label_path,
            "rb",
        ) as file:

            return pickle.load(file)

    # ---------------------------------------------------------

    def preprocess(
        self,
        image: np.ndarray,
    ):

        image = cv2.resize(
            image,
            (28, 28),
        )

        if image.ndim == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        image = image.astype("float32")

        image /= 255.0

        image = np.expand_dims(
            image,
            axis=-1,
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        return image

    # ---------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
    ):

        image = self.preprocess(image)

        probabilities = self.model.predict(
            image,
            verbose=0,
        )[0]

        prediction = int(np.argmax(probabilities))

        confidence = float(probabilities[prediction])

        if self.labels is not None:

            try:

                label = self.labels.inverse_transform([prediction])[0]

            except Exception:

                label = str(prediction)

        else:

            label = str(prediction)

        return {
            "prediction": label,
            "class_index": prediction,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
        }

    # ---------------------------------------------------------

    def predict_file(
        self,
        image_path,
    ):

        image = cv2.imread(str(image_path))

        if image is None:

            raise ValueError("Unable to read image.")

        return self.predict(image)

    # ---------------------------------------------------------

    def predict_folder(
        self,
        folder_path,
    ):

        folder = Path(folder_path)

        if not folder.exists():

            raise FileNotFoundError(folder)

        results = []

        for image_path in sorted(folder.iterdir()):

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:

                continue

            prediction = self.predict_file(image_path)

            prediction["filename"] = image_path.name

            results.append(prediction)

        return results

    # ---------------------------------------------------------

    def health(self):

        return {
            "model_loaded": self.model is not None,
            "num_classes": self.model.output_shape[-1],
            "input_shape": self.model.input_shape,
        }
