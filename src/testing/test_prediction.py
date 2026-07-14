"""
test_prediction.py

Production Prediction Tests

Author: Argha Sarkar Project
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.api.predictor import APIPredictor

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------


def create_test_image():

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    temp_dir = tempfile.mkdtemp()

    image_path = Path(temp_dir) / "sample.png"

    cv2.imwrite(
        str(image_path),
        image,
    )

    return image_path


# ---------------------------------------------------------
# Predictor Initialization
# ---------------------------------------------------------


def test_predictor_initialization():

    predictor = APIPredictor()

    assert predictor is not None


# ---------------------------------------------------------
# Model Loaded
# ---------------------------------------------------------


def test_model_loaded():

    predictor = APIPredictor()

    assert predictor.model is not None


# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------


def test_preprocess():

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    tensor = predictor.preprocess(image)

    assert tensor.shape == (
        1,
        28,
        28,
        1,
    )


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------


def test_predict():

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    result = predictor.predict(image)

    assert "prediction" in result

    assert "confidence" in result

    assert "probabilities" in result


# ---------------------------------------------------------
# Predict File
# ---------------------------------------------------------


def test_predict_file():

    predictor = APIPredictor()

    image_path = create_test_image()

    result = predictor.predict_file(image_path)

    assert "prediction" in result


# ---------------------------------------------------------
# Folder Prediction
# ---------------------------------------------------------


def test_predict_folder():

    predictor = APIPredictor()

    temp_dir = tempfile.mkdtemp()

    folder = Path(temp_dir)

    for i in range(3):

        image = np.random.randint(
            0,
            255,
            (224, 224, 3),
            dtype=np.uint8,
        )

        cv2.imwrite(
            str(folder / f"{i}.png"),
            image,
        )

    results = predictor.predict_folder(folder)

    assert len(results) == 3


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------


def test_confidence_range():

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    result = predictor.predict(image)

    assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------
# Probability Sum
# ---------------------------------------------------------


def test_probability_sum():

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    result = predictor.predict(image)

    total = sum(result["probabilities"])

    assert np.isclose(
        total,
        1.0,
        atol=1e-5,
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------


def test_health():

    predictor = APIPredictor()

    health = predictor.health()

    assert health["model_loaded"]


# ---------------------------------------------------------
# Invalid File
# ---------------------------------------------------------


def test_invalid_file():

    predictor = APIPredictor()

    with pytest.raises(Exception):

        predictor.predict_file("invalid.png")


# ---------------------------------------------------------
# Multiple Predictions
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    range(10),
)
def test_multiple_predictions(
    index,
):

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    result = predictor.predict(image)

    assert result is not None


# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------


def test_prediction_speed():

    import time

    predictor = APIPredictor()

    image = np.random.randint(
        0,
        255,
        (224, 224, 3),
        dtype=np.uint8,
    )

    start = time.time()

    predictor.predict(image)

    elapsed = time.time() - start

    assert elapsed < 2.0


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    pytest.main(
        [
            "-v",
            __file__,
        ]
    )
