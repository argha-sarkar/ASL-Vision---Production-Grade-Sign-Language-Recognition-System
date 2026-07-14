"""
test_model.py

Production Model Tests

Author: Argha Sarkar Project
"""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.cnn import CNNModel

MODEL_PATH = Path("models/best_model.keras")


# ---------------------------------------------------------
# Model Creation
# ---------------------------------------------------------


def test_model_creation():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    assert isinstance(
        model,
        tf.keras.Model,
    )


# ---------------------------------------------------------
# Model Compilation
# ---------------------------------------------------------


def test_model_compile():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    assert model.optimizer is not None


# ---------------------------------------------------------
# Model Prediction
# ---------------------------------------------------------


def test_prediction_shape():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    sample = np.random.rand(
        1,
        28,
        28,
        1,
    ).astype(np.float32)

    prediction = model.predict(
        sample,
        verbose=0,
    )

    assert prediction.shape == (
        1,
        24,
    )


# ---------------------------------------------------------
# Output Probability
# ---------------------------------------------------------


def test_probability_sum():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    sample = np.random.rand(
        1,
        28,
        28,
        1,
    ).astype(np.float32)

    prediction = model.predict(
        sample,
        verbose=0,
    )

    probability = np.sum(prediction)

    assert np.isclose(
        probability,
        1.0,
        atol=1e-5,
    )


# ---------------------------------------------------------
# Model Save
# ---------------------------------------------------------


def test_model_save(
    tmp_path,
):

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    save_path = tmp_path / "model.keras"

    model.save(save_path)

    assert save_path.exists()


# ---------------------------------------------------------
# Model Load
# ---------------------------------------------------------


def test_model_load():

    if not MODEL_PATH.exists():

        pytest.skip("Trained model not available.")

    model = tf.keras.models.load_model(MODEL_PATH)

    assert isinstance(
        model,
        tf.keras.Model,
    )


# ---------------------------------------------------------
# Input Shape
# ---------------------------------------------------------


def test_input_shape():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    assert model.input_shape == (
        None,
        28,
        28,
        1,
    )


# ---------------------------------------------------------
# Output Classes
# ---------------------------------------------------------


def test_output_classes():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    assert model.output_shape[-1] == 24


# ---------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------


def test_parameter_count():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    params = model.count_params()

    assert params > 0


# ---------------------------------------------------------
# Inference Speed
# ---------------------------------------------------------


def test_inference_speed():

    import time

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    sample = np.random.rand(
        1,
        28,
        28,
        1,
    ).astype(np.float32)

    start = time.time()

    model.predict(
        sample,
        verbose=0,
    )

    elapsed = time.time() - start

    assert elapsed < 2.0


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------


def test_summary():

    model = CNNModel(
        input_shape=(28, 28, 1),
        num_classes=24,
    ).build()

    model.summary()


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
