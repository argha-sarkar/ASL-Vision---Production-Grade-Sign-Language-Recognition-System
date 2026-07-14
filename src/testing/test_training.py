"""
test_training.py

Production Training Tests

Author: Argha Sarkar Project
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.cnn import CNNModel
from src.models.trainer import ModelTrainer


# ---------------------------------------------------------
# Dummy Dataset
# ---------------------------------------------------------

def create_dataset():

    x = np.random.rand(

        64,

        28,

        28,

        1,

    ).astype(np.float32)

    y = np.random.randint(

        0,

        24,

        64,

    )

    dataset = tf.data.Dataset.from_tensor_slices(

        (x, y)

    )

    dataset = dataset.batch(16)

    return dataset


# ---------------------------------------------------------
# Dummy Model
# ---------------------------------------------------------

def create_model():

    builder = CNNModel(

        input_shape=(28, 28, 1),

        num_classes=24,

    )

    model = builder.build()

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )

    return model


# ---------------------------------------------------------
# Dataset Test
# ---------------------------------------------------------

def test_dataset():

    dataset = create_dataset()

    assert dataset is not None


# ---------------------------------------------------------
# Model Test
# ---------------------------------------------------------

def test_model():

    model = create_model()

    assert isinstance(

        model,

        tf.keras.Model,

    )


# ---------------------------------------------------------
# Trainer Initialization
# ---------------------------------------------------------

def test_trainer():

    model = create_model()

    train = create_dataset()

    validation = create_dataset()

    trainer = ModelTrainer(

        model=model,

        train_dataset=train,

        validation_dataset=validation,

        epochs=1,

    )

    assert trainer is not None


# ---------------------------------------------------------
# Single Epoch Training
# ---------------------------------------------------------

def test_training():

    model = create_model()

    train = create_dataset()

    validation = create_dataset()

    trainer = ModelTrainer(

        model=model,

        train_dataset=train,

        validation_dataset=validation,

        epochs=1,

    )

    history = trainer.train()

    assert history is not None


# ---------------------------------------------------------
# History Test
# ---------------------------------------------------------

def test_history():

    model = create_model()

    train = create_dataset()

    validation = create_dataset()

    trainer = ModelTrainer(

        model=model,

        train_dataset=train,

        validation_dataset=validation,

        epochs=1,

    )

    history = trainer.train()

    assert "loss" in history.history


# ---------------------------------------------------------
# Accuracy Metric
# ---------------------------------------------------------

def test_accuracy():

    model = create_model()

    train = create_dataset()

    validation = create_dataset()

    trainer = ModelTrainer(

        model=model,

        train_dataset=train,

        validation_dataset=validation,

        epochs=1,

    )

    history = trainer.train()

    keys = history.history.keys()

    assert any(

        "accuracy" in key

        for key in keys

    )


# ---------------------------------------------------------
# Model Save
# ---------------------------------------------------------

def test_save_model():

    model = create_model()

    temp_dir = tempfile.mkdtemp()

    save_path = Path(temp_dir) / "test.keras"

    model.save(

        save_path

    )

    assert save_path.exists()


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def test_load_model():

    model = create_model()

    temp_dir = tempfile.mkdtemp()

    save_path = Path(temp_dir) / "model.keras"

    model.save(

        save_path

    )

    loaded = tf.keras.models.load_model(

        save_path

    )

    assert isinstance(

        loaded,

        tf.keras.Model,

    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def test_prediction():

    model = create_model()

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
# Evaluation
# ---------------------------------------------------------

def test_evaluation():

    model = create_model()

    dataset = create_dataset()

    result = model.evaluate(

        dataset,

        verbose=0,

    )

    assert len(result) >= 2


# ---------------------------------------------------------
# Parameter Count
# ---------------------------------------------------------

def test_parameters():

    model = create_model()

    params = model.count_params()

    assert params > 1000


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