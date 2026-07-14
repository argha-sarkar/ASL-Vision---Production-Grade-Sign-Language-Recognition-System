"""
callbacks.py

Production-ready callback manager.
"""

from pathlib import Path

import tensorflow as tf


class CallbackManager:

    def __init__(self):

        self.model_dir = Path("models")
        self.log_dir = Path("logs")
        self.tensorboard_dir = self.log_dir / "tensorboard"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def build(self):

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(self.log_dir / "training.csv"),
                append=False,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
                profile_batch=0,
            ),
        ]

        return callbacks
