"""
feature_maps.py

Visualize CNN feature maps.

This module automatically detects all Conv2D layers
and saves their feature maps for inspection.

Author: Argha Sarkar Project
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class FeatureMapVisualizer:
    """
    Visualize feature maps from every Conv2D layer.
    """

    def __init__(self, model):

        self.model = model

        self.output_dir = Path("reports/explainability/feature_maps")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _get_conv_layers(self):
        """
        Return all Conv2D layer outputs.
        """

        conv_outputs = []

        layer_names = []

        for layer in self.model.layers:

            if isinstance(layer, tf.keras.layers.Conv2D):

                conv_outputs.append(layer.output)

                layer_names.append(layer.name)

        return conv_outputs, layer_names

    def _activation_model(self):
        """
        Create a model that outputs feature maps.
        """

        outputs, names = self._get_conv_layers()

        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=outputs,
        )

        return activation_model, names

    def generate(self, image, max_filters=16):
        """
        Generate feature maps for one image.

        Parameters
        ----------
        image : np.ndarray

            Shape:
            (28,28,1)

        max_filters : int

            Number of filters to display.
        """

        if image.ndim == 3:

            image = np.expand_dims(
                image,
                axis=0,
            )

        activation_model, layer_names = self._activation_model()

        feature_maps = activation_model.predict(
            image,
            verbose=0,
        )

        saved_files = []

        for feature_map, layer_name in zip(
            feature_maps,
            layer_names,
        ):

            channels = feature_map.shape[-1]

            filters = min(
                channels,
                max_filters,
            )

            cols = 4

            rows = int(np.ceil(filters / cols))

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(10, 8),
            )

            axes = np.array(axes).reshape(-1)

            for i in range(filters):

                axes[i].imshow(
                    feature_map[0, :, :, i],
                    cmap="viridis",
                )

                axes[i].set_title(
                    f"Filter {i+1}",
                    fontsize=8,
                )

                axes[i].axis("off")

            for j in range(filters, len(axes)):

                axes[j].axis("off")

            plt.suptitle(
                layer_name,
                fontsize=14,
            )

            plt.tight_layout()

            save_path = self.output_dir / f"{layer_name}.png"

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

            saved_files.append(save_path)

        return saved_files
