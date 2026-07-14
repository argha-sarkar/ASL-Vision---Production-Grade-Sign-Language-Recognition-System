"""
activation_maps.py

Visualize activation maps produced by all convolutional
layers of a trained CNN model.

Author: Argha Sarkar Project
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ActivationMapVisualizer:
    """
    Visualize activation maps from all Conv2D layers.
    """

    def __init__(self, model):

        self.model = model

        self.output_dir = Path("reports/explainability/activation_maps")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _find_conv_layers(self):

        outputs = []
        names = []

        for layer in self.model.layers:

            if isinstance(layer, tf.keras.layers.Conv2D):

                outputs.append(layer.output)
                names.append(layer.name)

        return outputs, names

    def _build_activation_model(self):

        outputs, names = self._find_conv_layers()

        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=outputs,
        )

        return activation_model, names

    def generate(
        self,
        image,
        max_maps=16,
    ):

        if image.ndim == 3:

            image = np.expand_dims(
                image,
                axis=0,
            )

        activation_model, names = self._build_activation_model()

        activations = activation_model.predict(
            image,
            verbose=0,
        )

        saved_files = []

        for activation, layer_name in zip(
            activations,
            names,
        ):

            channels = activation.shape[-1]

            total_maps = min(
                channels,
                max_maps,
            )

            rows = 4
            cols = 4

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(10, 10),
            )

            axes = np.array(axes).reshape(-1)

            for index in range(total_maps):

                axes[index].imshow(
                    activation[0, :, :, index],
                    cmap="inferno",
                )

                axes[index].set_title(
                    f"Map {index+1}",
                    fontsize=8,
                )

                axes[index].axis("off")

            for index in range(
                total_maps,
                len(axes),
            ):

                axes[index].axis("off")

            plt.suptitle(
                f"Activation Maps - {layer_name}",
                fontsize=15,
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
