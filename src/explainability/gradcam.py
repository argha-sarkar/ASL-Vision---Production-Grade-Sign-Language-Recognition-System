"""
gradcam.py

Generate Grad-CAM visualizations for CNN models.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class GradCAM:

    def __init__(
        self,
        model,
        last_conv_layer="conv3",
    ):

        self.model = model

        self.last_conv_layer = last_conv_layer

        self.output_dir = Path(
            "reports/explainability/gradcam"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate_heatmap(
        self,
        image,
    ):

        grad_model = tf.keras.models.Model(

            inputs=self.model.inputs,

            outputs=[

                self.model.get_layer(
                    self.last_conv_layer
                ).output,

                self.model.output,

            ],

        )

        image_tensor = tf.expand_dims(
            image,
            axis=0,
        )

        with tf.GradientTape() as tape:

            conv_output, predictions = grad_model(
                image_tensor
            )

            predicted_class = tf.argmax(
                predictions[0]
            )

            loss = predictions[
                :,
                predicted_class,
            ]

        gradients = tape.gradient(
            loss,
            conv_output,
        )

        pooled_gradients = tf.reduce_mean(

            gradients,

            axis=(0, 1, 2),

        )

        conv_output = conv_output[0]

        heatmap = conv_output @ pooled_gradients[..., tf.newaxis]

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        heatmap /= tf.reduce_max(
            heatmap
        ) + 1e-10

        return heatmap.numpy()

    def overlay(
        self,
        image,
        heatmap,
        alpha=0.4,
    ):

        heatmap = cv2.resize(

            heatmap,

            (28, 28),

        )

        heatmap = np.uint8(
            255 * heatmap
        )

        colored = cv2.applyColorMap(

            heatmap,

            cv2.COLORMAP_JET,

        )

        original = np.uint8(
            image.squeeze() * 255
        )

        original = cv2.cvtColor(

            original,

            cv2.COLOR_GRAY2BGR,

        )

        overlay = cv2.addWeighted(

            original,

            1 - alpha,

            colored,

            alpha,

            0,

        )

        return overlay

    def save(
        self,
        image,
        filename,
    ):

        heatmap = self.generate_heatmap(
            image
        )

        overlay = self.overlay(
            image,
            heatmap,
        )

        save_path = self.output_dir / filename

        plt.figure(figsize=(4, 4))

        plt.imshow(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB,
            )
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()

        return save_path