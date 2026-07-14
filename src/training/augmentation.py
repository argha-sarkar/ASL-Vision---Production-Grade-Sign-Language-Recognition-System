import tensorflow as tf


class DataAugmentation:

    @staticmethod
    def build():

        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(factor=0.05),
                tf.keras.layers.RandomZoom(
                    height_factor=0.10,
                    width_factor=0.10,
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=0.05,
                    width_factor=0.05,
                ),
            ],
            name="augmentation_pipeline",
        )
