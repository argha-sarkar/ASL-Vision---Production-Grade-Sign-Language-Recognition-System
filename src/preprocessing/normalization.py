import numpy as np


class ImageNormalization:

    @staticmethod
    def normalize(images: np.ndarray):

        return images.astype("float32") / 255.0
