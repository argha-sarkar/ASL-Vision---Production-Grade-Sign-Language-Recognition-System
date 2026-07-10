import numpy as np


class TensorConverter:

    @staticmethod
    def to_tensor(images: np.ndarray):

        return np.expand_dims(
            images,
            axis=-1
        )