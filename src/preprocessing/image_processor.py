import numpy as np
import pandas as pd


class ImageProcessor:
    """
    Utility class for converting flattened pixel values
    into 28x28 grayscale images.
    """

    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    @staticmethod
    def reconstruct_image(row: pd.Series):

        pixels = row.iloc[1:].to_numpy(dtype=np.uint8)

        image = pixels.reshape(
            ImageProcessor.IMAGE_HEIGHT,
            ImageProcessor.IMAGE_WIDTH,
        )

        return image

    @staticmethod
    def reconstruct_images(df: pd.DataFrame):

        images = []

        for _, row in df.iterrows():

            image = ImageProcessor.reconstruct_image(row)

            images.append(image)

        return np.array(images)
