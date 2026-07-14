import random

import matplotlib.pyplot as plt
import numpy as np


class DatasetPlots:

    @staticmethod
    def class_distribution(distribution):

        plt.figure(figsize=(12, 5))

        distribution.plot(kind="bar")

        plt.title("Class Distribution")

        plt.xlabel("Label")

        plt.ylabel("Images")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def show_random_images(images, labels, rows=3, cols=5):

        plt.figure(figsize=(12, 7))

        total = rows * cols

        indices = random.sample(range(len(images)), total)

        for position, index in enumerate(indices):

            plt.subplot(rows, cols, position + 1)

            plt.imshow(images[index], cmap="gray")

            plt.title(f"Label : {labels[index]}")

            plt.axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def show_class_examples(images, labels):

        unique_labels = sorted(set(labels))

        plt.figure(figsize=(15, 8))

        for position, label in enumerate(unique_labels):

            index = list(labels).index(label)

            plt.subplot(5, 5, position + 1)

            plt.imshow(images[index], cmap="gray")

            plt.title(str(label))

            plt.axis("off")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def pixel_histogram(images):

        plt.figure(figsize=(10, 5))

        plt.hist(
            images.flatten(),
            bins=50,
        )

        plt.title("Pixel Intensity Distribution")

        plt.xlabel("Pixel Value")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.show()

    @staticmethod
    def mean_image(images):

        mean = images.mean(axis=0)

        plt.figure(figsize=(4, 4))

        plt.imshow(mean, cmap="gray")

        plt.title("Mean Image")

        plt.axis("off")

        plt.show()

    @staticmethod
    def median_image(images):

        median = np.median(images, axis=0)

        plt.figure(figsize=(4, 4))

        plt.imshow(median, cmap="gray")

        plt.title("Median Image")

        plt.axis("off")

        plt.show()
