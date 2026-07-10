import matplotlib.pyplot as plt


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