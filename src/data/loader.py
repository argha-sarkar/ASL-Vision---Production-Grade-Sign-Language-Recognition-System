import pandas as pd

from src.data.constants import TRAIN_FILE, TEST_FILE


class DataLoader:
    """
    Load raw datasets.
    """

    @staticmethod
    def load_train():

        return pd.read_csv(TRAIN_FILE)

    @staticmethod
    def load_test():

        return pd.read_csv(TEST_FILE)