import pandas as pd


class DatasetStatistics:

    @staticmethod
    def dataset_shape(df: pd.DataFrame):

        return df.shape

    @staticmethod
    def missing_values(df):

        return df.isnull().sum().sum()

    @staticmethod
    def duplicate_rows(df):

        return df.duplicated().sum()

    @staticmethod
    def memory_usage(df):

        return df.memory_usage(deep=True).sum() / 1024**2

    @staticmethod
    def class_distribution(df):

        return df["label"].value_counts().sort_index()

    @staticmethod
    def pixel_min(df):

        return df.iloc[:, 1:].min().min()

    @staticmethod
    def pixel_max(df):

        return df.iloc[:, 1:].max().max()

    @staticmethod
    def pixel_mean(df):

        return df.iloc[:, 1:].mean().mean()

    @staticmethod
    def pixel_std(df):

        return df.iloc[:, 1:].std().mean()