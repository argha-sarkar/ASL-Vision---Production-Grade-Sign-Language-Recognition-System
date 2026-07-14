import pandas as pd

from src.data.schema import EXPECTED_PIXELS, LABEL_COLUMN


class DataValidator:

    @staticmethod
    def validate_columns(df: pd.DataFrame):

        if LABEL_COLUMN not in df.columns:
            raise ValueError("Label column missing.")

        if len(df.columns) != EXPECTED_PIXELS + 1:
            raise ValueError("Incorrect number of columns.")

    @staticmethod
    def validate_missing(df):

        missing = df.isnull().sum().sum()

        if missing > 0:
            raise ValueError(f"Dataset contains {missing} missing values.")

    @staticmethod
    def validate_duplicates(df):

        duplicates = df.duplicated().sum()

        return duplicates
