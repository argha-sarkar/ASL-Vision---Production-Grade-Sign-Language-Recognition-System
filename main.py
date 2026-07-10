"""
main.py

Entry point for the ASL Vision project.

Responsibilities:
1. Load datasets
2. Validate datasets
3. Generate profiling statistics
4. Display dataset summary
5. Visualize class distribution
"""

from pprint import pprint

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.profiling.profiler import DatasetProfiler
from src.profiling.statistics import DatasetStatistics
from src.visualization.plots import DatasetPlots


def validate_dataset(dataset_name: str, dataframe):
    """
    Validate a dataset and print validation results.
    """

    print("=" * 70)
    print(f"Validating {dataset_name} Dataset")
    print("=" * 70)

    # Validate dataset structure
    DataValidator.validate_columns(dataframe)

    # Validate missing values
    DataValidator.validate_missing(dataframe)

    # Count duplicates
    duplicates = DataValidator.validate_duplicates(dataframe)

    print(f"Validation Completed Successfully")
    print(f"Duplicate Rows : {duplicates}")
    print()


def profile_dataset(dataset_name: str, dataframe):
    """
    Generate dataset profile and display it.
    """

    print("=" * 70)
    print(f"{dataset_name} Dataset Profile")
    print("=" * 70)

    report = DatasetProfiler.profile(dataframe)

    pprint(report)

    print()

    return report


def visualize_dataset(dataframe):
    """
    Generate dataset visualizations.
    """

    distribution = DatasetStatistics.class_distribution(dataframe)

    DatasetPlots.class_distribution(distribution)


def main():

    print("\n")
    print("=" * 70)
    print("ASL Vision - Data Validation & Profiling")
    print("=" * 70)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_df = DataLoader.load_train()
    test_df = DataLoader.load_test()

    print("Datasets Loaded Successfully")
    print()

    # --------------------------------------------------
    # Validate datasets
    # --------------------------------------------------

    validate_dataset("Training", train_df)
    validate_dataset("Testing", test_df)

    # --------------------------------------------------
    # Profile datasets
    # --------------------------------------------------

    train_report = profile_dataset("Training", train_df)

    test_report = profile_dataset("Testing", test_df)

    # --------------------------------------------------
    # Display Shapes
    # --------------------------------------------------

    print("=" * 70)
    print("Dataset Shapes")
    print("=" * 70)

    print(f"Training Shape : {train_df.shape}")
    print(f"Testing Shape  : {test_df.shape}")

    print()

    # --------------------------------------------------
    # Class Distribution
    # --------------------------------------------------

    print("=" * 70)
    print("Displaying Training Class Distribution")
    print("=" * 70)

    visualize_dataset(train_df)

    print()

    print("=" * 70)
    print("Pipeline Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()