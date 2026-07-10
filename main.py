from src.data.loader import DataLoader
from src.data.validator import DataValidator


def main():

    train = DataLoader.load_train()

    test = DataLoader.load_test()

    DataValidator.validate_columns(train)
    DataValidator.validate_columns(test)

    DataValidator.validate_missing(train)
    DataValidator.validate_missing(test)

    train_duplicates = DataValidator.validate_duplicates(train)
    test_duplicates = DataValidator.validate_duplicates(test)

    print("=" * 60)
    print("DATA VALIDATION COMPLETED")
    print("=" * 60)
    print(f"Training Shape : {train.shape}")
    print(f"Testing Shape  : {test.shape}")
    print(f"Training Duplicates : {train_duplicates}")
    print(f"Testing Duplicates  : {test_duplicates}")


if __name__ == "__main__":
    main()