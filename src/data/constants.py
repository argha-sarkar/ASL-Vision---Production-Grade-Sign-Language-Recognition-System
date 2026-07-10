from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Files
TRAIN_FILE = RAW_DATA_DIR / "sign_mnist_train.csv"
TEST_FILE = RAW_DATA_DIR / "sign_mnist_test.csv"

# Reports
REPORT_DIR = PROJECT_ROOT / "reports"