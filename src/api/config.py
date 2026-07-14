"""
config.py

FastAPI Configuration

Author: Argha Sarkar Project
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application Configuration
    """

    APP_NAME: str = "ASL Vision API"

    VERSION: str = "1.0.0"

    DESCRIPTION: str = "Production API for American Sign Language Recognition"

    HOST: str = "0.0.0.0"

    PORT: int = 8000

    DEBUG: bool = False

    MODEL_PATH: str = "models/best_model.keras"

    LABELS_PATH: str = "artifacts/label_encoder.pkl"

    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024

    ALLOWED_EXTENSIONS: list = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    API_PREFIX: str = "/api/v1"

    REPORT_DIR: str = "reports"

    LOG_DIR: str = "logs"

    LOG_LEVEL: str = "INFO"

    TEMP_DIR: str = "temp"

    CORS_ORIGINS: list = [
        "*",
    ]

    class Config:

        env_file = ".env"

        case_sensitive = True


settings = Settings()


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

ROOT_DIR = Path.cwd()

MODEL_DIR = ROOT_DIR / "models"

REPORT_DIR = ROOT_DIR / settings.REPORT_DIR

LOG_DIR = ROOT_DIR / settings.LOG_DIR

TEMP_DIR = ROOT_DIR / settings.TEMP_DIR

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
