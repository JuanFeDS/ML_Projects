"""Modulo de configuracion centralizada."""

from src.config.settings import (
    DATA_FEATURES_DIR,
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_RAW,
    TRAIN_RAW,
)

__all__ = [
    "TRAIN_RAW",
    "TEST_RAW",
    "DATA_PROCESSED_DIR",
    "DATA_FEATURES_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DOCS_DIR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
]
