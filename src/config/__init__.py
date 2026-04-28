"""Modulo de configuracion centralizada."""

from src.config.settings import (
    TRAIN_RAW,
    TEST_RAW,
    DATA_PROCESSED_DIR,
    DATA_FEATURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    DOCS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
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
