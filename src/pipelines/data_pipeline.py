"""
Pipeline de datos: ingesta y cadena hasta features listos para modelado.

La limpieza y transformaciones base viven en el `pipeline` de cada
`FeatureSet` (ver src/features/feature_sets.py). Este módulo ofrece una API
única para scripts y tests.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.config.settings import TRAIN_RAW
from src.pipelines.feature_pipeline import run_feature_pipeline


def load_raw_train() -> pd.DataFrame:
    """Carga el CSV de entrenamiento desde data/raw (o DATA_RAW_DIR)."""
    return pd.read_csv(TRAIN_RAW)


def run_ingestion_to_features_pipeline(
    df_raw: pd.DataFrame, fs: Any, fs_name: str
) -> Dict[str, Any]:
    """Ejecuta ingeniería de características completa para un feature set.

    Equivale a `run_feature_pipeline`: ingesta ya materializada como DataFrame
    más el objeto FeatureSet que aplica limpieza → encoding → escalado.

    Args:
        df_raw: Datos crudos (mismas columnas que train.csv de Kaggle).
        fs: Instancia de FeatureSet.
        fs_name: Identificador del set (ej. fs-001_baseline).

    Returns:
        Dict con X_raw, X_scaled, y, scaler, target_encoder y metadata.
    """
    return run_feature_pipeline(df_raw, fs, fs_name)
