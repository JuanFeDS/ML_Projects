"""
Funciones de validación y calidad de datos del pipeline.

Usadas en 01_eda.py (validación del raw) y en 02_features.py
(validación post-engineering antes del escalado).
"""
from typing import Dict, List

import pandas as pd


def validate_raw_dataset(
    df: pd.DataFrame,
    expected_cols: List[str],
    null_threshold: float = 0.10,
) -> Dict[str, list]:
    """Valida que el dataset crudo cumple los requisitos mínimos del pipeline.

    Verifica:
    - Que las columnas esperadas están presentes.
    - Que ninguna columna supera el umbral de nulos.
    - Que no hay filas completamente duplicadas.

    Args:
        df: DataFrame a validar.
        expected_cols: Lista de columnas que deben estar presentes.
        null_threshold: Fracción máxima de nulos permitida por columna (default 0.10).

    Returns:
        Diccionario con claves "missing_cols", "high_null_cols" y "n_duplicates".
        Cada lista está vacía si no hay problemas.
    """
    missing_cols = [c for c in expected_cols if c not in df.columns]

    null_fracs = df.isnull().mean()
    high_null_cols = null_fracs[null_fracs > null_threshold].index.tolist()

    n_duplicates = int(df.duplicated().sum())

    return {
        "missing_cols": missing_cols,
        "high_null_cols": high_null_cols,
        "n_duplicates": n_duplicates,
    }


def check_residual_nulls(
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> List[str]:
    """Retorna las columnas con más del threshold% de nulos.

    Útil para detectar nulos residuales después del pipeline de imputación.

    Args:
        df: DataFrame post-procesado.
        threshold: Fracción máxima de nulos aceptable por columna (default 0.05).

    Returns:
        Lista de nombres de columnas que superan el umbral.
    """
    null_fracs = df.isnull().mean()
    return null_fracs[null_fracs > threshold].index.tolist()


def validate_feature_alignment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Verifica que los conjuntos de features de train y test están alineados.

    Args:
        df_train: DataFrame de features de entrenamiento.
        df_test: DataFrame de features de test.

    Returns:
        Diccionario con claves:
            - "only_in_train": columnas presentes en train pero no en test.
            - "only_in_test": columnas presentes en test pero no en train.
        Ambas listas vacías indican alineación perfecta.
    """
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    return {
        "only_in_train": sorted(train_cols - test_cols),
        "only_in_test": sorted(test_cols - train_cols),
    }
