"""
Modulo para preprocesamiento de datos.

Funciones genericas para limpiar, transformar y preparar datos
para analisis o modelado.
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


def clean_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    columns: Optional[List[str]] = None,
    fill_value: Any = None
) -> pd.DataFrame:
    """
    Maneja valores faltantes en el DataFrame.

    Args:
        df: DataFrame a procesar
        strategy: Estrategia ('drop', 'fill', 'median', 'mean', 'mode')
        columns: Columnas especificas a procesar (None = todas)
        fill_value: Valor para rellenar si strategy='fill'

    Returns:
        DataFrame con valores faltantes manejados
    """
    df_copy = df.copy()
    cols = columns if columns else df_copy.columns

    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=cols)
    elif strategy == 'fill':
        df_copy[cols] = df_copy[cols].fillna(fill_value)
    elif strategy == 'median':
        for col in cols:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    elif strategy == 'mean':
        for col in cols:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    elif strategy == 'mode':
        for col in cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    return df_copy


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Elimina filas duplicadas del DataFrame.

    Args:
        df: DataFrame a procesar
        subset: Columnas a considerar para detectar duplicados
        keep: Que duplicado mantener ('first', 'last', False)

    Returns:
        DataFrame sin duplicados
    """
    return df.drop_duplicates(subset=subset, keep=keep)


def convert_dates(
    df: pd.DataFrame,
    date_columns: List[str],
    format: Optional[str] = None,
    errors: str = 'coerce'
) -> pd.DataFrame:
    """
    Convierte columnas a tipo datetime.

    Args:
        df: DataFrame a procesar
        date_columns: Lista de columnas a convertir
        format: Formato de fecha (None = inferir automaticamente)
        errors: Como manejar errores ('raise', 'coerce', 'ignore')

    Returns:
        DataFrame con columnas convertidas a datetime
    """
    df_copy = df.copy()
    for col in date_columns:
        df_copy[col] = pd.to_datetime(df_copy[col], format=format, errors=errors)
    return df_copy


def normalize_text(
    df: pd.DataFrame,
    text_columns: List[str],
    lowercase: bool = True,
    strip: bool = True,
    remove_special: bool = False
) -> pd.DataFrame:
    """
    Normaliza columnas de texto.

    Args:
        df: DataFrame a procesar
        text_columns: Columnas de texto a normalizar
        lowercase: Convertir a minusculas
        strip: Eliminar espacios al inicio/final
        remove_special: Eliminar caracteres especiales

    Returns:
        DataFrame con texto normalizado
    """
    df_copy = df.copy()

    for col in text_columns:
        if lowercase:
            df_copy[col] = df_copy[col].str.lower()
        if strip:
            df_copy[col] = df_copy[col].str.strip()
        if remove_special:
            df_copy[col] = df_copy[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    return df_copy


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detecta outliers en una columna numerica.

    Args:
        df: DataFrame a analizar
        column: Columna a analizar
        method: Metodo de deteccion ('iqr', 'zscore')
        threshold: Umbral para considerar outlier

    Returns:
        Serie booleana indicando outliers (True = outlier)
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (df[column] < lower) | (df[column] > upper)

    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        return pd.Series(z_scores > threshold, index=df.index)

    return pd.Series(False, index=df.index)
