"""
Funciones de análisis estadístico para exploración de datos.

Estas funciones extraen cálculos reutilizables del script 01_eda.py,
permitiendo que el script sea un orquestador delgado.
"""
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def compute_null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el resumen de valores nulos por columna.

    Args:
        df: DataFrame a analizar.

    Returns:
        DataFrame con columnas [Columna, Nulos, % Nulos],
        filtrado solo a columnas con al menos un nulo.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return (
        pd.DataFrame(
            {
                "Columna": null_counts.index,
                "Nulos": null_counts.values,
                "% Nulos": null_pct.values,
            }
        )
        .query("Nulos > 0")
        .reset_index(drop=True)
    )


def compute_numeric_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Calcula estadísticas descriptivas extendidas para variables numéricas.

    Incluye count, mean, std, min, percentiles, max, skew y kurtosis.

    Args:
        df: DataFrame que contiene las columnas.
        cols: Lista de columnas numéricas a analizar.

    Returns:
        DataFrame transpuesto con estadísticas por columna.
    """
    desc = df[cols].describe().T.round(2)
    desc["skew"] = df[cols].skew().round(3)
    desc["kurtosis"] = df[cols].kurtosis().round(3)
    return desc


def compute_chi2_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Calcula el test chi-cuadrado entre una variable categórica y el target.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna categórica.
        target: Nombre de la columna target (booleana o binaria).

    Returns:
        Diccionario con claves:
            - chi2 (float): Estadístico chi-cuadrado.
            - p (float): p-valor del test.
            - dof (int): Grados de libertad.
            - summary (DataFrame): Frecuencias y tasa del target por categoría.
    """
    freq = df[col].value_counts(dropna=False).reset_index()
    freq.columns = [col, "count"]
    freq["pct"] = (freq["count"] / len(df) * 100).round(2)

    target_rate = (
        df.groupby(col, dropna=False)[target]
        .mean()
        .reset_index()
        .rename(columns={target: "tasa_transported"})
    )
    target_rate["tasa_transported"] = target_rate["tasa_transported"].round(4)
    summary = freq.merge(target_rate, on=col, how="left")

    contingency = pd.crosstab(df[col].fillna("NaN"), df[target])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    return {"chi2": round(chi2, 2), "p": p_val, "dof": dof, "summary": summary}


def compute_mannwhitney_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Calcula Mann-Whitney U y correlación de Pearson para una variable numérica vs target.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna numérica.
        target: Nombre de la columna target (booleana o binaria).

    Returns:
        Diccionario con claves:
            - stat_mw (float): Estadístico Mann-Whitney.
            - p_mw (float): p-valor del test.
            - r_pearson (float): Correlación de Pearson con el target.
            - mean_true (float): Media del grupo target=True.
            - mean_false (float): Media del grupo target=False.
            - group_true (Series): Valores del grupo True (sin NaN).
            - group_false (Series): Valores del grupo False (sin NaN).
    """
    group_true = df.loc[df[target].astype(bool), col].dropna()
    group_false = df.loc[~df[target].astype(bool), col].dropna()

    stat_mw, p_mw = stats.mannwhitneyu(group_true, group_false, alternative="two-sided")

    mask = df[col].notna() & df[target].notna()
    r_p, _ = stats.pearsonr(df.loc[mask, col], df.loc[mask, target].astype(int))

    return {
        "stat_mw": round(stat_mw, 0),
        "p_mw": p_mw,
        "r_pearson": round(r_p, 4),
        "mean_true": round(group_true.mean(), 3),
        "mean_false": round(group_false.mean(), 3),
        "group_true": group_true,
        "group_false": group_false,
    }


def compute_derived_spending_stats(
    df: pd.DataFrame,
    spending_cols: List[str],
    target: str,
) -> dict:
    """Calcula estadísticas para TotalSpending y su transformación logarítmica.

    Args:
        df: DataFrame con las columnas de gasto.
        spending_cols: Lista de columnas de gasto (RoomService, FoodCourt, etc.).
        target: Nombre de la columna target.

    Returns:
        Diccionario con claves:
            - r_raw (float): Correlación de Pearson de TotalSpending vs target.
            - r_log (float): Correlación de Pearson de TotalSpending_Log vs target.
            - group_f_raw / group_t_raw (Series): Grupos por target sin transformación.
            - group_f_log / group_t_log (Series): Grupos por target con log.
    """
    total = df[spending_cols].fillna(0).sum(axis=1)
    total_log = np.log1p(total)

    mask = total.notna() & df[target].notna()
    r_raw, _ = stats.pearsonr(total[mask], df.loc[mask, target].astype(int))
    r_log, _ = stats.pearsonr(total_log[mask], df.loc[mask, target].astype(int))

    bool_target = df[target].astype(bool)
    return {
        "r_raw": round(r_raw, 4),
        "r_log": round(r_log, 4),
        "group_f_raw": total[~bool_target],
        "group_t_raw": total[bool_target],
        "group_f_log": total_log[~bool_target],
        "group_t_log": total_log[bool_target],
    }
