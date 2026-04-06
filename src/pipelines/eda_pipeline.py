"""
Análisis Exploratorio de Datos — funciones de cálculo y orquestación.

Las funciones compute_* calculan estadísticas puras sobre un DataFrame.
Las funciones run_* las agrupan por bloque temático y devuelven dicts
listos para ser reportados por 01_eda.py.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.features.constants import (
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    SPENDING_COLS,
    TARGET
)

# ---------------------------------------------------------------------------
# Cálculos estadísticos
# ---------------------------------------------------------------------------

def compute_null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de valores nulos por columna.

    Args:
        df: DataFrame a analizar.

    Returns:
        DataFrame con columnas [Columna, Nulos, % Nulos],
        filtrado a columnas con al menos un nulo.
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
    """Estadísticas descriptivas extendidas para variables numéricas.

    Args:
        df: DataFrame con las columnas.
        cols: Lista de columnas numéricas a analizar.

    Returns:
        DataFrame transpuesto con estadísticas por columna.
    """
    desc = df[cols].describe().T.round(2)
    desc["skew"] = df[cols].skew().round(3)
    desc["kurtosis"] = df[cols].kurtosis().round(3)
    return desc


def compute_chi2_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Test chi-cuadrado entre una variable categórica y el target.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna categórica.
        target: Nombre de la columna target.

    Returns:
        Diccionario con chi2, p, dof y summary (frecuencias + tasa del target).
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

    result = {
        "chi2": round(chi2, 2), 
        "p": p_val, 
        "dof": dof, 
        "summary": summary
    }

    return result


def compute_mannwhitney_stats(
    df: pd.DataFrame,
    col: str,
    target: str
) -> dict:
    """Mann-Whitney U y correlación de Pearson para una variable numérica vs target.

    Args:
        df: DataFrame con los datos.
        col: Nombre de la columna numérica.
        target: Nombre de la columna target.

    Returns:
        Diccionario con stat_mw, p_mw, r_pearson, medias y grupos por clase.
    """
    group_true = df.loc[df[target].astype(bool), col].dropna()
    group_false = df.loc[~df[target].astype(bool), col].dropna()

    stat_mw, p_mw = stats.mannwhitneyu(group_true, group_false, alternative="two-sided")

    mask = df[col].notna() & df[target].notna()
    r_p, _ = stats.pearsonr(df.loc[mask, col], df.loc[mask, target].astype(int))

    result = {
        "stat_mw": round(stat_mw, 0),
        "p_mw": p_mw,
        "r_pearson": round(r_p, 4),
        "mean_true": round(group_true.mean(), 3),
        "mean_false": round(group_false.mean(), 3),
        "group_true": group_true,
        "group_false": group_false,
    }

    return result


def compute_derived_spending_stats(
    df: pd.DataFrame,
    spending_cols: List[str],
    target: str,
) -> dict:
    """Correlación de TotalSpending crudo vs log-transformado con el target.

    Args:
        df: DataFrame con las columnas de gasto.
        spending_cols: Lista de columnas de gasto individuales.
        target: Nombre de la columna target.

    Returns:
        Diccionario con r_raw, r_log y grupos por clase para ambas versiones.
    """
    total = df[spending_cols].fillna(0).sum(axis=1)
    total_log = np.log1p(total)

    mask = total.notna() & df[target].notna()
    r_raw, _ = stats.pearsonr(total[mask], df.loc[mask, target].astype(int))
    r_log, _ = stats.pearsonr(total_log[mask], df.loc[mask, target].astype(int))

    bool_target = df[target].astype(bool)
    result = {
        "r_raw": round(r_raw, 4),
        "r_log": round(r_log, 4),
        "group_f_raw": total[~bool_target],
        "group_t_raw": total[bool_target],
        "group_f_log": total_log[~bool_target],
        "group_t_log": total_log[bool_target],
    }

    return result

# ---------------------------------------------------------------------------
# Orquestación por bloques
# ---------------------------------------------------------------------------

def run_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Dimensiones, tipos de dato y nulos del dataset.

    Args:
        df: DataFrame crudo.

    Returns:
        Dict con shape, nulls, dupes y dtypes.
    """
    result = {
        "shape": df.shape,
        "nulls": compute_null_summary(df),
        "dupes": int(df.duplicated().sum()),
        "dtypes": pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "Valores unicos": [df[c].nunique() for c in df.columns],
        }),
    }

    return result

def run_target_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Balance del target.

    Args:
        df: DataFrame crudo.

    Returns:
        Dict con counts, pcts y balance_df para reportar.
    """
    counts = df[TARGET].value_counts()
    pcts = df[TARGET].value_counts(normalize=True) * 100
    result = {
        "counts": counts,
        "pcts": pcts,
        "balance_df": pd.DataFrame({
            "Clase": counts.index.astype(str),
            "Conteo": counts.values,
            "% del total": pcts.values.round(2),
        }),
    }

    return result

def run_statistical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Tests estadísticos para variables numéricas y categóricas vs target.

    Args:
        df: DataFrame crudo.

    Returns:
        Dict con numeric_desc, num_vs_target y cat_stats.
    """
    num_stats = compute_numeric_stats(df, RAW_NUMERIC)
    num_vs_target = [compute_mannwhitney_stats(df, col, TARGET) for col in RAW_NUMERIC]

    cat_stats = {col: compute_chi2_stats(df, col, TARGET) for col in RAW_CATEGORICAL}

    result = {
        "numeric_desc": num_stats,
        "num_vs_target": num_vs_target,
        "cat_stats": cat_stats,
    }

    return result


def run_derived_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis de features derivadas: TotalSpending_Log, GroupSize, SpendingCategories.

    Args:
        df: DataFrame crudo.

    Returns:
        Dict con spending, groupsize y spending_categories.
    """
    temp_df = df.copy()
    temp_df["TravelGroup"] = temp_df["PassengerId"].str.split("_").str[0]
    temp_df["GroupSize"] = temp_df.groupby("TravelGroup")["TravelGroup"].transform("count")
    temp_df["SpendingCategories"] = (temp_df[SPENDING_COLS].fillna(0) > 0).sum(axis=1)

    result = {
        "spending": compute_derived_spending_stats(temp_df, SPENDING_COLS, TARGET),
        "groupsize": compute_chi2_stats(temp_df, "GroupSize", TARGET),
        "spending_categories": compute_chi2_stats(temp_df, "SpendingCategories", TARGET),
    }

    return result
