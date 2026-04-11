"""
Análisis básico del dataset: dimensiones, nulos, tipos, balance del target
y features derivadas simples (TotalSpending, GroupSize, SpendingCategories).
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.features.constants import RAW_NUMERIC, SPENDING_COLS, TARGET


def compute_null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de valores nulos por columna (solo columnas con al menos un nulo)."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    return (
        pd.DataFrame({
            "Columna": null_counts.index,
            "Nulos": null_counts.values,
            "% Nulos": null_pct.values,
        })
        .query("Nulos > 0")
        .reset_index(drop=True)
    )


def compute_numeric_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Estadísticas descriptivas extendidas (media, std, skew, kurtosis) para numéricas."""
    desc = df[cols].describe().T.round(2)
    desc["skew"] = df[cols].skew().round(3)
    desc["kurtosis"] = df[cols].kurtosis().round(3)
    return desc


def compute_mannwhitney_stats(df: pd.DataFrame, col: str, target: str) -> dict:
    """Mann-Whitney U y correlación Pearson para una variable numérica vs target."""
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


def compute_derived_spending_stats(df: pd.DataFrame) -> dict:
    """Correlación TotalSpending crudo vs log-transformado con el target."""
    total = df[SPENDING_COLS].fillna(0).sum(axis=1)
    total_log = np.log1p(total)
    mask = total.notna() & df[TARGET].notna()
    r_raw, _ = stats.pearsonr(total[mask], df.loc[mask, TARGET].astype(int))
    r_log, _ = stats.pearsonr(total_log[mask], df.loc[mask, TARGET].astype(int))
    bool_target = df[TARGET].astype(bool)
    return {
        "r_raw": round(r_raw, 4),
        "r_log": round(r_log, 4),
        "group_f_raw": total[~bool_target],
        "group_t_raw": total[bool_target],
        "group_f_log": total_log[~bool_target],
        "group_t_log": total_log[bool_target],
    }


def run_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Dimensiones, tipos de dato, nulos y duplicados."""
    return {
        "shape": df.shape,
        "nulls": compute_null_summary(df),
        "dupes": int(df.duplicated().sum()),
        "dtypes": pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "Valores unicos": [df[c].nunique() for c in df.columns],
        }),
    }


def run_target_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Balance del target."""
    counts = df[TARGET].value_counts()
    pcts = df[TARGET].value_counts(normalize=True) * 100
    return {
        "counts": counts,
        "pcts": pcts,
        "balance_df": pd.DataFrame({
            "Clase": counts.index.astype(str),
            "Conteo": counts.values,
            "% del total": pcts.values.round(2),
        }),
    }


def run_statistical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Tests Mann-Whitney para numéricas vs target."""
    from src.features.constants import RAW_CATEGORICAL
    from src.pipelines.eda.statistical import compute_chi2_stats

    num_stats = compute_numeric_stats(df, RAW_NUMERIC)
    num_vs_target = [compute_mannwhitney_stats(df, col, TARGET) for col in RAW_NUMERIC]
    cat_stats = {col: compute_chi2_stats(df, col, TARGET) for col in RAW_CATEGORICAL}
    return {
        "numeric_desc": num_stats,
        "num_vs_target": num_vs_target,
        "cat_stats": cat_stats,
    }


def run_derived_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """TotalSpending, GroupSize y SpendingCategories vs target."""
    from src.pipelines.eda.statistical import compute_chi2_stats

    temp = df.copy()
    temp["TravelGroup"] = temp["PassengerId"].str.split("_").str[0]
    temp["GroupSize"] = temp.groupby("TravelGroup")["TravelGroup"].transform("count")
    temp["SpendingCategories"] = (temp[SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    return {
        "spending": compute_derived_spending_stats(temp),
        "groupsize": compute_chi2_stats(temp, "GroupSize", TARGET),
        "spending_categories": compute_chi2_stats(temp, "SpendingCategories", TARGET),
    }
