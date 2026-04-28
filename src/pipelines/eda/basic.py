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
        "dtypes": pd.DataFrame(
            {
                "Columna": df.columns,
                "Tipo": df.dtypes.values,
                "Valores unicos": [df[c].nunique() for c in df.columns],
            }
        ),
    }


def run_target_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Balance del target."""
    counts = df[TARGET].value_counts()
    pcts = df[TARGET].value_counts(normalize=True) * 100
    return {
        "counts": counts,
        "pcts": pcts,
        "balance_df": pd.DataFrame(
            {
                "Clase": counts.index.astype(str),
                "Conteo": counts.values,
                "% del total": pcts.values.round(2),
            }
        ),
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


def compute_feature_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Pearson entre todas las features numéricas del raw dataset.

    Incluye Age y los 5 servicios de gasto (log1p). Útil para detectar
    redundancias antes de construir el feature set.

    Returns:
        DataFrame: matriz de correlación simétrica con las features numéricas.
    """
    temp = df[RAW_NUMERIC].copy()
    for col in SPENDING_COLS:
        temp[f"log_{col}"] = np.log1p(temp[col].fillna(0))
    numeric_cols = ["Age"] + [f"log_{c}" for c in SPENDING_COLS]
    return temp[numeric_cols].corr().round(4)


def compute_vip_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Perfil detallado de pasajeros VIP.

    VIP tiene chi2=11.5 (p<0.001) con tasa transported=38.2% vs 50.6% no-VIP.
    Este análisis caracteriza quiénes son los VIP para guiar decisiones
    de feature engineering o imputación.

    Returns:
        dict con counts, transported_rate, homeplanet_dist, destination_dist,
        cryo_dist, age_stats y spending_median.
    """
    vip_mask = df["VIP"].isin([True, "True"])
    vip = df[vip_mask]
    non_vip = df[~vip_mask & df["VIP"].notna()]

    return {
        "counts": {
            "VIP": int(vip_mask.sum()),
            "no_VIP": int((~vip_mask & df["VIP"].notna()).sum()),
        },
        "transported_rate": {
            "VIP": round(vip[TARGET].mean(), 4),
            "no_VIP": round(non_vip[TARGET].mean(), 4),
        },
        "homeplanet_dist": vip["HomePlanet"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "destination_dist": vip["Destination"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "cryo_dist": vip["CryoSleep"].value_counts(normalize=True).round(4).to_dict(),
        "age_median": {
            "VIP": round(vip["Age"].median(), 1),
            "no_VIP": round(non_vip["Age"].median(), 1),
        },
        "total_spending_median": {
            "VIP": round(vip[SPENDING_COLS].fillna(0).sum(axis=1).median(), 1),
            "no_VIP": round(non_vip[SPENDING_COLS].fillna(0).sum(axis=1).median(), 1),
        },
    }


def run_derived_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """TotalSpending, GroupSize, SpendingCategories, correlaciones y perfil VIP."""
    from src.pipelines.eda.statistical import compute_chi2_stats

    temp = df.copy()
    temp["TravelGroup"] = temp["PassengerId"].str.split("_").str[0]
    temp["GroupSize"] = temp.groupby("TravelGroup")["TravelGroup"].transform("count")
    temp["SpendingCategories"] = (temp[SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    return {
        "spending": compute_derived_spending_stats(temp),
        "groupsize": compute_chi2_stats(temp, "GroupSize", TARGET),
        "spending_categories": compute_chi2_stats(temp, "SpendingCategories", TARGET),
        "feature_correlations": compute_feature_correlation_matrix(df),
        "vip_profile": compute_vip_profile(df),
    }
