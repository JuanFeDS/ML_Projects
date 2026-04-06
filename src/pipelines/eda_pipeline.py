"""
Orquestador del Análisis Exploratorio de Datos (EDA).

Provee funciones de alto nivel para ejecutar el análisis estadístico
completo, devolviendo objetos de datos listos para ser reportados.
"""
from typing import Dict, Any, List
import pandas as pd
from src.data.eda import (
    compute_null_summary,
    compute_numeric_stats,
    compute_chi2_stats,
    compute_mannwhitney_stats,
    compute_derived_spending_stats
)
from src.features.constants import TARGET, RAW_NUMERIC, RAW_CATEGORICAL, SPENDING_COLS

def run_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Ejecuta el análisis básico de dimensiones, tipos y nulos."""
    return {
        "shape": df.shape,
        "nulls": compute_null_summary(df),
        "dupes": int(df.duplicated().sum()),
        "dtypes": pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "Valores unicos": [df[c].nunique() for c in df.columns],
        })
    }

def run_target_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza el balance del target."""
    counts = df[TARGET].value_counts()
    pcts = df[TARGET].value_counts(normalize=True) * 100
    return {
        "counts": counts,
        "pcts": pcts,
        "balance_df": pd.DataFrame({
            "Clase": counts.index.astype(str),
            "Conteo": counts.values,
            "% del total": pcts.values.round(2),
        })
    }

def run_statistical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Ejecuta tests estadísticos para variables numéricas y categóricas."""
    # Numéricas
    num_stats = compute_numeric_stats(df, RAW_NUMERIC)
    num_vs_target = []
    for col in RAW_NUMERIC:
        num_vs_target.append(compute_mannwhitney_stats(df, col, TARGET))
    
    # Categóricas
    cat_stats = {}
    for col in RAW_CATEGORICAL:
        cat_stats[col] = compute_chi2_stats(df, col, TARGET)
        
    return {
        "numeric_desc": num_stats,
        "num_vs_target": num_vs_target,
        "cat_stats": cat_stats
    }

def run_derived_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza features derivadas comunes."""
    # Nota: No mutamos el df original aquí para mantener pureza, 
    # pero necesitamos las columnas para el análisis
    temp_df = df.copy()
    temp_df["TravelGroup"] = temp_df["PassengerId"].str.split("_").str[0]
    temp_df["GroupSize"] = temp_df.groupby("TravelGroup")["TravelGroup"].transform("count")
    temp_df["SpendingCategories"] = (temp_df[SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    
    return {
        "spending": compute_derived_spending_stats(temp_df, SPENDING_COLS, TARGET),
        "groupsize": compute_chi2_stats(temp_df, "GroupSize", TARGET),
        "spending_categories": compute_chi2_stats(temp_df, "SpendingCategories", TARGET)
    }
