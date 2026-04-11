"""
Análisis bivariado: interacciones entre pares de variables clave.

Los modelos de árbol capturan interacciones automáticamente, pero documentarlas
aquí justifica las features de fs-011/fs-012/fs-013 y guía futuros experimentos.

Cruces analizados:
  - HomePlanet × CryoSleep → tasa transported (motivó fs-013 GroupAllCryo)
  - Deck × HomePlanet → distribución (valida reglas físicas R4/R5)
  - AgeGroup × CryoSleep → patrón niños sin cryo con cero gasto (R3)
  - CryoSleep × Spending → separación clean entre clases
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.features.constants import SPENDING_COLS, TARGET


def compute_cryo_homeplanet_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de Transported por (HomePlanet × CryoSleep).

    Returns:
        Pivot table: HomePlanet (filas) × CryoSleep (columnas) → transport rate.
    """
    temp = df.dropna(subset=["HomePlanet", "CryoSleep"]).copy()
    temp["CryoSleep_str"] = temp["CryoSleep"].astype(str)
    pivot = temp.groupby(["HomePlanet", "CryoSleep_str"])[TARGET].mean().unstack(fill_value=np.nan)
    pivot.columns.name = "CryoSleep"
    return pivot.round(4)


def compute_deck_homeplanet_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de pasajeros por (HomePlanet × Deck).

    Valida visualmente las reglas físicas R4 y R5.

    Returns:
        Pivot table: HomePlanet (filas) × Deck (columnas) → conteo de pasajeros.
    """
    temp = df.dropna(subset=["HomePlanet", "Cabin"]).copy()
    temp["Deck"] = temp["Cabin"].str.split("/").str[0]
    pivot = temp.groupby(["HomePlanet", "Deck"]).size().unstack(fill_value=0)
    pivot.columns.name = "Deck"
    return pivot


def compute_age_cryo_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Medias de Age y gasto total por segmento CryoSleep.

    Muestra que niños (Age<=12) en CryoSleep tienen gasto=0 (R3),
    y que CryoSleep segmenta fuertemente el patrón de spending.

    Returns:
        DataFrame: CryoSleep_str × [mean_age, mean_spending, transport_rate, n].
    """
    temp = df.copy()
    temp["CryoSleep_str"] = temp["CryoSleep"].map(
        {True: "True", "True": "True", False: "False", "False": "False"}
    ).fillna("Unknown")
    temp["TotalSpending"] = temp[SPENDING_COLS].fillna(0).sum(axis=1)

    agg = (
        temp.groupby("CryoSleep_str")
        .agg(
            mean_age=("Age", "mean"),
            mean_spending=("TotalSpending", "mean"),
            transport_rate=(TARGET, "mean"),
            n=(TARGET, "count"),
        )
        .round(3)
        .reset_index()
    )
    agg.columns = ["CryoSleep", "Edad media", "Gasto medio", "Transported rate", "n"]
    return agg


def run_bivariate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis bivariado de interacciones clave entre variables.

    Returns:
        dict con:
        - cryo_homeplanet: pivot tasa transported (HomePlanet × CryoSleep)
        - deck_homeplanet: pivot conteo (HomePlanet × Deck)
        - age_cryo: stats Age/Spending por estado CryoSleep
    """
    return {
        "cryo_homeplanet": compute_cryo_homeplanet_pivot(df),
        "deck_homeplanet": compute_deck_homeplanet_pivot(df),
        "age_cryo": compute_age_cryo_stats(df),
    }
