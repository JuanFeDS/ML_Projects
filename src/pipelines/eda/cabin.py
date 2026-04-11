"""
Análisis de la columna Cabin: distribución por Deck, Side y CabinNumber vs target.

Cabin tiene cardinalidad ~6500 — se descompone en Deck/CabinNumber/Side.
NB02: Deck (chi²=392.3, p<0.001) y Side (chi²=91.1, p<0.001).
"""
from typing import Any, Dict

import pandas as pd

from src.features.constants import TARGET
from src.pipelines.eda.statistical import compute_chi2_stats


def _extract_cabin_components(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae Deck y Side desde Cabin sin mutar el input."""
    out = df.copy()
    out["Deck"] = out["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) else "Unknown"
    )
    out["Side"] = out["Cabin"].apply(
        lambda x: x.split("/")[2] if pd.notna(x) else "Unknown"
    )
    out["CabinNumber"] = out["Cabin"].apply(
        lambda x: int(x.split("/")[1]) if pd.notna(x) else None
    )
    return out


def run_cabin_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Distribución y poder discriminativo de Deck, Side y CabinNumber.

    Returns:
        dict con:
        - deck: chi2, p, dof, summary (Deck × target rate)
        - side: chi2, p, dof, summary (Side × target rate)
        - cabin_null_pct: porcentaje de Cabin con valor nulo
        - deck_homeplanet: pivot HomePlanet × Deck (cuenta) — valida reglas físicas
    """
    cabin_df = _extract_cabin_components(df)
    cabin_null_pct = round(df["Cabin"].isna().mean() * 100, 2)

    deck_stats = compute_chi2_stats(cabin_df, "Deck", TARGET)
    side_stats = compute_chi2_stats(cabin_df, "Side", TARGET)

    # Pivot HomePlanet × Deck para validar reglas físicas
    # (Decks A/B/C son casi exclusivos de Europa, G de Earth)
    hp_deck = (
        cabin_df.dropna(subset=["HomePlanet"])
        .groupby(["HomePlanet", "Deck"])
        .size()
        .unstack(fill_value=0)
    )

    return {
        "deck": deck_stats,
        "side": side_stats,
        "cabin_null_pct": cabin_null_pct,
        "deck_homeplanet": hp_deck,
    }
