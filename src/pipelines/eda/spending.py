"""
Análisis detallado de las columnas de gasto individual (RoomService, FoodCourt,
ShoppingMall, Spa, VRDeck).

Dimensiones de análisis:
- Zero-inflation: la mayoría de pasajeros no gasta en ningún servicio.
- Por servicio: correlación con target y tasa de transporte entre gastadores vs no.
- Perfil de gasto: cuántos servicios usa cada pasajero.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.features.constants import SPENDING_COLS, TARGET


def compute_per_service_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas por columna de gasto: zero-inflation, correlación y separación por target.

    Returns:
        DataFrame con una fila por servicio y columnas:
        - pct_zero: % de pasajeros con gasto = 0
        - r_pearson: correlación con el target (gasto log1p)
        - transport_rate_spender: tasa transported entre quienes sí gastan
        - transport_rate_non_spender: tasa transported entre quienes no gastan
        - median_spender: mediana de gasto (entre los que gastan > 0)
    """
    rows = []
    for col in SPENDING_COLS:
        values = df[col].fillna(0)
        target_vals = df[TARGET].astype(int)
        pct_zero = round((values == 0).mean() * 100, 2)
        r_p, _ = stats.pearsonr(np.log1p(values), target_vals)
        spender_mask = values > 0
        transport_spender = df.loc[spender_mask, TARGET].mean() if spender_mask.sum() > 0 else None
        transport_non = df.loc[~spender_mask, TARGET].mean()
        median_spender = values[spender_mask].median() if spender_mask.sum() > 0 else 0
        rows.append({
            "Servicio": col,
            "% sin gasto": pct_zero,
            "r Pearson (log)": round(r_p, 4),
            "Transported (gasta)": round(transport_spender, 4) if transport_spender else None,
            "Transported (no gasta)": round(transport_non, 4),
            "Mediana (gastadores)": round(median_spender, 1),
        })
    return pd.DataFrame(rows)


def compute_zero_inflation(df: pd.DataFrame) -> dict:
    """Cuantifica el patrón zero-inflated en las columnas de gasto.

    Returns:
        dict con:
        - per_service_zero_pct: Serie con % de ceros por servicio
        - all_zero_pct: % de pasajeros con todos los servicios en 0
        - zero_by_cryo: % de ceros en cryo vs activos (valida regla física)
    """
    values = df[SPENDING_COLS].fillna(0)
    per_service = (values == 0).mean() * 100

    all_zero_mask = (values == 0).all(axis=1)
    all_zero_pct = round(all_zero_mask.mean() * 100, 2)

    cryo_true = df["CryoSleep"].isin([True, "True"])
    zero_by_cryo = pd.DataFrame({
        "Segmento": ["CryoSleep=True", "CryoSleep=False/Unknown"],
        "% todos en cero": [
            round((values[cryo_true] == 0).all(axis=1).mean() * 100, 2),
            round((values[~cryo_true] == 0).all(axis=1).mean() * 100, 2),
        ],
        "n": [cryo_true.sum(), (~cryo_true).sum()],
    })

    return {
        "per_service_zero_pct": per_service.round(2),
        "all_zero_pct": all_zero_pct,
        "zero_by_cryo": zero_by_cryo,
    }


def run_spending_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis completo del gasto por servicio.

    Returns:
        dict con per_service (DataFrame) y zero_inflation (dict).
    """
    return {
        "per_service": compute_per_service_stats(df),
        "zero_inflation": compute_zero_inflation(df),
    }
