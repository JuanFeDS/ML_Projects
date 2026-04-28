"""Features derivadas de contexto demográfico, familiar y estructural."""

import numpy as np
import pandas as pd

from src.features.engineering.encoders import _cryo_to_int
from src.features.engineering.base import _SPENDING_COLS


def create_structural_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea 7 features de contexto estructural (fs-005).

    Requiere pipeline fs-001 aplicado (TotalSpending, CabinNumber, Deck,
    TravelGroup, CryoSleep, Age, Name disponibles).

    Features:
        SpendingEntropy: entropía de Shannon sobre la distribución de gasto.
        GroupSpendingZScore: desviación del gasto individual respecto al grupo.
        CabinNeighborhoodDensity: pasajeros en el mismo Deck a distancia <= 50.
        FamilySizeFromName: compañeros con el mismo apellido en el dataset.
        GroupCryoAlignment: nivel de consenso del grupo en CryoSleep.
        SpendingCategoryProfile: huella binaria de qué servicios usa el pasajero.
        GroupAgeDispersion: std de Age dentro del TravelGroup.

    Args:
        df: DataFrame con pipeline fs-001 aplicado.

    Returns:
        DataFrame con las 7 features añadidas.
    """
    df_copy = df.copy()

    total = df_copy["TotalSpending"].values
    entropy_vals = np.zeros(len(df_copy))
    for col in _SPENDING_COLS:
        p = df_copy[col].fillna(0).values / (total + 1e-10)
        entropy_vals += np.where(p > 0, -p * np.log(p), 0.0)
    df_copy["SpendingEntropy"] = np.where(total > 0, entropy_vals, 0.0)

    g_mean = df_copy.groupby("TravelGroup")["TotalSpending"].transform("mean")
    g_std = df_copy.groupby("TravelGroup")["TotalSpending"].transform("std").fillna(0)
    df_copy["GroupSpendingZScore"] = (df_copy["TotalSpending"] - g_mean) / (g_std + 1)

    density = np.zeros(len(df_copy))
    cabin_vals = df_copy["CabinNumber"].values
    deck_vals = df_copy["Deck"].values
    for deck in np.unique(deck_vals):
        mask = deck_vals == deck
        idx = np.where(mask)[0]
        cn = cabin_vals[idx]
        diff = np.abs(cn[:, np.newaxis] - cn[np.newaxis, :])
        density[idx] = (diff <= 50).sum(axis=1) - 1
    df_copy["CabinNeighborhoodDensity"] = density

    last_names = df_copy["Name"].apply(
        lambda x: x.split()[-1] if pd.notna(x) and str(x).strip() else None
    )
    family_counts = last_names.value_counts().to_dict()
    df_copy["FamilySizeFromName"] = last_names.map(family_counts).fillna(1).astype(int)

    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_num"] = cryo_int
    g_cryo_rate = df_copy.groupby("TravelGroup")["_cryo_num"].transform("mean")
    df_copy["GroupCryoAlignment"] = np.maximum(g_cryo_rate, 1 - g_cryo_rate)
    df_copy = df_copy.drop(columns=["_cryo_num"])

    profile = pd.Series([""] * len(df_copy), index=df_copy.index)
    for col in _SPENDING_COLS:
        profile = profile + (df_copy[col].fillna(0) > 0).astype(int).astype(str)
    df_copy["SpendingCategoryProfile"] = profile

    df_copy["GroupAgeDispersion"] = (
        df_copy.groupby("TravelGroup")["Age"].transform("std").fillna(0)
    )

    return df_copy


def create_child_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de contexto familiar y ruta (fs-011, fs-012).

    Dirigido a los segmentos con mayor error en exp-013: niños (28% error)
    y destino PSO J318.5-22 (30% error).

    Requiere Age, TravelGroup, HomePlanet, Destination disponibles.

    Features:
        IsChild: 1 si Age < 13.
        GroupHasChild: 1 si el TravelGroup tiene al menos un niño.
        GroupChildRate: proporción de niños en el TravelGroup.
        Route: HomePlanet + '_' + Destination (candidato a target encoding en fs-012).

    Args:
        df: DataFrame con pipeline fs-004 aplicado.

    Returns:
        DataFrame con las 4 features añadidas.
    """
    df_copy = df.copy()

    df_copy["IsChild"] = (df_copy["Age"] < 13).astype(int)

    df_copy["_is_child"] = df_copy["IsChild"]
    df_copy["GroupHasChild"] = (
        df_copy.groupby("TravelGroup")["_is_child"].transform("max")
    ).astype(int)
    df_copy["GroupChildRate"] = df_copy.groupby("TravelGroup")["_is_child"].transform(
        "mean"
    )
    df_copy = df_copy.drop(columns=["_is_child"])

    hp = df_copy["HomePlanet"].fillna("Unknown").astype(str)
    dest = df_copy["Destination"].fillna("Unknown").astype(str)
    df_copy["Route"] = hp + "_" + dest

    return df_copy


def extract_last_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el apellido de la columna Name (fs-017).

    El apellido es proxy potente: familias viajan juntas y comparten HomePlanet,
    destino y comportamiento. Se espera TE suavizado (k=30) para evitar leakage
    en apellidos raros (n=1-2).

    Args:
        df: DataFrame con columna Name (formato "FirstName LastName").

    Returns:
        DataFrame con columna LastName añadida. NaN en Name → "Unknown".
    """
    df_copy = df.copy()
    df_copy["LastName"] = (
        df_copy["Name"].fillna("Unknown Unknown").str.split().str[-1].fillna("Unknown")
    )
    return df_copy
