"""
Features derivadas por experimento para Spaceship Titanic.

Cada función añade un conjunto de features específico, motivado por análisis
estadístico previo. Las funciones son aditivas y puras (nunca mutan el input).

Convención de nombrado: create_<nombre_descriptivo>_features.
"""
import pandas as pd
import numpy as np

from src.features.engineering.encoders import _cryo_to_int
from src.features.engineering.base import _SPENDING_COLS


# ---------------------------------------------------------------------------
# fs-002 / exp-007
# ---------------------------------------------------------------------------

def create_group_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de contexto socioeconómico de grupo y de cabina.

    Usadas en fs-002 (cryo_interactions). Requiere TotalSpending, CabinNumber,
    Deck, TravelGroup, CryoSleep disponibles.

    Features:
        Route: HomePlanet + '_to_' + Destination — trayectoria completa.
        GroupCryoSleepRate: fracción de miembros del grupo en CryoSleep.
        CryoSleepViolation: 1 si CryoSleep=True pero TotalSpending > 0.
        LuxurySpendingRatio: (Spa + VRDeck) / (TotalSpending + 1).
        CabinNumber_DeckPercentile: posición relativa dentro del deck.
        GroupSpendingMean: media de TotalSpending del grupo.

    Args:
        df: DataFrame con pipeline fs-001 aplicado.

    Returns:
        DataFrame con las 6 features añadidas.
    """
    df_copy = df.copy()

    hp = df_copy["HomePlanet"].fillna("Unknown").astype(str)
    dest = df_copy["Destination"].fillna("Unknown").astype(str)
    df_copy["Route"] = hp + "_to_" + dest

    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_num"] = cryo_int
    df_copy["GroupCryoSleepRate"] = (
        df_copy.groupby("TravelGroup")["_cryo_num"].transform("mean")
    )
    df_copy = df_copy.drop(columns=["_cryo_num"])

    cryo_true = df_copy["CryoSleep"].isin([True, "True"])
    total_spend = df_copy[_SPENDING_COLS].fillna(0).sum(axis=1)
    df_copy["CryoSleepViolation"] = (cryo_true & (total_spend > 0)).astype(int)

    luxury = df_copy["Spa"].fillna(0) + df_copy["VRDeck"].fillna(0)
    df_copy["LuxurySpendingRatio"] = luxury / (df_copy["TotalSpending"] + 1)

    df_copy = _add_cabin_percentile(df_copy)

    df_copy["GroupSpendingMean"] = (
        df_copy.groupby("TravelGroup")["TotalSpending"].transform("mean")
    )

    return df_copy


def _add_cabin_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Añade CabinNumber_DeckPercentile: posición relativa de la cabina en su deck."""
    df_copy = df.copy()
    deck_min = df_copy.groupby("Deck")["CabinNumber"].transform("min")
    deck_max = df_copy.groupby("Deck")["CabinNumber"].transform("max")
    span = (deck_max - deck_min).replace(0, 1)
    df_copy["CabinNumber_DeckPercentile"] = (
        (df_copy["CabinNumber"] - deck_min) / span
    ).fillna(0.5)
    return df_copy


# ---------------------------------------------------------------------------
# fs-003
# ---------------------------------------------------------------------------

def create_solo_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea IsAlone, IsChild y SpendingIntensity (fs-003).

    Requiere GroupSize, Age y TotalSpending/SpendingCategories disponibles.

    Features:
        IsAlone: 1 si GroupSize == 1.
        IsChild: 1 si Age < 13.
        SpendingIntensity: TotalSpending / (SpendingCategories + 1).

    Args:
        df: DataFrame con pipeline fs-001 aplicado.

    Returns:
        DataFrame con las 3 features añadidas.
    """
    df_copy = df.copy()
    df_copy["IsAlone"] = (df_copy["GroupSize"] == 1).astype(int)
    df_copy["IsChild"] = (df_copy["Age"] < 13).astype(int)
    df_copy["SpendingIntensity"] = (
        df_copy["TotalSpending"] / (df_copy["SpendingCategories"] + 1)
    )
    return df_copy


# ---------------------------------------------------------------------------
# fs-005
# ---------------------------------------------------------------------------

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

    # SpendingEntropy
    total = df_copy["TotalSpending"].values
    entropy_vals = np.zeros(len(df_copy))
    for col in _SPENDING_COLS:
        p = df_copy[col].fillna(0).values / (total + 1e-10)
        entropy_vals += np.where(p > 0, -p * np.log(p), 0.0)
    df_copy["SpendingEntropy"] = np.where(total > 0, entropy_vals, 0.0)

    # GroupSpendingZScore
    g_mean = df_copy.groupby("TravelGroup")["TotalSpending"].transform("mean")
    g_std = df_copy.groupby("TravelGroup")["TotalSpending"].transform("std").fillna(0)
    df_copy["GroupSpendingZScore"] = (df_copy["TotalSpending"] - g_mean) / (g_std + 1)

    # CabinNeighborhoodDensity (vectorizado por Deck)
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

    # FamilySizeFromName
    last_names = df_copy["Name"].apply(
        lambda x: x.split()[-1] if pd.notna(x) and str(x).strip() else None
    )
    family_counts = last_names.value_counts().to_dict()
    df_copy["FamilySizeFromName"] = last_names.map(family_counts).fillna(1).astype(int)

    # GroupCryoAlignment
    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_num"] = cryo_int
    g_cryo_rate = df_copy.groupby("TravelGroup")["_cryo_num"].transform("mean")
    df_copy["GroupCryoAlignment"] = np.maximum(g_cryo_rate, 1 - g_cryo_rate)
    df_copy = df_copy.drop(columns=["_cryo_num"])

    # SpendingCategoryProfile
    profile = pd.Series([""] * len(df_copy), index=df_copy.index)
    for col in _SPENDING_COLS:
        profile = profile + (df_copy[col].fillna(0) > 0).astype(int).astype(str)
    df_copy["SpendingCategoryProfile"] = profile

    # GroupAgeDispersion
    df_copy["GroupAgeDispersion"] = (
        df_copy.groupby("TravelGroup")["Age"].transform("std").fillna(0)
    )

    return df_copy


# ---------------------------------------------------------------------------
# fs-010
# ---------------------------------------------------------------------------

def create_cryo_spending_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de interacción CryoSleep × spending (fs-010).

    Requiere pipeline fs-004 aplicado (TotalSpending, CryoSleep, TravelGroup,
    Side disponibles).

    Features:
        CryoSpendingAnomaly: gasto total cuando CryoSleep=True (debería ser 0).
        GroupTransportedProxy: ratio de miembros del grupo con HasSpending=0.
        SideSpendingDiff: diferencia de gasto medio entre lados P y S.
        CryoSleepBinary: CryoSleep numérico (1/0/-1 para Unknown).

    Args:
        df: DataFrame con CryoSleep, TotalSpending, TravelGroup, Side.

    Returns:
        DataFrame con las 4 features añadidas.
    """
    df_copy = df.copy()

    cryo_flag = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["CryoSpendingAnomaly"] = cryo_flag * df_copy["TotalSpending"].fillna(0)

    no_spend = (df_copy["TotalSpending"].fillna(0) == 0).astype(int)
    df_copy["_no_spend"] = no_spend
    df_copy["GroupTransportedProxy"] = (
        df_copy.groupby("TravelGroup")["_no_spend"].transform("mean")
    )
    df_copy = df_copy.drop(columns=["_no_spend"])

    cabin_p = df_copy["CabinNumber"].map(
        df_copy[df_copy["Side"] == "P"].groupby("CabinNumber")["TotalSpending"].mean()
    ).fillna(0)
    cabin_s = df_copy["CabinNumber"].map(
        df_copy[df_copy["Side"] == "S"].groupby("CabinNumber")["TotalSpending"].mean()
    ).fillna(0)
    df_copy["SideSpendingDiff"] = np.abs(cabin_p - cabin_s)

    df_copy["CryoSleepBinary"] = df_copy["CryoSleep"].map(
        {True: 1, "True": 1, False: 0, "False": 0, "Unknown": -1}
    ).fillna(-1)

    return df_copy


# ---------------------------------------------------------------------------
# fs-011 / fs-012
# ---------------------------------------------------------------------------

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
    df_copy["GroupChildRate"] = (
        df_copy.groupby("TravelGroup")["_is_child"].transform("mean")
    )
    df_copy = df_copy.drop(columns=["_is_child"])

    hp = df_copy["HomePlanet"].fillna("Unknown").astype(str)
    dest = df_copy["Destination"].fillna("Unknown").astype(str)
    df_copy["Route"] = hp + "_" + dest

    return df_copy


# ---------------------------------------------------------------------------
# fs-013
# ---------------------------------------------------------------------------

def create_group_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de comportamiento colectivo de grupo (fs-013).

    Inspiradas en soluciones top de Kaggle (>0.83). Requiere create_spending_features
    y extract_group_features aplicados antes (TotalSpending y TravelGroup disponibles).

    Señal estadística:
    - Grupos donde TODOS están en CryoSleep: 80.5% transported vs 42.4% (nadie).
    - SpendShare (gasto individual / grupo): corr=-0.15 entre no-CryoSleep.
    - GroupSpendOthers_Log (gasto del resto del grupo): corr=+0.09.

    Features:
        GroupAllCryo: 1 si todos los miembros del grupo están en CryoSleep.
        GroupAnyCryo: 1 si al menos un miembro está en CryoSleep.
        SpendShare: TotalSpending_i / (TotalSpending_grupo + 1).
        GroupSpendOthers_Log: log1p del gasto del resto del grupo.

    Args:
        df: DataFrame con TravelGroup, CryoSleep y columnas de gasto individuales.

    Returns:
        DataFrame con las 4 features añadidas.
    """
    df_copy = df.copy()

    available = [c for c in _SPENDING_COLS if c in df_copy.columns]
    df_copy["_raw_spend"] = df_copy[available].fillna(0).sum(axis=1)

    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_int"] = cryo_int
    df_copy["GroupAllCryo"] = (
        df_copy.groupby("TravelGroup")["_cryo_int"].transform("min").astype(int)
    )
    df_copy["GroupAnyCryo"] = (
        df_copy.groupby("TravelGroup")["_cryo_int"].transform("max").astype(int)
    )
    df_copy = df_copy.drop(columns=["_cryo_int"])

    group_total = df_copy.groupby("TravelGroup")["_raw_spend"].transform("sum")
    df_copy["SpendShare"] = df_copy["_raw_spend"] / (group_total + 1)
    df_copy["GroupSpendOthers_Log"] = np.log1p(group_total - df_copy["_raw_spend"])

    df_copy = df_copy.drop(columns=["_raw_spend"])
    return df_copy
