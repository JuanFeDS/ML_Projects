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
# fs-014
# ---------------------------------------------------------------------------

# p99 calculados sobre el training set (8693 registros) para aplicar en test
# sin data leakage. Fuente: EDA run 2026-04-12.
_P99_THRESHOLDS: dict = {
    "RoomService":  3096.2,
    "FoodCourt":    8033.3,
    "ShoppingMall": 2333.4,
    "Spa":          5390.1,
    "VRDeck":       5646.7,
}

# Medianas de Age por HomePlanet (training set). Fuente: EDA run 2026-04-12.
_PLANET_AGE_MEDIAN: dict = {
    "Earth":   23.0,
    "Europa":  33.0,
    "Mars":    28.0,
}
_GLOBAL_AGE_MEDIAN: float = 27.0  # fallback para HomePlanet desconocido


def create_spend_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de clusters de gasto y contexto grupal ampliado (fs-014).

    Motivación estadística (EDA 2026-04-12):
    - FoodCourt ↔ VRDeck: r=0.46 | Spa ↔ FoodCourt: r=0.42 | Spa ↔ VRDeck: r=0.38
      → cluster de entretenimiento (FoodCourt + VRDeck + Spa).
    - RoomService ↔ ShoppingMall: r=0.36 → cluster de confort, comportamiento diferente.
    - ~86 pasajeros por encima del p99 por servicio (consistente en los 5 servicios).
    - Earth median Age=23, Europa=33, Mars=28 → residual respecto al planeta.
    - GroupCryoSegment (4 niveles): AllCryo 92.2% / AnyCryo 60.4% / Solo 45.2% / NoCryo 33.9%
      captura el gradiente completo mejor que dos binarios independientes.

    Requiere extract_group_features y create_spending_features aplicados antes,
    y CryoSleep disponible en crudo (antes de handle_missing_values_spaceship).

    Features:
        EntertainmentSpend_Log: log1p(FoodCourt + VRDeck + Spa).
        ComfortSpend_Log: log1p(RoomService + ShoppingMall).
        EntVsComfort_Ratio: EntSpend / (ComfortSpend + 1).
        IsExtremeSpender: 1 si algún servicio supera el p99 de entrenamiento.
        AgeVsPlanetMedian: Age - mediana de Age del HomePlanet.
        GroupCryoSegment: ordinal 0-3 (NoCryo=0, Solo=1, AnyCryo=2, AllCryo=3).

    Args:
        df: DataFrame con TravelGroup, CryoSleep, HomePlanet, Age y columnas de gasto.

    Returns:
        DataFrame con las 6 features añadidas.
    """
    df_copy = df.copy()

    # --- Clusters de gasto ---
    entertainment = (
        df_copy["FoodCourt"].fillna(0)
        + df_copy["VRDeck"].fillna(0)
        + df_copy["Spa"].fillna(0)
    )
    comfort = (
        df_copy["RoomService"].fillna(0)
        + df_copy["ShoppingMall"].fillna(0)
    )
    df_copy["EntertainmentSpend_Log"] = np.log1p(entertainment)
    df_copy["ComfortSpend_Log"] = np.log1p(comfort)
    # log-ratio para evitar valores extremos cuando comfort ≈ 0
    df_copy["EntVsComfort_Ratio"] = np.log1p(entertainment) - np.log1p(comfort)

    # --- Extreme spender (p99 del training set) ---
    is_extreme = pd.Series(False, index=df_copy.index)
    for col, threshold in _P99_THRESHOLDS.items():
        if col in df_copy.columns:
            is_extreme = is_extreme | (df_copy[col].fillna(0) > threshold)
    df_copy["IsExtremeSpender"] = is_extreme.astype(int)

    # --- Age vs mediana del planeta ---
    planet_median = (
        df_copy["HomePlanet"]
        .map(_PLANET_AGE_MEDIAN)
        .fillna(_GLOBAL_AGE_MEDIAN)
    )
    df_copy["AgeVsPlanetMedian"] = df_copy["Age"].fillna(_GLOBAL_AGE_MEDIAN) - planet_median

    # --- GroupCryoSegment (ordinal 0-3) ---
    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_int"] = cryo_int
    group_min = df_copy.groupby("TravelGroup")["_cryo_int"].transform("min")
    group_max = df_copy.groupby("TravelGroup")["_cryo_int"].transform("max")
    is_solo = (df_copy.groupby("TravelGroup")["TravelGroup"].transform("count") == 1)

    segment = pd.Series(0, index=df_copy.index, dtype=int)  # NoCryo default
    segment[is_solo] = 1                                      # Solo
    segment[(~is_solo) & (group_max == 1) & (group_min == 0)] = 2  # AnyCryo
    segment[(group_min == 1)] = 3                             # AllCryo

    df_copy["GroupCryoSegment"] = segment
    df_copy = df_copy.drop(columns=["_cryo_int"])

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


# ---------------------------------------------------------------------------
# fs-017 — LastName extraction (proxy de familia + HomePlanet)
# ---------------------------------------------------------------------------

def extract_last_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el apellido de la columna Name.

    El apellido es un proxy potente porque familias viajan juntas y comparten
    HomePlanet, destino y comportamiento. Se espera TE suavizado (k=30) en lugar
    de TE simple para evitar leakage en apellidos raros (n=1-2).

    Señal esperada: +0.01 en accuracy con TE suavizado sin leakage.

    Args:
        df: DataFrame con columna Name (formato "FirstName LastName").

    Returns:
        DataFrame con columna LastName añadida. NaN en Name → "Unknown".
    """
    df_copy = df.copy()
    df_copy["LastName"] = (
        df_copy["Name"]
        .fillna("Unknown Unknown")
        .str.split()
        .str[-1]
        .fillna("Unknown")
    )
    return df_copy


# ---------------------------------------------------------------------------
# fs-018 — Group consistency features (sin uso del target)
# ---------------------------------------------------------------------------

def create_group_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de coherencia interna de grupo (fs-018).

    Mide qué tan homogéneo es el grupo en variables observables (sin usar el
    target Transported). Grupos cohesivos tienden a tener comportamientos más
    predecibles.

    Señal estadística:
    - Grupos donde TODOS van al mismo destino: más cohesivos → mejor señal.
    - Grupos con HomePlanet homogéneo: correlaciona con Deck → refuerza TE.
    - GroupSizeIsOne: viajeros solos tienen tasas de transporte distintas
      a grupos (ya existe IsAlone en fs-003; aquí como binario directo).

    Features:
        GroupAllSameDest: 1 si todos los miembros del grupo tienen el mismo
            Destination (excluyendo Unknown). 0 si hay heterogeneidad.
        GroupAllSameHomePlanet: 1 si todos los miembros del grupo comparten
            HomePlanet (excluyendo Unknown). Complementa Deck_TE y HomePlanet_TE.
        GroupConsistencyScore: suma de GroupAllSameDest + GroupAllSameHomePlanet
            (0-2, score ordinal de cohesión grupal).

    Args:
        df: DataFrame con TravelGroup, Destination, HomePlanet disponibles
            (después de extract_cabin_features y extract_group_features).

    Returns:
        DataFrame con las 3 features añadidas.
    """
    df_copy = df.copy()

    # Destino homogéneo en el grupo
    dest_filled = df_copy["Destination"].fillna("Unknown")
    dest_nunique = df_copy.groupby("TravelGroup")["Destination"].transform(
        lambda x: x.fillna("Unknown").nunique()
    )
    # Solo contar como homogéneo si hay más de 1 miembro Y todos tienen el mismo destino
    group_size = df_copy.groupby("TravelGroup")["TravelGroup"].transform("count")
    df_copy["GroupAllSameDest"] = (
        ((dest_nunique == 1) & (dest_filled != "Unknown") & (group_size > 1)).astype(int)
    )

    # HomePlanet homogéneo en el grupo
    hp_nunique = df_copy.groupby("TravelGroup")["HomePlanet"].transform(
        lambda x: x.fillna("Unknown").nunique()
    )
    hp_filled = df_copy["HomePlanet"].fillna("Unknown")
    df_copy["GroupAllSameHomePlanet"] = (
        ((hp_nunique == 1) & (hp_filled != "Unknown") & (group_size > 1)).astype(int)
    )

    # Score ordinal de cohesión
    df_copy["GroupConsistencyScore"] = (
        df_copy["GroupAllSameDest"] + df_copy["GroupAllSameHomePlanet"]
    )

    return df_copy
