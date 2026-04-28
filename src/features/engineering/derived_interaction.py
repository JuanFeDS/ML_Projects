"""Features derivadas de interacciones de gasto, CryoSleep y clusters."""

import numpy as np
import pandas as pd

from src.features.engineering.encoders import _cryo_to_int

# p99 calculados sobre el training set (8693 registros).
# Fuente: EDA run 2026-04-12. Aplicados en test sin data leakage.
_P99_THRESHOLDS: dict = {
    "RoomService": 3096.2,
    "FoodCourt": 8033.3,
    "ShoppingMall": 2333.4,
    "Spa": 5390.1,
    "VRDeck": 5646.7,
}

# Medianas de Age por HomePlanet (training set). Fuente: EDA run 2026-04-12.
_PLANET_AGE_MEDIAN: dict = {
    "Earth": 23.0,
    "Europa": 33.0,
    "Mars": 28.0,
}
_GLOBAL_AGE_MEDIAN: float = 27.0  # fallback para HomePlanet desconocido


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
    df_copy["SpendingIntensity"] = df_copy["TotalSpending"] / (
        df_copy["SpendingCategories"] + 1
    )
    return df_copy


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
    df_copy["GroupTransportedProxy"] = df_copy.groupby("TravelGroup")[
        "_no_spend"
    ].transform("mean")
    df_copy = df_copy.drop(columns=["_no_spend"])

    cabin_p = (
        df_copy["CabinNumber"]
        .map(
            df_copy[df_copy["Side"] == "P"]
            .groupby("CabinNumber")["TotalSpending"]
            .mean()
        )
        .fillna(0)
    )
    cabin_s = (
        df_copy["CabinNumber"]
        .map(
            df_copy[df_copy["Side"] == "S"]
            .groupby("CabinNumber")["TotalSpending"]
            .mean()
        )
        .fillna(0)
    )
    df_copy["SideSpendingDiff"] = np.abs(cabin_p - cabin_s)

    df_copy["CryoSleepBinary"] = (
        df_copy["CryoSleep"]
        .map({True: 1, "True": 1, False: 0, "False": 0, "Unknown": -1})
        .fillna(-1)
    )

    return df_copy


def create_spend_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de clusters de gasto y contexto grupal ampliado (fs-014).

    Motivación estadística (EDA 2026-04-12):
    - FoodCourt ↔ VRDeck: r=0.46 | Spa ↔ FoodCourt: r=0.42 | Spa ↔ VRDeck: r=0.38
      → cluster de entretenimiento (FoodCourt + VRDeck + Spa).
    - RoomService ↔ ShoppingMall: r=0.36 → cluster de confort.
    - ~86 pasajeros por encima del p99 por servicio.
    - GroupCryoSegment (4 niveles): AllCryo 92.2% / AnyCryo 60.4% / Solo 45.2% / NoCryo 33.9%.

    Requiere extract_group_features y create_spending_features aplicados antes.

    Features:
        EntertainmentSpend_Log: log1p(FoodCourt + VRDeck + Spa).
        ComfortSpend_Log: log1p(RoomService + ShoppingMall).
        EntVsComfort_Ratio: log-ratio entertainment vs comfort.
        IsExtremeSpender: 1 si algún servicio supera el p99 de entrenamiento.
        AgeVsPlanetMedian: Age - mediana de Age del HomePlanet.
        GroupCryoSegment: ordinal 0-3 (NoCryo=0, Solo=1, AnyCryo=2, AllCryo=3).

    Args:
        df: DataFrame con TravelGroup, CryoSleep, HomePlanet, Age y columnas de gasto.

    Returns:
        DataFrame con las 6 features añadidas.
    """
    df_copy = df.copy()

    entertainment = (
        df_copy["FoodCourt"].fillna(0)
        + df_copy["VRDeck"].fillna(0)
        + df_copy["Spa"].fillna(0)
    )
    comfort = df_copy["RoomService"].fillna(0) + df_copy["ShoppingMall"].fillna(0)
    df_copy["EntertainmentSpend_Log"] = np.log1p(entertainment)
    df_copy["ComfortSpend_Log"] = np.log1p(comfort)
    df_copy["EntVsComfort_Ratio"] = np.log1p(entertainment) - np.log1p(comfort)

    is_extreme = pd.Series(False, index=df_copy.index)
    for col, threshold in _P99_THRESHOLDS.items():
        if col in df_copy.columns:
            is_extreme = is_extreme | (df_copy[col].fillna(0) > threshold)
    df_copy["IsExtremeSpender"] = is_extreme.astype(int)

    planet_median = (
        df_copy["HomePlanet"].map(_PLANET_AGE_MEDIAN).fillna(_GLOBAL_AGE_MEDIAN)
    )
    df_copy["AgeVsPlanetMedian"] = (
        df_copy["Age"].fillna(_GLOBAL_AGE_MEDIAN) - planet_median
    )

    cryo_int = _cryo_to_int(df_copy["CryoSleep"])
    df_copy["_cryo_int"] = cryo_int
    group_min = df_copy.groupby("TravelGroup")["_cryo_int"].transform("min")
    group_max = df_copy.groupby("TravelGroup")["_cryo_int"].transform("max")
    is_solo = df_copy.groupby("TravelGroup")["TravelGroup"].transform("count") == 1

    segment = pd.Series(0, index=df_copy.index, dtype=int)
    segment[is_solo] = 1
    segment[(~is_solo) & (group_max == 1) & (group_min == 0)] = 2
    segment[(group_min == 1)] = 3

    df_copy["GroupCryoSegment"] = segment
    df_copy = df_copy.drop(columns=["_cryo_int"])

    return df_copy
