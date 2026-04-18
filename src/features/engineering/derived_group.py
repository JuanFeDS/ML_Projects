"""Features derivadas de comportamiento y coherencia de grupo."""
import numpy as np
import pandas as pd

from src.features.engineering.encoders import _cryo_to_int
from src.features.engineering.base import _SPENDING_COLS


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


def create_group_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de contexto socioeconómico de grupo y de cabina (fs-002, exp-007).

    Requiere TotalSpending, CabinNumber, Deck, TravelGroup, CryoSleep disponibles.

    Features:
        Route: HomePlanet + '_to_' + Destination.
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


def create_group_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de comportamiento colectivo de grupo (fs-013).

    Requiere create_spending_features y extract_group_features aplicados antes.

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


def create_group_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de coherencia interna de grupo (fs-018).

    Mide homogeneidad del grupo en variables observables (sin usar el target).

    Señal estadística:
    - Grupos donde TODOS van al mismo destino: más cohesivos → mejor señal.
    - Grupos con HomePlanet homogéneo: correlaciona con Deck → refuerza TE.

    Features:
        GroupAllSameDest: 1 si todos los miembros tienen el mismo Destination.
        GroupAllSameHomePlanet: 1 si todos comparten HomePlanet.
        GroupConsistencyScore: suma de los dos anteriores (0-2, ordinal).

    Args:
        df: DataFrame con TravelGroup, Destination, HomePlanet disponibles.

    Returns:
        DataFrame con las 3 features añadidas.
    """
    df_copy = df.copy()

    dest_filled = df_copy["Destination"].fillna("Unknown")
    dest_nunique = df_copy.groupby("TravelGroup")["Destination"].transform(
        lambda x: x.fillna("Unknown").nunique()
    )
    group_size = df_copy.groupby("TravelGroup")["TravelGroup"].transform("count")
    df_copy["GroupAllSameDest"] = (
        ((dest_nunique == 1) & (dest_filled != "Unknown") & (group_size > 1)).astype(int)
    )

    hp_nunique = df_copy.groupby("TravelGroup")["HomePlanet"].transform(
        lambda x: x.fillna("Unknown").nunique()
    )
    hp_filled = df_copy["HomePlanet"].fillna("Unknown")
    df_copy["GroupAllSameHomePlanet"] = (
        ((hp_nunique == 1) & (hp_filled != "Unknown") & (group_size > 1)).astype(int)
    )

    df_copy["GroupConsistencyScore"] = (
        df_copy["GroupAllSameDest"] + df_copy["GroupAllSameHomePlanet"]
    )

    return df_copy
