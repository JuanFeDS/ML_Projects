"""
Registro de Feature Sets — Spaceship Titanic.

Cada FeatureSetConfig define una configuracion inmutable de features.

CONVENCION:
  - Una vez creado un feature set, NO modificarlo.
  - Para nuevas ideas, agregar una nueva entrada a FEATURE_SETS.
  - Las funciones de engineering.py son aditivas: nunca se borran, solo se anaden.

Uso en scripts:
    from src.features.feature_sets import FEATURE_SETS, DEFAULT_FEATURE_SET
    fs = FEATURE_SETS["fs-001_baseline"]
    df_train = fs.pipeline(df_raw)
    df_test  = fs.test_pipeline(df_raw_test)
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from src.features.engineering import (
    apply_domain_rules,
    create_age_features,
    create_exp007_features,
    create_fs003_features,
    create_fs005_features,
    create_spending_features,
    extract_cabin_features,
    extract_group_features,
    handle_missing_values_spaceship,
    impute_spending_group_aware,
)

_CATEGORICAL_FILL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class FeatureSetConfig:
    """Configuracion inmutable de un feature set.

    Attributes:
        description: Descripcion del feature set y sus diferencias vs parent.
        pipeline: Funcion que transforma el DataFrame de train (elimina Age NaN).
        test_pipeline: Funcion que transforma el DataFrame de test (imputa Age NaN).
        numeric_features: Features numericas a escalar con StandardScaler.
        categorical_cols: Columnas a codificar con One-Hot Encoding.
        features_to_drop: Columnas a eliminar antes del entrenamiento.
        target_encode_cols: Columnas a codificar con Target Encoding (media del target
            por categoria). Se excluyen de categorical_cols automaticamente.
        parent: Nombre del feature set del que hereda (None si es el primero).
    """

    description: str
    pipeline: Callable[[pd.DataFrame], pd.DataFrame]
    test_pipeline: Callable[[pd.DataFrame], pd.DataFrame]
    numeric_features: List[str]
    categorical_cols: List[str]
    features_to_drop: List[str]
    target_encode_cols: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    deprecated: bool = False


# ---------------------------------------------------------------------------
# Pipelines de train (eliminan filas con Age NaN)
# ---------------------------------------------------------------------------

def _pipeline_fs001(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-001: features base (train)."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=False)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs003(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-003: fs-001 + IsAlone, IsChild, SpendingIntensity (train)."""
    df_out = _pipeline_fs001(df)
    df_out = create_fs003_features(df_out)
    return df_out


def _pipeline_fs002(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-002: fs-001 + interacciones cryo/spending/cabin (train)."""
    df_out = _pipeline_fs001(df)
    df_out = create_exp007_features(df_out)
    return df_out


def _pipeline_fs004(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-004: fs-001 base (train). Target encoding aplicado despues."""
    return _pipeline_fs001(df)


# ---------------------------------------------------------------------------
# Pipelines de test (imputan Age NaN en lugar de eliminar filas)
# ---------------------------------------------------------------------------

def _pipeline_fs001_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-001: features base (test)."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=True)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs003_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-003: fs-001 + IsAlone, IsChild, SpendingIntensity (test)."""
    df_out = _pipeline_fs001_test(df)
    df_out = create_fs003_features(df_out)
    return df_out


def _pipeline_fs002_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-002: fs-001 + interacciones cryo/spending/cabin (test)."""
    df_out = _pipeline_fs001_test(df)
    df_out = create_exp007_features(df_out)
    return df_out


def _pipeline_fs004_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-004: fs-001 base (test). Target encoding aplicado despues."""
    return _pipeline_fs001_test(df)


def _pipeline_fs005(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-005: fs-001 + 7 features estructurales/contextuales (train)."""
    df_out = _pipeline_fs001(df)
    df_out = create_fs005_features(df_out)
    return df_out


def _pipeline_fs005_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-005: fs-001 + 7 features estructurales/contextuales (test)."""
    df_out = _pipeline_fs001_test(df)
    df_out = create_fs005_features(df_out)
    return df_out


# ---------------------------------------------------------------------------
# Configuraciones de features por set
# ---------------------------------------------------------------------------

_FS001_NUMERIC: List[str] = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupSize",
    "CabinNumber",
    "TotalSpending_Log",
    "SpendingCategories",
]
_FS001_CATEGORICAL: List[str] = ["HomePlanet", "Destination", "Deck", "AgeCategory"]
_FS001_DROP: List[str] = [
    "PassengerId",
    "Name",
    "Cabin",
    "TravelGroup",
    "CryoSleep",
    "VIP",
    "Side",
    "TotalSpending",
]

_FS003_NUMERIC: List[str] = _FS001_NUMERIC + [
    "IsAlone",
    "IsChild",
    "SpendingIntensity",
]
_FS003_CATEGORICAL: List[str] = _FS001_CATEGORICAL
_FS003_DROP: List[str] = _FS001_DROP

_FS002_NUMERIC: List[str] = _FS001_NUMERIC + [
    "GroupCryoSleepRate",
    "CryoSleepViolation",
    "LuxurySpendingRatio",
    "CabinNumber_DeckPercentile",
    "GroupSpendingMean",
]
_FS002_CATEGORICAL: List[str] = _FS001_CATEGORICAL + ["Route"]
_FS002_DROP: List[str] = _FS001_DROP

# fs-004: Deck y HomePlanet pasan a target encoding (salen de categorical_cols)
_FS004_TARGET_ENCODE: List[str] = ["Deck", "HomePlanet"]
_FS004_NUMERIC: List[str] = _FS001_NUMERIC + ["Deck_TE", "HomePlanet_TE"]
_FS004_CATEGORICAL: List[str] = ["Destination", "AgeCategory"]  # Deck y HomePlanet → TE
_FS004_DROP: List[str] = _FS001_DROP

# fs-005: fs-001 + 7 features estructurales/contextuales
# SpendingCategoryProfile → target encoding (perfil de consumo → señal ordinal)
_FS005_TARGET_ENCODE: List[str] = ["SpendingCategoryProfile"]
_FS005_NUMERIC: List[str] = _FS001_NUMERIC + [
    "SpendingEntropy",
    "GroupSpendingZScore",
    "CabinNeighborhoodDensity",
    "FamilySizeFromName",
    "GroupCryoAlignment",
    "GroupAgeDispersion",
    "SpendingCategoryProfile_TE",
]
_FS005_CATEGORICAL: List[str] = _FS001_CATEGORICAL
_FS005_DROP: List[str] = _FS001_DROP


# ---------------------------------------------------------------------------
# Registro principal
# ---------------------------------------------------------------------------

FEATURE_SETS: dict = {
    "fs-001_baseline": FeatureSetConfig(
        description=(
            "Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, "
            "spending log+categorias, AgeCategory. "
            "Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227)."
        ),
        parent=None,
        pipeline=_pipeline_fs001,
        test_pipeline=_pipeline_fs001_test,
        numeric_features=_FS001_NUMERIC,
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
    ),
    "fs-003_solo_interactions": FeatureSetConfig(
        description=(
            "fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), "
            "SpendingIntensity (TotalSpending/(SpendingCategories+1)). "
            "Features simples de alta senal, sin riesgo de multicolinealidad."
        ),
        parent="fs-001_baseline",
        pipeline=_pipeline_fs003,
        test_pipeline=_pipeline_fs003_test,
        numeric_features=_FS003_NUMERIC,
        categorical_cols=_FS003_CATEGORICAL,
        features_to_drop=_FS003_DROP,
    ),
    "fs-002_cryo_interactions": FeatureSetConfig(
        description=(
            "fs-001 + Route (HomePlanet+Destination), GroupCryoSleepRate, "
            "CryoSleepViolation, LuxurySpendingRatio, CabinNumber_DeckPercentile, "
            "GroupSpendingMean. "
            "Referencia: Exp-007 (val_accuracy=0.8156, no supero fs-001)."
        ),
        parent="fs-001_baseline",
        pipeline=_pipeline_fs002,
        test_pipeline=_pipeline_fs002_test,
        numeric_features=_FS002_NUMERIC,
        categorical_cols=_FS002_CATEGORICAL,
        features_to_drop=_FS002_DROP,
    ),
    "fs-004_target_encoding": FeatureSetConfig(
        description=(
            "fs-001 con Deck y HomePlanet reemplazados por Target Encoding "
            "(media del target por categoria, con suavizado). "
            "Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad "
            "y aportan informacion ordinal que OHE no captura."
        ),
        parent="fs-001_baseline",
        pipeline=_pipeline_fs004,
        test_pipeline=_pipeline_fs004_test,
        numeric_features=_FS004_NUMERIC,
        categorical_cols=_FS004_CATEGORICAL,
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS004_TARGET_ENCODE,
    ),
    "fs-005_structural_context": FeatureSetConfig(
        description=(
            "fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), "
            "GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity "
            "(densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), "
            "GroupCryoAlignment (consenso CryoSleep en el grupo), "
            "GroupAgeDispersion (std Age por grupo), "
            "SpendingCategoryProfile → TE (patron de servicios usados)."
        ),
        parent="fs-001_baseline",
        pipeline=_pipeline_fs005,
        test_pipeline=_pipeline_fs005_test,
        numeric_features=_FS005_NUMERIC,
        categorical_cols=_FS005_CATEGORICAL,
        features_to_drop=_FS005_DROP,
        target_encode_cols=_FS005_TARGET_ENCODE,
    ),
}

def _pipeline_fs006(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-006: imputation group-aware antes de calcular spending (train).

    Diferencia vs fs-001: el orden es extract → fill categoricals → impute spending
    group-aware → create_spending_features. Esto hace que TotalSpending_Log refleje
    la mediana del grupo en lugar de 0 para pasajeros no-cryo con spending nulo.
    """
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_age_features(df_out)
    for col in _CATEGORICAL_FILL_COLS:
        df_out[col] = df_out[col].fillna("Unknown")
    df_out = impute_spending_group_aware(df_out)
    df_out = create_spending_features(df_out)
    df_out = df_out.dropna(subset=["Age"])
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs006_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-006: imputation group-aware antes de calcular spending (test)."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_age_features(df_out)
    for col in _CATEGORICAL_FILL_COLS:
        df_out[col] = df_out[col].fillna("Unknown")
    df_out = impute_spending_group_aware(df_out)
    df_out = create_spending_features(df_out)
    df_out["Age"] = df_out["Age"].fillna(df_out["Age"].median())
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs007(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-007: domain rules + TravelGroup_TE (train).

    Orden: extract → domain rules → fill restantes → spending → age → drop Age NaN.
    El TravelGroup_TE se calcula en 02_features.py (target encoding del grupo).
    """
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=False)
    df_out = create_spending_features(df_out)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs007_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-007: domain rules + TravelGroup_TE (test)."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=True)
    df_out = create_spending_features(df_out)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


# fs-007: TravelGroup fuera de DROP (lo gestiona target_encode_cols)
_FS007_TARGET_ENCODE: List[str] = ["TravelGroup"]
_FS007_DROP: List[str] = [c for c in _FS001_DROP if c != "TravelGroup"]
_FS007_NUMERIC: List[str] = _FS001_NUMERIC + ["TravelGroup_TE"]
_FS007_CATEGORICAL: List[str] = _FS001_CATEGORICAL

FEATURE_SETS["fs-007_domain_rules"] = FeatureSetConfig(
    description=(
        "Imputacion por 6 reglas fisicas del dataset + TravelGroup_TE. "
        "Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, "
        "Deck/Side por grupo, CryoSleep=True→spending=0, "
        "spending>0→CryoSleep=False, Age<=12→spending=0. "
        "TravelGroup_TE: tasa de transporte media del grupo de viaje (target encoding)."
    ),
    parent="fs-001_baseline",
    pipeline=_pipeline_fs007,
    test_pipeline=_pipeline_fs007_test,
    numeric_features=_FS007_NUMERIC,
    categorical_cols=_FS007_CATEGORICAL,
    features_to_drop=_FS007_DROP,
    target_encode_cols=_FS007_TARGET_ENCODE,
)

def _pipeline_fs009(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-009: domain rules + CabinNumber_DeckPercentile (train).

    Reemplaza CabinNumber por CabinNumber_DeckPercentile para reducir el
    distributional shift detectado en adversarial validation (AUC=0.79).
    CabinNumber_DeckPercentile normaliza la posicion relativa dentro del deck,
    haciendo la feature invariante al rango absoluto de cabinas en test.
    """
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=False)
    df_out = create_spending_features(df_out)
    # CabinNumber_DeckPercentile: posicion relativa dentro del deck
    deck_min = df_out.groupby("Deck")["CabinNumber"].transform("min")
    deck_max = df_out.groupby("Deck")["CabinNumber"].transform("max")
    span = (deck_max - deck_min).replace(0, 1)
    df_out["CabinNumber_DeckPercentile"] = (
        (df_out["CabinNumber"] - deck_min) / span
    ).fillna(0.5)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


def _pipeline_fs009_test(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de fs-009: domain rules + CabinNumber_DeckPercentile (test)."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=True)
    df_out = create_spending_features(df_out)
    deck_min = df_out.groupby("Deck")["CabinNumber"].transform("min")
    deck_max = df_out.groupby("Deck")["CabinNumber"].transform("max")
    span = (deck_max - deck_min).replace(0, 1)
    df_out["CabinNumber_DeckPercentile"] = (
        (df_out["CabinNumber"] - deck_min) / span
    ).fillna(0.5)
    df_out["GroupSize"] = df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    return df_out


# fs-009: CabinNumber reemplazado por CabinNumber_DeckPercentile
_FS009_NUMERIC: List[str] = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber_DeckPercentile", "TotalSpending_Log", "SpendingCategories",
]
_FS009_DROP: List[str] = _FS001_DROP + ["CabinNumber"]

FEATURE_SETS["fs-009_percentile_cabin"] = FeatureSetConfig(
    description=(
        "fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. "
        "Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor "
        "distributional shift entre train y test. La percentil normaliza la posicion "
        "relativa dentro del deck, eliminando el shift de rango absoluto."
    ),
    parent="fs-008_domain_rules_only",
    pipeline=_pipeline_fs009,
    test_pipeline=_pipeline_fs009_test,
    numeric_features=_FS009_NUMERIC,
    categorical_cols=_FS001_CATEGORICAL,
    features_to_drop=_FS009_DROP,
)

FEATURE_SETS["fs-008_domain_rules_only"] = FeatureSetConfig(
    description=(
        "Imputacion por 6 reglas fisicas del dataset SIN target encoding de grupo. "
        "Elimina el leakage de TravelGroup_TE (fs-007). "
        "Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, "
        "Deck/Side por grupo, CryoSleep=True→spending=0, "
        "spending>0→CryoSleep=False, Age<=12→spending=0. "
        "Mismo pipeline que fs-007 pero con el mismo espacio de features que fs-001."
    ),
    parent="fs-001_baseline",
    pipeline=_pipeline_fs007,
    test_pipeline=_pipeline_fs007_test,
    numeric_features=_FS001_NUMERIC,
    categorical_cols=_FS001_CATEGORICAL,
    features_to_drop=_FS001_DROP,
)

FEATURE_SETS["fs-006_group_imputation"] = FeatureSetConfig(
    description=(
        "fs-001 con imputacion group-aware para columnas de gasto. "
        "Pasajeros no-cryo con spending NaN reciben la mediana del TravelGroup "
        "(en lugar de 0), haciendo que TotalSpending_Log capture mejor su perfil real. "
        "El orden del pipeline cambia: fill categoricals → impute → create_spending_features."
    ),
    parent="fs-001_baseline",
    pipeline=_pipeline_fs006,
    test_pipeline=_pipeline_fs006_test,
    numeric_features=_FS001_NUMERIC,
    categorical_cols=_FS001_CATEGORICAL,
    features_to_drop=_FS001_DROP,
)

DEFAULT_FEATURE_SET: str = "fs-001_baseline"
