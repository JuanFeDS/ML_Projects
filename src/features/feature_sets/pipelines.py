"""
Funciones de pipeline para cada feature set.

Convención: cada pipeline acepta `impute_age: bool = False`.
El registry los instancia con lambdas:
    pipeline=lambda df: _pipeline_fsXXX(df),
    test_pipeline=lambda df: _pipeline_fsXXX(df, impute_age=True),

Esto elimina la duplicación train/test — solo existe una función por feature set.
"""
import pandas as pd

from src.features.engineering.base import (
    _CATEGORICAL_FILL,
    apply_domain_rules,
    create_age_features,
    create_spending_features,
    extract_cabin_features,
    extract_group_features,
    handle_missing_values_spaceship,
    impute_age_by_group,
    impute_spending_group_aware,
)
from src.features.engineering.derived import (
    _add_cabin_percentile,
    create_child_route_features,
    create_cryo_spending_interaction_features,
    create_group_consistency_features,
    create_group_context_features,
    create_group_spending_features,
    create_solo_interaction_features,
    create_spend_cluster_features,
    create_structural_context_features,
    extract_last_name,
)


def _add_group_size(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula GroupSize tras eliminar filas (train) o imputar (test)."""
    df_out = df.copy()
    df_out["GroupSize"] = (
        df_out.groupby("TravelGroup")["TravelGroup"].transform("count")
    )
    return df_out


# ---------------------------------------------------------------------------
# Pipelines activos
# ---------------------------------------------------------------------------

def _pipeline_fs001(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-001: features base."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    return _add_group_size(df_out)


def _pipeline_fs003(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-003: fs-001 + IsAlone, IsChild, SpendingIntensity."""
    df_out = _pipeline_fs001(df, impute_age=impute_age)
    return create_solo_interaction_features(df_out)


def _pipeline_fs004(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-004: fs-001 base (target encoding se aplica después en 02_features.py)."""
    return _pipeline_fs001(df, impute_age=impute_age)


def _pipeline_fs005(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-005: fs-001 + 7 features de contexto estructural."""
    df_out = _pipeline_fs001(df, impute_age=impute_age)
    return create_structural_context_features(df_out)


def _pipeline_fs010(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-010: fs-004 + interacciones CryoSleep × spending."""
    df_out = _pipeline_fs004(df, impute_age=impute_age)
    return create_cryo_spending_interaction_features(df_out)


def _pipeline_fs011(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-011: fs-004 + contexto familiar + Route (OHE)."""
    df_out = _pipeline_fs004(df, impute_age=impute_age)
    return create_child_route_features(df_out)


def _pipeline_fs013(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-013: fs-004 + Age imputada por grupo + 4 features de contexto colectivo."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = impute_age_by_group(df_out)
    df_out = create_age_features(df_out)
    df_out = create_group_context_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    return _add_group_size(df_out)


def _pipeline_fs015(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-015: fs-004 + imputación agresiva por reglas de dominio antes de engineering.

    La diferencia respecto a fs-004 es el orden: apply_domain_rules corre antes de
    create_spending_features, de modo que los NaN de HomePlanet/CryoSleep/Deck/Side
    se resuelven por inferencia (grupo, deck→planeta, gasto>0) en lugar de caer
    en la categoría 'Unknown'. Misma dimensionalidad que fs-004.
    """
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)        # imputa antes de crear features
    df_out = create_spending_features(df_out)
    df_out = impute_age_by_group(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    return _add_group_size(df_out)


def _pipeline_fs014(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-014: fs-013 + clusters de gasto + GroupCryoSegment + AgeVsPlanetMedian + IsExtremeSpender."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = impute_age_by_group(df_out)
    df_out = create_age_features(df_out)
    df_out = create_group_context_features(df_out)     # GroupAllCryo, GroupAnyCryo, SpendShare, GroupSpendOthers_Log
    df_out = create_spend_cluster_features(df_out)     # clusters + GroupCryoSegment + AgeVsPlanetMedian + IsExtremeSpender
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    return _add_group_size(df_out)


def _pipeline_fs017(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-017: fs-004 + LastName extraído de Name (para fold-aware TE).

    Extract last name antes de que Name sea eliminado por features_to_drop.
    El TE fold-aware se aplica dentro del Pipeline de entrenamiento via
    sklearn TargetEncoder(cv=5), eliminando leakage en apellidos raros.
    """
    df_out = _pipeline_fs004(df, impute_age=impute_age)
    return extract_last_name(df_out)


def _pipeline_fs018(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-018: fs-017 + 3 features de consistencia interna de grupo.

    GroupAllSameDest, GroupAllSameHomePlanet y GroupConsistencyScore miden
    cohesión observable del grupo sin usar el target (sin leakage).
    """
    df_out = _pipeline_fs017(df, impute_age=impute_age)
    return create_group_consistency_features(df_out)


def _pipeline_fs019(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-019: idéntico a fs-017 pero entrenado sobre train + pseudo-etiquetas de test.

    El pipeline es el mismo; la diferencia está en los datos de entrada
    (train_pseudo.csv en lugar de train.csv), pasados via --train-path en
    02_features.py.
    """
    return _pipeline_fs017(df, impute_age=impute_age)


# ---------------------------------------------------------------------------
# Pipelines deprecados (no usar en nuevos experimentos)
# ---------------------------------------------------------------------------

def _pipeline_fs002(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-002: fs-001 + features de interacción cryo/spending/cabin. [DEPRECADO]"""
    df_out = _pipeline_fs001(df, impute_age=impute_age)
    return create_group_spending_features(df_out)


def _pipeline_fs006(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-006: imputación group-aware de spending antes de calcular totales. [DEPRECADO]"""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_age_features(df_out)
    for col in _CATEGORICAL_FILL:
        df_out[col] = df_out[col].fillna("Unknown")
    df_out = impute_spending_group_aware(df_out)
    df_out = create_spending_features(df_out)
    if impute_age:
        df_out["Age"] = df_out["Age"].fillna(df_out["Age"].median())
    else:
        df_out = df_out.dropna(subset=["Age"])
    return _add_group_size(df_out)


def _pipeline_fs007(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-007 / fs-008: domain rules + spending. [DEPRECADO]"""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    df_out = create_spending_features(df_out)
    return _add_group_size(df_out)


def _pipeline_fs009(df: pd.DataFrame, *, impute_age: bool = False) -> pd.DataFrame:
    """fs-009: domain rules + CabinNumber_DeckPercentile. [DEPRECADO]"""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = apply_domain_rules(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out, impute_age=impute_age)
    df_out = create_spending_features(df_out)
    df_out = _add_cabin_percentile(df_out)
    return _add_group_size(df_out)
