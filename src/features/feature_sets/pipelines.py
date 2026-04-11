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
    create_group_context_features,
    create_group_spending_features,
    create_solo_interaction_features,
    create_structural_context_features,
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
