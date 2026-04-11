"""
Registro de feature sets — Spaceship Titanic.

Cada FeatureSetConfig define una configuración inmutable.
Una vez creado, un feature set NO se modifica; para nuevas ideas se agrega
una entrada nueva.

Uso:
    from src.features.feature_sets import FEATURE_SETS
    fs = FEATURE_SETS["fs-004_target_encoding"]
    df_train = fs.pipeline(df_raw)
    df_test  = fs.test_pipeline(df_raw_test)
"""
from typing import List

from src.features.feature_sets.config import FeatureSetConfig
from src.features.feature_sets.pipelines import (
    _pipeline_fs001,
    _pipeline_fs002,
    _pipeline_fs003,
    _pipeline_fs004,
    _pipeline_fs005,
    _pipeline_fs006,
    _pipeline_fs007,
    _pipeline_fs009,
    _pipeline_fs010,
    _pipeline_fs011,
    _pipeline_fs013,
)

# ---------------------------------------------------------------------------
# Listas de features base (fs-001)
# ---------------------------------------------------------------------------

_FS001_NUMERIC: List[str] = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
]
_FS001_CATEGORICAL: List[str] = ["HomePlanet", "Destination", "Deck", "AgeCategory"]
_FS001_DROP: List[str] = [
    "PassengerId", "Name", "Cabin", "TravelGroup",
    "CryoSleep", "VIP", "Side", "TotalSpending",
]

# ---------------------------------------------------------------------------
# Listas de features derivadas
# ---------------------------------------------------------------------------

_FS004_TARGET_ENCODE: List[str] = ["Deck", "HomePlanet"]
_FS004_NUMERIC: List[str] = _FS001_NUMERIC + ["Deck_TE", "HomePlanet_TE"]
_FS004_CATEGORICAL: List[str] = ["Destination", "AgeCategory"]
_FS004_DROP: List[str] = _FS001_DROP

_FS012_TARGET_ENCODE: List[str] = _FS004_TARGET_ENCODE + ["Route"]
_FS012_NUMERIC: List[str] = _FS004_NUMERIC + ["IsChild", "GroupHasChild", "GroupChildRate", "Route_TE"]

# ---------------------------------------------------------------------------
# Registro principal
# ---------------------------------------------------------------------------

FEATURE_SETS: dict = {

    # --- Activos ---

    "fs-001_baseline": FeatureSetConfig(
        description=(
            "Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, "
            "spending log+categorias, AgeCategory."
        ),
        parent=None,
        pipeline=lambda df: _pipeline_fs001(df),
        test_pipeline=lambda df: _pipeline_fs001(df, impute_age=True),
        numeric_features=_FS001_NUMERIC,
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
    ),

    "fs-003_solo_interactions": FeatureSetConfig(
        description=(
            "fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), "
            "SpendingIntensity (TotalSpending/(SpendingCategories+1))."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs003(df),
        test_pipeline=lambda df: _pipeline_fs003(df, impute_age=True),
        numeric_features=_FS001_NUMERIC + ["IsAlone", "IsChild", "SpendingIntensity"],
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
    ),

    "fs-004_target_encoding": FeatureSetConfig(
        description=(
            "fs-001 con Deck y HomePlanet reemplazados por Target Encoding. "
            "Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad "
            "y aportan información ordinal que OHE no captura."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs004(df),
        test_pipeline=lambda df: _pipeline_fs004(df, impute_age=True),
        numeric_features=_FS004_NUMERIC,
        categorical_cols=_FS004_CATEGORICAL,
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS004_TARGET_ENCODE,
    ),

    "fs-005_structural_context": FeatureSetConfig(
        description=(
            "fs-001 + 7 features estructurales: SpendingEntropy, GroupSpendingZScore, "
            "CabinNeighborhoodDensity, FamilySizeFromName, GroupCryoAlignment, "
            "SpendingCategoryProfile (TE), GroupAgeDispersion."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs005(df),
        test_pipeline=lambda df: _pipeline_fs005(df, impute_age=True),
        numeric_features=_FS001_NUMERIC + [
            "SpendingEntropy", "GroupSpendingZScore", "CabinNeighborhoodDensity",
            "FamilySizeFromName", "GroupCryoAlignment", "GroupAgeDispersion",
            "SpendingCategoryProfile_TE",
        ],
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
        target_encode_cols=["SpendingCategoryProfile"],
    ),

    "fs-010_cryo_spending": FeatureSetConfig(
        description=(
            "fs-004 + 4 interacciones CryoSleep × spending: "
            "CryoSpendingAnomaly, GroupTransportedProxy, SideSpendingDiff, CryoSleepBinary."
        ),
        parent="fs-004_target_encoding",
        pipeline=lambda df: _pipeline_fs010(df),
        test_pipeline=lambda df: _pipeline_fs010(df, impute_age=True),
        numeric_features=_FS004_NUMERIC + [
            "CryoSpendingAnomaly", "GroupTransportedProxy",
            "SideSpendingDiff", "CryoSleepBinary",
        ],
        categorical_cols=_FS004_CATEGORICAL,
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS004_TARGET_ENCODE,
    ),

    "fs-011_child_route": FeatureSetConfig(
        description=(
            "fs-004 + contexto familiar (IsChild, GroupHasChild, GroupChildRate) "
            "+ Route (OHE, 9 combinaciones)."
        ),
        parent="fs-004_target_encoding",
        pipeline=lambda df: _pipeline_fs011(df),
        test_pipeline=lambda df: _pipeline_fs011(df, impute_age=True),
        numeric_features=_FS004_NUMERIC + ["IsChild", "GroupHasChild", "GroupChildRate"],
        categorical_cols=_FS004_CATEGORICAL + ["Route"],
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS004_TARGET_ENCODE,
    ),

    "fs-012_child_route_te": FeatureSetConfig(
        description=(
            "fs-011 con Route como Target Encoding. "
            "Route_TE captura la tasa media de transporte por ruta (señal ordinal "
            "que OHE no puede expresar). Mejor Kaggle: 0.80617."
        ),
        parent="fs-011_child_route",
        pipeline=lambda df: _pipeline_fs011(df),
        test_pipeline=lambda df: _pipeline_fs011(df, impute_age=True),
        numeric_features=_FS012_NUMERIC,
        categorical_cols=_FS004_CATEGORICAL,
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS012_TARGET_ENCODE,
    ),

    "fs-013_group_context": FeatureSetConfig(
        description=(
            "fs-004 + Age imputada por grupo + 4 features de comportamiento colectivo: "
            "GroupAllCryo (80.5% vs 42.4% transported), GroupAnyCryo, "
            "SpendShare (corr=-0.15), GroupSpendOthers_Log (corr=+0.09)."
        ),
        parent="fs-004_target_encoding",
        pipeline=lambda df: _pipeline_fs013(df),
        test_pipeline=lambda df: _pipeline_fs013(df, impute_age=True),
        numeric_features=_FS004_NUMERIC + [
            "GroupAllCryo", "GroupAnyCryo", "SpendShare", "GroupSpendOthers_Log",
        ],
        categorical_cols=_FS004_CATEGORICAL,
        features_to_drop=_FS004_DROP,
        target_encode_cols=_FS004_TARGET_ENCODE,
    ),

    # --- Deprecados (referencia histórica, no usar en nuevos experimentos) ---

    "fs-002_cryo_interactions": FeatureSetConfig(
        description=(
            "[DEPRECADO] fs-001 + Route, GroupCryoSleepRate, CryoSleepViolation, "
            "LuxurySpendingRatio, CabinNumber_DeckPercentile, GroupSpendingMean. "
            "No superó al baseline (exp-002)."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs002(df),
        test_pipeline=lambda df: _pipeline_fs002(df, impute_age=True),
        numeric_features=_FS001_NUMERIC + [
            "GroupCryoSleepRate", "CryoSleepViolation",
            "LuxurySpendingRatio", "CabinNumber_DeckPercentile", "GroupSpendingMean",
        ],
        categorical_cols=_FS001_CATEGORICAL + ["Route"],
        features_to_drop=_FS001_DROP,
        deprecated=True,
    ),

    "fs-006_group_imputation": FeatureSetConfig(
        description=(
            "[DEPRECADO] fs-001 con imputación group-aware de spending. "
            "No superó al baseline (exp-006)."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs006(df),
        test_pipeline=lambda df: _pipeline_fs006(df, impute_age=True),
        numeric_features=_FS001_NUMERIC,
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
        deprecated=True,
    ),

    "fs-007_domain_rules": FeatureSetConfig(
        description=(
            "[DEPRECADO] 6 reglas físicas + TravelGroup_TE. "
            "TravelGroup_TE introdujo data leakage (val 0.945, Kaggle n/a)."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs007(df),
        test_pipeline=lambda df: _pipeline_fs007(df, impute_age=True),
        numeric_features=_FS001_NUMERIC + ["TravelGroup_TE"],
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=[c for c in _FS001_DROP if c != "TravelGroup"],
        target_encode_cols=["TravelGroup"],
        deprecated=True,
    ),

    "fs-008_domain_rules_only": FeatureSetConfig(
        description=(
            "[DEPRECADO] 6 reglas físicas sin TravelGroup_TE. "
            "No superó al baseline (exp-008)."
        ),
        parent="fs-001_baseline",
        pipeline=lambda df: _pipeline_fs007(df),
        test_pipeline=lambda df: _pipeline_fs007(df, impute_age=True),
        numeric_features=_FS001_NUMERIC,
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP,
        deprecated=True,
    ),

    "fs-009_percentile_cabin": FeatureSetConfig(
        description=(
            "[DEPRECADO] fs-008 + CabinNumber_DeckPercentile. "
            "No superó al baseline (exp-009)."
        ),
        parent="fs-008_domain_rules_only",
        pipeline=lambda df: _pipeline_fs009(df),
        test_pipeline=lambda df: _pipeline_fs009(df, impute_age=True),
        numeric_features=[
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "GroupSize", "CabinNumber_DeckPercentile", "TotalSpending_Log",
            "SpendingCategories",
        ],
        categorical_cols=_FS001_CATEGORICAL,
        features_to_drop=_FS001_DROP + ["CabinNumber"],
        deprecated=True,
    ),
}

DEFAULT_FEATURE_SET: str = "fs-001_baseline"
