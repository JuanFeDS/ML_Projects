"""Constantes y preprocesamiento compartido entre scripts de CatBoost nativo y ensembles."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from src.features.constants import TARGET
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets import FEATURE_SETS
from src.features.feature_sets.pipelines import _pipeline_fs017

FS_NATIVE = "fs-017_lastname_te"

CAT_FEATURES_NATIVE = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]

NUMERIC_FEATURES_NATIVE = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]

ALL_FEATURES_NATIVE = NUMERIC_FEATURES_NATIVE + CAT_FEATURES_NATIVE


def preprocess_native(
    df: pd.DataFrame, is_test: bool = False
) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """Pipeline compartido para experimentos con categoricas nativas de CatBoost.

    Aplica fs-017 (con TargetEncoder en LastName), codifica CryoSleep y Side,
    convierte categoricas a string y retorna (X, y) en train o solo X en test.

    Args:
        df: DataFrame crudo (train.csv o test.csv).
        is_test: Si True, usa test_pipeline (imputa Age NaN con mediana) y no
            retorna y.

    Returns:
        Tupla (X, y) si is_test=False; DataFrame X si is_test=True.
    """
    fs = FEATURE_SETS[FS_NATIVE]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)

    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)

    for col in CAT_FEATURES_NATIVE:
        out[col] = out[col].fillna("Unknown").astype(str)

    feat_cols = [c for c in ALL_FEATURES_NATIVE if c in out.columns]

    if not is_test:
        y = out[TARGET].astype(int)
        return out[feat_cols], y

    return out[feat_cols]


def preprocess_native_with_groups(
    df: pd.DataFrame, is_test: bool = False
) -> Union[
    Tuple[pd.DataFrame, pd.Series, np.ndarray],
    Tuple[pd.DataFrame, np.ndarray],
]:
    """Como preprocess_native pero tambien retorna grupos de LastName.

    Necesario para GroupKFold (validacion honesta sin leakage de familia).

    Args:
        df: DataFrame crudo (train.csv o test.csv).
        is_test: Si True, no retorna y.

    Returns:
        (X, y, groups) si is_test=False; (X, groups) si is_test=True.
    """
    fs = FEATURE_SETS[FS_NATIVE]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)

    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)

    for col in CAT_FEATURES_NATIVE:
        out[col] = out[col].fillna("Unknown").astype(str)

    groups = out["LastName"].values
    feat_cols = [c for c in ALL_FEATURES_NATIVE if c in out.columns]

    if not is_test:
        y = out[TARGET].astype(int)
        return out[feat_cols], y, groups

    return out[feat_cols], groups
