"""Constantes y preprocesamiento compartido entre scripts de CatBoost nativo y ensembles."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.features.constants import TARGET
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets import FEATURE_SETS
from src.features.feature_sets.pipelines import _pipeline_fs017

FS_NATIVE = "fs-017_lastname_te"

CAT_FEATURES_NATIVE = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]

NUMERIC_FEATURES_NATIVE = [
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
    "HasSpending",
    "CryoSleep_Encoded",
    "Side_Encoded",
]

ALL_FEATURES_NATIVE = NUMERIC_FEATURES_NATIVE + CAT_FEATURES_NATIVE


def _apply_native_preprocessing(out: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificacion CryoSleep/Side y convierte categoricas a string."""
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES_NATIVE:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def _select_native_features(out: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in ALL_FEATURES_NATIVE if c in out.columns]
    return out[feat_cols]


def preprocess_native_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocessing nativo para entrenamiento: retorna (X, y).

    Args:
        df: DataFrame crudo (train.csv).

    Returns:
        Tupla (X, y) con features y target.
    """
    out = _apply_native_preprocessing(_pipeline_fs017(df))
    feat = _select_native_features(out)
    return feat, out[TARGET].astype(int)


def preprocess_native_test(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing nativo para inferencia: retorna solo X.

    Args:
        df: DataFrame crudo (test.csv).

    Returns:
        DataFrame de features listo para predict.
    """
    out = _apply_native_preprocessing(FEATURE_SETS[FS_NATIVE].test_pipeline(df))
    return _select_native_features(out)


def preprocess_native_train_groups(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Como preprocess_native_train pero retorna grupos de LastName para GroupKFold.

    Args:
        df: DataFrame crudo (train.csv).

    Returns:
        Tupla (X, y, groups) donde groups es el array de LastName.
    """
    out = _apply_native_preprocessing(_pipeline_fs017(df))
    feat = _select_native_features(out)
    return feat, out[TARGET].astype(int), out["LastName"].values


def preprocess_native_test_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Como preprocess_native_test pero retorna grupos de LastName.

    Args:
        df: DataFrame crudo (test.csv).

    Returns:
        Tupla (X, groups) donde groups es el array de LastName.
    """
    out = _apply_native_preprocessing(FEATURE_SETS[FS_NATIVE].test_pipeline(df))
    feat = _select_native_features(out)
    return feat, out["LastName"].values
