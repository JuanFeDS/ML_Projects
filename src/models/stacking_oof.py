"""Generacion de predicciones OOF para los modelos base del stacking (exp-027, 033, 047)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from src.models.catboost_native import oof_proba as catboost_oof_proba
from src.scripts.common import CAT_FEATURES_NATIVE

EXP_033_PARAMS = {
    "iterations": 546,
    "depth": 7,
    "learning_rate": 0.07071005741224391,
    "l2_leaf_reg": 21.119964967140604,
    "bagging_temperature": 0.8011251001518414,
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}


def _build_027_pipeline() -> Pipeline:
    """Reconstruye el Pipeline de exp-027 desde cero.

    Necesario para evitar incompatibilidad de sklearn entre la version con
    la que fue entrenado (1.6) y la actual (1.8+).
    """
    ct = ColumnTransformer(
        transformers=[
            ("te", TargetEncoder(random_state=42, target_type="binary"), ["LastName"])
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    cb = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.021652690184947157,
        l2_leaf_reg=16.121101290648085,
        bagging_temperature=0.003097017597283175,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    return Pipeline([("preprocessor", ct), ("model", cb)])


def oof_027(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> np.ndarray:
    """Genera predicciones OOF para exp-027 (Pipeline TargetEncoder + CatBoost).

    Args:
        x_train: Features escaladas del feature set fs-017.
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation.

    Returns:
        Array de probabilidades OOF.
    """
    pipe = _build_027_pipeline()
    print("  [027] cross_val_predict (5-fold)...")
    proba = cross_val_predict(pipe, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    print(f"  [027] OOF acc={accuracy_score(y_train, (proba >= 0.5).astype(int)):.4f}")
    return proba


def oof_033(
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> np.ndarray:
    """Genera predicciones OOF para exp-033 (CatBoost native con Pool).

    Args:
        x: Features nativas (con categoricas como strings).
        y: Target de entrenamiento.
        cv: Estrategia de cross-validation.

    Returns:
        Array de probabilidades OOF.
    """
    proba = catboost_oof_proba(EXP_033_PARAMS, x, y, cv, CAT_FEATURES_NATIVE)
    for fold_idx, (_, val_idx) in enumerate(cv.split(x, y), 1):
        acc = accuracy_score(y.iloc[val_idx], (proba[val_idx] >= 0.5).astype(int))
        print(f"  [033] Fold {fold_idx}/5 acc={acc:.4f}")
    print(f"  [033] OOF acc={accuracy_score(y, (proba >= 0.5).astype(int)):.4f}")
    return proba


def oof_tabpfn(
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    n_estimators: int = 8,
) -> np.ndarray:
    """Genera predicciones OOF para TabPFN (cloud API).

    Requiere que TABPFN_TOKEN este configurado en el entorno antes de llamar.

    Args:
        x: Features de entrenamiento.
        y: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        n_estimators: Numero de estimadores del forward pass TabPFN.

    Returns:
        Array de probabilidades OOF.
    """
    from tabpfn_client import TabPFNClassifier  # pylint: disable=import-outside-toplevel

    proba = np.zeros(len(x))
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(x, y), 1):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = TabPFNClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(x_tr, y_tr)
        proba[val_idx] = model.predict_proba(x_val)[:, 1]
        acc = accuracy_score(y_val, (proba[val_idx] >= 0.5).astype(int))
        print(f"  [047] Fold {fold_idx}/5 acc={acc:.4f}")
    print(f"  [047] OOF acc={accuracy_score(y, (proba >= 0.5).astype(int)):.4f}")
    return proba
