"""Entrenamiento y evaluacion de CatBoost con categoricas nativas."""
from __future__ import annotations

from typing import List

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.scripts.common import CAT_FEATURES_NATIVE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pool(
    x: pd.DataFrame,
    y: pd.Series | None = None,
    cat_features: List[str] = CAT_FEATURES_NATIVE,
) -> Pool:
    """Crea un CatBoost Pool con las categoricas definidas.

    Args:
        x: Features de entrada.
        y: Target (opcional, None para test).
        cat_features: Lista de columnas categoricas.

    Returns:
        Pool de CatBoost listo para fit/predict.
    """
    return Pool(x, label=y, cat_features=cat_features)


def cv_score(
    params: dict,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    cat_features: List[str] = CAT_FEATURES_NATIVE,
) -> float:
    """Evalua CatBoost con CV manual usando Pool.

    Usa CV manual en lugar de cross_val_score de sklearn para preservar
    las cat_features del Pool, que sklearn no puede serializar.

    Args:
        params: Hiperparametros de CatBoostClassifier.
        x: Features de entrenamiento.
        y: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        cat_features: Columnas categoricas para el Pool.

    Returns:
        Accuracy media de CV.
    """
    scores = []
    for train_idx, val_idx in cv.split(x, y):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(
            make_pool(x_tr, y_tr, cat_features),
            eval_set=make_pool(x_val, cat_features=cat_features),
            early_stopping_rounds=50,
            verbose=False,
        )
        proba = model.predict_proba(make_pool(x_val, cat_features=cat_features))[:, 1]
        scores.append(accuracy_score(y_val, (proba >= 0.5).astype(int)))
    return float(np.mean(scores))


def tune(
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    n_iter: int,
    study_db: str,
    cat_features: List[str] = CAT_FEATURES_NATIVE,
) -> dict:
    """Optimiza hiperparametros de CatBoost con Optuna TPE.

    Persiste el estudio en SQLite para reanudar si el script falla.

    Args:
        x: Features de entrenamiento.
        y: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        n_iter: Total de trials deseados (se resta los ya completados).
        study_db: Ruta al archivo SQLite del estudio.
        cat_features: Columnas categoricas para el Pool.

    Returns:
        Diccionario con los mejores hiperparametros encontrados.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
        }
        return cv_score(params, x, y, cv, cat_features)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{study_db}",
        study_name="catboost_native",
        load_if_exists=True,
    )
    completed = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = max(0, n_iter - completed)
    if remaining == 0:
        print(f"  Estudio ya tiene {completed} trials. Saltando tuning.")
    else:
        print(f"  Continuando desde {completed} trials. Ejecutando {remaining} mas...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    print(f"  Mejor CV: {study.best_value:.4f}")
    best = {**study.best_params, "random_seed": 42}
    return best


def oof_proba(
    params: dict,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    cat_features: List[str] = CAT_FEATURES_NATIVE,
) -> np.ndarray:
    """Genera probabilidades out-of-fold con CV manual usando Pool.

    Args:
        params: Hiperparametros de CatBoostClassifier.
        x: Features de entrenamiento.
        y: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        cat_features: Columnas categoricas para el Pool.

    Returns:
        Array de probabilidades OOF (misma longitud que x).
    """
    proba = np.zeros(len(x))
    for train_idx, val_idx in cv.split(x, y):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(make_pool(x_tr, y_tr, cat_features))
        proba[val_idx] = model.predict_proba(
            make_pool(x_val, cat_features=cat_features)
        )[:, 1]
    return proba


def oof_proba_groupkfold(
    params: dict,
    x: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    n_splits: int = 5,
    cat_features: List[str] = CAT_FEATURES_NATIVE,
) -> np.ndarray:
    """Genera probabilidades OOF con GroupKFold para evitar leakage de familia.

    Garantiza que registros del mismo LastName no aparecen simultaneamente
    en train y val, eliminando el target leakage del TargetEncoder.

    Args:
        params: Hiperparametros de CatBoostClassifier.
        x: Features de entrenamiento (con categoricas como strings).
        y: Target de entrenamiento.
        groups: Array de grupos (ej. LastName) de la misma longitud que x.
        n_splits: Numero de folds GroupKFold.
        cat_features: Columnas categoricas para el Pool.

    Returns:
        Array de probabilidades OOF (misma longitud que x).
    """
    gkf = GroupKFold(n_splits=n_splits)
    proba = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), 1):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pool_tr = Pool(x_tr, y_tr, cat_features=cat_features)
        pool_val = Pool(x_val, cat_features=cat_features)
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(pool_tr)
        proba[val_idx] = model.predict_proba(pool_val)[:, 1]

        fold_acc = accuracy_score(y_val, (proba[val_idx] >= 0.5).astype(int))
        groups_tr = set(groups[tr_idx])
        groups_val_set = set(groups[val_idx])
        overlap = len(groups_tr & groups_val_set)
        print(
            f"  Fold {fold}/{n_splits} — acc={fold_acc:.4f} | "
            f"val_groups={len(groups_val_set):,} | overlap={overlap}"
        )

    return proba
