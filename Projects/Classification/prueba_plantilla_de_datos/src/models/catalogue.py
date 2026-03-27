"""
Catalogo de modelos e hiperparametros para Spaceship Titanic.

Centraliza las definiciones de estimadores y espacios de busqueda
para que puedan importarse desde cualquier script sin duplicar codigo.

PARAM_SPACES contiene callables (trial) -> dict compatibles con Optuna.
Cada funcion define el espacio de busqueda de un modelo usando la API
de sugerencias de Optuna (suggest_int, suggest_float, suggest_categorical).
"""
from typing import Any, Callable, Dict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.moe import MixtureOfExperts

MODELS: Dict[str, Any] = {
    "Baseline": DummyClassifier(strategy="most_frequent", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200, random_state=42
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, random_state=42, verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200, random_seed=42, verbose=0
    ),
    "MoE_CatBoost": MixtureOfExperts(
        base_estimator=CatBoostClassifier(iterations=200, random_seed=42, verbose=0),
    ),
}


# ---------------------------------------------------------------------------
# Espacios de busqueda Optuna — (trial) -> dict de hiperparametros
# ---------------------------------------------------------------------------

def _logistic_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 0.001, 100.0, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
    }


def _random_forest_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _gradient_boosting_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _histgb_space(trial) -> dict:
    return {
        "max_iter": trial.suggest_categorical("max_iter", [100, 200, 400]),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63, 127]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }


def _extra_trees_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _xgboost_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }


def _lightgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }


def _catboost_space(trial) -> dict:
    return {
        "iterations": trial.suggest_categorical("iterations", [100, 200, 400, 600]),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


PARAM_SPACES: Dict[str, Callable] = {
    "LogisticRegression": _logistic_space,
    "RandomForest": _random_forest_space,
    "GradientBoosting": _gradient_boosting_space,
    "HistGradientBoosting": _histgb_space,
    "ExtraTrees": _extra_trees_space,
    "XGBoost": _xgboost_space,
    "LightGBM": _lightgbm_space,
    "CatBoost": _catboost_space,
}
