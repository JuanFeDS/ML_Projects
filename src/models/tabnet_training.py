"""Tuning y entrenamiento de TabNet con Optuna (sin dependencia de sklearn CV)."""
from __future__ import annotations

from typing import Dict, Tuple

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score

from src.models.tabnet_wrapper import TabNetWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)


def param_space(trial: optuna.Trial) -> Dict:
    """Define el espacio de busqueda de hiperparametros para TabNet.

    Args:
        trial: Trial de Optuna.

    Returns:
        Diccionario de hiperparametros sugeridos.
    """
    n_d = trial.suggest_categorical("n_d", [16, 32, 64])
    return {
        "n_d": n_d,
        "n_a": n_d,
        "n_steps": trial.suggest_int("n_steps", 2, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "n_independent": trial.suggest_int("n_independent", 1, 3),
        "n_shared": trial.suggest_int("n_shared", 1, 3),
        "momentum": trial.suggest_float("momentum", 0.01, 0.4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    }


def tune_tabnet(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = 25,
    max_epochs: int = 200,
    patience: int = 20,
) -> Tuple[Dict, float]:
    """Optimiza hiperparametros de TabNet con Optuna usando split train/val directo.

    TabNet no es compatible con cross_val_score de sklearn (lento + necesita eval_set
    para early stopping), por lo que usa un split fijo en lugar de CV.

    Args:
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        x_val: Features de validacion (para early stopping y evaluacion).
        y_val: Target de validacion.
        n_iter: Numero de trials de Optuna.
        max_epochs: Maximo de epocas por trial.
        patience: Epocas de early stopping.

    Returns:
        Tupla (best_params, best_val_accuracy).
    """
    def objective(trial: optuna.Trial) -> float:
        params = param_space(trial)
        model = TabNetWrapper(
            max_epochs=max_epochs,
            patience=patience,
            virtual_batch_size=min(params["batch_size"] // 4, 256),
            **params,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        return float(accuracy_score(y_val, model.predict(x_val)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    return study.best_params, round(study.best_value, 4)


def train_tabnet(
    best_params: Dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    max_epochs: int = 200,
    patience: int = 20,
) -> TabNetWrapper:
    """Entrena un TabNet final con los mejores hiperparametros.

    Args:
        best_params: Hiperparametros obtenidos de tune_tabnet.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        x_val: Features de validacion para early stopping (opcional).
        y_val: Target de validacion para early stopping (opcional).
        max_epochs: Maximo de epocas de entrenamiento.
        patience: Epocas de early stopping.

    Returns:
        Modelo TabNetWrapper entrenado.
    """
    eval_set = [(x_val, y_val)] if x_val is not None else None
    model = TabNetWrapper(
        max_epochs=max_epochs,
        patience=patience,
        virtual_batch_size=min(best_params["batch_size"] // 4, 256),
        **best_params,
    )
    model.fit(x_train, y_train, eval_set=eval_set)
    return model
