"""Ajuste de hiperparametros con Optuna (TPE sampler)."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import optuna
import optuna.logging
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

from src.models.tracking import log_metrics_dict, log_params_dict, mlrun, setup_mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuneConfig:
    """Configuracion para tune_model."""

    param_space_fn: Callable
    n_iter: int = 25
    param_transform: Optional[Callable[[Dict], Dict]] = field(default=None)


def tune_model(
    model: Any,
    x_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
    config: TuneConfig,
) -> Tuple[Any, Dict, float]:
    """Ajusta hiperparametros con Optuna (TPE sampler).

    Usa Bayesian optimization (Tree-structured Parzen Estimator) para explorar
    el espacio de hiperparametros de forma mas eficiente que RandomizedSearch.
    Cada trial evalua un set de params con CV y reporta la accuracy media.

    Args:
        model: Estimador base sin ajustar.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        config: TuneConfig con param_space_fn, n_iter y param_transform.

    Returns:
        Tupla (best_estimator, best_params, best_score).
    """
    setup_mlflow()
    with mlrun(
        run_name=f"Hype-Opt: {type(model).__name__}", tags={"type": "optuna_tuning"}
    ):

        def objective(trial) -> float:
            params = config.param_space_fn(trial)
            est = clone(model)
            est.set_params(**params)
            scores = cross_val_score(
                est, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            with mlrun(run_name=f"Trial {trial.number}", nested=True):
                log_metrics_dict({"trial_accuracy": float(scores.mean())})
                log_params_dict(params)
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        with tqdm(
            total=config.n_iter, desc=f"Optuna ({type(model).__name__})", unit="trial"
        ) as pbar:

            def _callback(study, trial):  # pylint: disable=unused-argument
                pbar.update(1)

            study.optimize(objective, n_trials=config.n_iter, callbacks=[_callback])

        best_params = study.best_params
        best_model = clone(model)
        params_to_set = (
            config.param_transform(best_params)
            if config.param_transform
            else best_params
        )
        best_model.set_params(**params_to_set)

        log_params_dict({f"best_{k}": v for k, v in best_params.items()})
        log_metrics_dict({"best_cv_accuracy": round(study.best_value, 4)})

    return best_model, best_params, round(study.best_value, 4)
