"""Ajuste de hiperparametros con Optuna (TPE sampler)."""
from typing import Any, Callable, Dict, Optional, Tuple

import optuna
import optuna.logging
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

from src.models.tracking import log_metrics_dict, log_params_dict, mlrun, setup_mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model: Any,
    param_space_fn: Callable,
    x_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
    n_iter: int = 25,
    param_transform: Optional[Callable[[Dict], Dict]] = None,
) -> Tuple[Any, Dict, float]:
    """Ajusta hiperparametros con Optuna (TPE sampler).

    Usa Bayesian optimization (Tree-structured Parzen Estimator) para explorar
    el espacio de hiperparametros de forma mas eficiente que RandomizedSearch.
    Cada trial evalua un set de params con CV y reporta la accuracy media.

    Args:
        model: Estimador base sin ajustar.
        param_space_fn: Callable (trial) -> dict que define el espacio de busqueda.
            Ver src/models/catalogue.py para las definiciones por modelo.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation.
        n_iter: Numero de trials de Optuna.
        param_transform: Callable opcional (dict) -> dict que transforma study.best_params
            antes de llamar set_params en el modelo reconstruido. Util cuando model es un
            Pipeline y los params necesitan prefijo (ej. 'model__depth' en vez de 'depth').

    Returns:
        Tupla (best_estimator, best_params, best_score).
    """
    setup_mlflow()
    with mlrun(run_name=f"Hype-Opt: {type(model).__name__}", tags={"type": "optuna_tuning"}):
        def objective(trial) -> float:
            params = param_space_fn(trial)
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

        with tqdm(total=n_iter, desc=f"Optuna ({type(model).__name__})", unit="trial") as pbar:
            def _callback(study, trial):  # pylint: disable=unused-argument
                pbar.update(1)
            study.optimize(objective, n_trials=n_iter, callbacks=[_callback])

        best_params = study.best_params
        best_model = clone(model)
        params_to_set = param_transform(best_params) if param_transform else best_params
        best_model.set_params(**params_to_set)

        log_params_dict({f"best_{k}": v for k, v in best_params.items()})
        log_metrics_dict({"best_cv_accuracy": round(study.best_value, 4)})

    return best_model, best_params, round(study.best_value, 4)
