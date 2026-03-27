"""
Utilidades de entrenamiento para modelos de clasificacion.

Funciones reutilizables para evaluacion con CV, tuning (Optuna) y stacking.
"""
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from tqdm import tqdm

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from src.models.moe import MixtureOfExperts


def evaluate_models(
    models: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Evalua multiples modelos con validacion cruzada.

    Args:
        models: Diccionario {nombre: estimador sklearn}.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation.

    Returns:
        DataFrame con cv_accuracy_mean, cv_accuracy_std, cv_roc_auc_mean
        ordenado de mayor a menor accuracy.
    """
    results = {}
    for name, model in tqdm(models.items(), desc="CV modelos", unit="modelo"):
        cv_acc = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        cv_auc = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {
            "cv_accuracy_mean": round(cv_acc.mean(), 4),
            "cv_accuracy_std": round(cv_acc.std(), 4),
            "cv_roc_auc_mean": round(cv_auc.mean(), 4),
        }
    return pd.DataFrame(results).T.sort_values("cv_accuracy_mean", ascending=False)


def tune_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model: Any,
    param_space_fn: Callable,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_iter: int = 25,
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

    Returns:
        Tupla (best_estimator, best_params, best_score).

    Raises:
        ImportError: Si optuna no esta instalado.
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna es requerido para el tuning. "
            "Instalar con: pip install optuna"
        )

    def objective(trial) -> float:
        params = param_space_fn(trial)
        est = clone(model)
        est.set_params(**params)
        scores = cross_val_score(
            est, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
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
    best_model.set_params(**best_params)
    return best_model, best_params, round(study.best_value, 4)


def evaluate_on_validation(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict:
    """Entrena el modelo y lo evalua en el set de validacion.

    Args:
        model: Estimador sklearn.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        x_val: Features de validacion.
        y_val: Target de validacion.

    Returns:
        Diccionario con val_accuracy, val_roc_auc y classification_report.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]
    return {
        "val_accuracy": round(accuracy_score(y_val, y_pred), 4),
        "val_roc_auc": round(roc_auc_score(y_val, y_proba), 4),
        "classification_report": classification_report(y_val, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def analyze_errors(
    x_val: pd.DataFrame,
    y_val: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Analiza la tasa de error por segmento en el conjunto de validacion.

    Reconstruye las variables originales desde columnas OHE/encoded y
    calcula cuantos errores comete el modelo en cada categoria.

    Args:
        x_val: Features del conjunto de validacion (escaldas, con OHE).
        y_val: Target real.
        y_pred: Predicciones del modelo (array o Series).

    Returns:
        Diccionario {nombre_segmento: DataFrame con [segmento, n, errors, error_rate]}.
    """
    df_err = x_val.copy()
    df_err["_y_true"] = y_val.values
    df_err["_y_pred"] = pd.array(y_pred)
    df_err["_error"] = (df_err["_y_true"] != df_err["_y_pred"]).astype(int)

    results = {}

    if "CryoSleep_Encoded" in df_err.columns:
        cryo_map = {1: "Cryo", 0: "Active", -1: "Unknown"}
        df_err["_CryoSleep"] = df_err["CryoSleep_Encoded"].map(cryo_map)
        g = df_err.groupby("_CryoSleep")["_error"].agg(["count", "sum"]).reset_index()
        g.columns = ["CryoSleep", "n", "errors"]
        g["error_rate"] = (g["errors"] / g["n"]).round(4)
        results["CryoSleep"] = g.sort_values("error_rate", ascending=False)

    for prefix in ["HomePlanet", "Destination", "AgeCategory", "Deck"]:
        cols = [c for c in df_err.columns if c.startswith(f"{prefix}_")]
        if not cols:
            continue
        segment = (
            df_err[cols]
            .idxmax(axis=1)
            .str.replace(f"{prefix}_", "", regex=False)
        )
        df_err[f"_{prefix}"] = segment
        g = (
            df_err.groupby(f"_{prefix}")["_error"]
            .agg(["count", "sum"])
            .reset_index()
        )
        g.columns = [prefix, "n", "errors"]
        g["error_rate"] = (g["errors"] / g["n"]).round(4)
        results[prefix] = g.sort_values("error_rate", ascending=False)

    return results


def optimize_threshold(
    y_val: pd.Series,
    y_proba: np.ndarray,
    grid_size: int = 200,
) -> Tuple[float, float]:
    """Busca el umbral de clasificacion que maximiza accuracy en validacion.

    Evalua umbrales equiespaciados entre 0.3 y 0.7. Util cuando la distribucion
    de probabilidades del modelo no esta perfectamente centrada en 0.5.

    Args:
        y_val: Target real.
        y_proba: Probabilidades de la clase positiva (output de predict_proba[:, 1]).
        grid_size: Numero de umbrales a evaluar en [0.3, 0.7].

    Returns:
        Tupla (best_threshold, best_accuracy).
    """
    thresholds = np.linspace(0.3, 0.7, grid_size)
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        acc = float(accuracy_score(y_val, y_pred_t))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return round(best_t, 4), round(best_acc, 4)


def build_moe(
    tuned_base: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Any, float]:
    """Construye y evalua un MixtureOfExperts usando el modelo tuneado como base.

    Cada experto es un clon independiente del tuned_base, entrenado sobre
    su segmento (cryo o activo). Con drop_zero_variance=True (default), el
    experto cryo recibe solo columnas con varianza positiva en ese segmento,
    eliminando automaticamente las features de gasto (siempre 0 en cryo).

    Args:
        tuned_base: Estimador ya tuneado (e.g. CatBoost con best_params).
        x_train: Features de entrenamiento (DataFrame con CryoSleep_Encoded).
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation.

    Returns:
        Tupla (moe_model, cv_accuracy_mean).
    """
    moe = MixtureOfExperts(base_estimator=clone(tuned_base), drop_zero_variance=True)
    cv_scores = cross_val_score(
        moe, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return moe, round(cv_scores.mean(), 4)


def build_stacking(
    base_estimators: List[Tuple[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Any, float]:
    """Construye y evalua un StackingClassifier con meta-modelo LogisticRegression.

    Args:
        base_estimators: Lista de (nombre, estimador) para la capa base.
        x_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv: Estrategia de cross-validation para evaluar el stack.

    Returns:
        Tupla (stacking_model, cv_accuracy_mean).
    """
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=HistGradientBoostingClassifier(max_iter=200, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    cv_scores = cross_val_score(stacking, x_train, y_train, cv=cv, scoring="accuracy")
    return stacking, round(cv_scores.mean(), 4)
