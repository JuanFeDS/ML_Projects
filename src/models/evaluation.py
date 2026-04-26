"""Evaluacion de modelos: CV, OOF, validacion y optimizacion de umbral."""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from tqdm import tqdm

from src.models.tracking import log_metrics_dict, log_params_dict, mlrun, setup_mlflow


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
    setup_mlflow()
    for name, model in tqdm(models.items(), desc="CV modelos", unit="modelo"):
        with mlrun(run_name=f"Comparison: {name}", nested=False, tags={"type": "model_selection"}):
            cv_acc = cross_val_score(
                model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            cv_auc = cross_val_score(
                model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            metrics = {
                "cv_accuracy_mean": round(cv_acc.mean(), 4),
                "cv_accuracy_std": round(cv_acc.std(), 4),
                "cv_roc_auc_mean": round(cv_auc.mean(), 4),
            }
            results[name] = metrics
            log_params_dict(model.get_params())
            log_metrics_dict(metrics)

    return pd.DataFrame(results).T.sort_values("cv_accuracy_mean", ascending=False)


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


def compute_oof_metrics(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Dict:
    """Genera predicciones out-of-fold y calcula metricas sobre el dataset completo.

    A diferencia de evaluate_on_validation, no depende de un split fijo.
    Las predicciones OOF dan una estimacion honesta del rendimiento real del modelo.

    Args:
        model: Estimador sklearn.
        x: Features completas (sin split previo).
        y: Target completo.
        cv: Estrategia de cross-validation.

    Returns:
        Diccionario con val_accuracy, val_roc_auc, classification_report,
        y_pred y y_proba (todos OOF).
    """
    oof_proba = cross_val_predict(
        model, x, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    return {
        "val_accuracy": round(float(accuracy_score(y, oof_pred)), 4),
        "val_roc_auc": round(float(roc_auc_score(y, oof_proba)), 4),
        "classification_report": classification_report(y, oof_pred),
        "y_pred": oof_pred,
        "y_proba": oof_proba,
    }


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
