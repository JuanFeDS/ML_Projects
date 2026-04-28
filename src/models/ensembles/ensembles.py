"""Construccion y evaluacion de modelos de ensemble: Stacking y Mixture of Experts."""

from typing import Any, List, Tuple

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.ensembles.moe import MixtureOfExperts


def build_stacking(
    base_estimators: List[Tuple[str, Any]],
    x_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
) -> Tuple[Any, float]:
    """Construye y evalua un StackingClassifier con meta-modelo HistGradientBoosting.

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


def build_moe(
    tuned_base: Any,
    x_train: Any,
    y_train: Any,
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
        moe, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
    )
    return moe, round(cv_scores.mean(), 4)
