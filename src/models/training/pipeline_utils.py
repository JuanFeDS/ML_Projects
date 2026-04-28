"""Utilidades para construir Pipelines sklearn con preprocesamiento fold-aware."""

from typing import Any, Callable, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder


def make_fold_te_pipeline(model: Any, fold_te_cols: List[str]) -> Pipeline:
    """Envuelve un estimador en un Pipeline con TargetEncoder fold-aware.

    El TargetEncoder (sklearn >= 1.3) computa el encoding por fold via fit_transform
    interno (cv=5), eliminando el leakage de columnas de alta cardinalidad como LastName.
    Al llamar predict/transform sobre datos nuevos aplica el encoding ajustado sin ver
    las etiquetas del test set.

    Args:
        model: Estimador sklearn base.
        fold_te_cols: Columnas a codificar con TargetEncoder fold-aware.

    Returns:
        Pipeline([ColumnTransformer(TargetEncoder), model]).
    """
    te = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=42)
    ct = ColumnTransformer(
        [("te", te, fold_te_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", ct), ("model", model)])


def prefix_param_space(param_space_fn: Callable) -> Callable:
    """Prefija 'model__' a los parametros de Optuna para uso dentro de un Pipeline.

    Args:
        param_space_fn: Callable (trial) -> dict del espacio original.

    Returns:
        Callable que devuelve el mismo dict con claves prefijadas con 'model__'.
    """

    def wrapped(trial: Any) -> Dict:
        return {f"model__{k}": v for k, v in param_space_fn(trial).items()}

    return wrapped


def add_model_prefix(params: Dict) -> Dict:
    """Agrega el prefijo 'model__' a un diccionario de parametros.

    Util para transformar best_params de Optuna al formato que espera
    Pipeline.set_params() cuando el estimador esta bajo la clave 'model'.

    Args:
        params: Diccionario de parametros sin prefijo.

    Returns:
        Diccionario con claves prefijadas con 'model__'.
    """
    return {f"model__{k}": v for k, v in params.items()}
