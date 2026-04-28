"""
Generacion de plots SHAP para el reporte de entrenamiento.

Selecciona automaticamente el explainer segun el tipo de modelo:
  - TreeExplainer   : RF, GB, HistGB, ExtraTrees, XGBoost, LightGBM, CatBoost
  - LinearExplainer : LogisticRegression
  - KernelExplainer : resto (lento, limitado a muestra de 150 filas)

Todas las funciones devuelven una cadena base64 (PNG) lista para
incrustar con HTMLReport.add_image().
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # sin GUI, compatible con entornos sin display
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

_TREE_TYPES = (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)

try:
    from xgboost import XGBClassifier

    _TREE_TYPES = _TREE_TYPES + (XGBClassifier,)
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier

    _TREE_TYPES = _TREE_TYPES + (LGBMClassifier,)
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier

    _TREE_TYPES = _TREE_TYPES + (CatBoostClassifier,)
except ImportError:
    pass

_KERNEL_SAMPLE = 150  # filas para KernelExplainer (lento)


class ValPredictions(NamedTuple):
    """Datos de validacion para calculos SHAP."""

    X: pd.DataFrame
    y_true: pd.Series
    y_proba: np.ndarray


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convierte una figura matplotlib a cadena base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _build_explainer(
    model: Any,
    x_train: pd.DataFrame,
) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    """Crea el explainer y calcula los SHAP values sobre x_train.

    Args:
        model: Estimador sklearn entrenado.
        x_train: Features de entrenamiento (DataFrame).

    Returns:
        Tupla (explainer, shap_values, x_aligned) para la clase positiva.
    """
    base = getattr(model, "final_estimator_", None) or model

    if isinstance(base, _TREE_TYPES):
        explainer = shap.TreeExplainer(base)
        sv = explainer.shap_values(x_train)
        shap_values = sv[1] if isinstance(sv, list) else sv

    elif isinstance(base, LogisticRegression):
        masker = shap.maskers.Independent(x_train)
        explainer = shap.LinearExplainer(base, masker)
        sv = explainer.shap_values(x_train)
        shap_values = sv[1] if isinstance(sv, list) else sv

    else:
        logger.warning(
            "Modelo %s no soportado por Tree/LinearExplainer. "
            "Usando KernelExplainer sobre %d muestras (lento).",
            type(base).__name__,
            _KERNEL_SAMPLE,
        )
        sample = shap.sample(x_train, _KERNEL_SAMPLE, random_state=42)
        explainer = shap.KernelExplainer(base.predict_proba, sample)
        sv = explainer.shap_values(sample)
        shap_values = sv[1] if isinstance(sv, list) else sv
        x_train = sample

    return explainer, shap_values, x_train


# ---------------------------------------------------------------------------
# Plots publicos
# ---------------------------------------------------------------------------


def shap_summary_bar(
    model: Any,
    x_train: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 20,
) -> Optional[str]:
    """Barplot de importancia global: media de |SHAP| por feature.

    Args:
        model: Estimador sklearn entrenado.
        x_train: Features de entrenamiento.
        feature_names: Nombres de las features (mismo orden que x_train).
        top_n: Numero maximo de features a mostrar.

    Returns:
        Imagen PNG en base64, o None si falla el calculo.
    """
    try:
        _, shap_values, x_aligned = _build_explainer(model, x_train)
        shap.summary_plot(
            shap_values,
            x_aligned,
            feature_names=feature_names,
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        fig = plt.gcf()  # pylint: disable=unreachable
        fig.set_size_inches(9, max(4, top_n * 0.35))
        plt.title("SHAP — Importancia global (media |SHAP|)", fontsize=12)
        return _fig_to_b64(fig)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("shap_summary_bar fallo: %s", exc)
        return None


def shap_beeswarm(
    model: Any,
    x_train: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 20,
) -> Optional[str]:
    """Beeswarm plot: distribucion del impacto SHAP por feature.

    Muestra la direccion del efecto (positivo/negativo) y la magnitud
    para cada observacion, agrupado por feature.

    Args:
        model: Estimador sklearn entrenado.
        x_train: Features de entrenamiento.
        feature_names: Nombres de las features.
        top_n: Numero maximo de features a mostrar.

    Returns:
        Imagen PNG en base64, o None si falla el calculo.
    """
    try:
        explainer, shap_values, x_aligned = _build_explainer(model, x_train)
        base_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(shap_values), base_val),
            data=x_aligned.values,
            feature_names=feature_names,
        )
        shap.plots.beeswarm(explanation, max_display=top_n, show=False)
        fig = plt.gcf()  # pylint: disable=unreachable
        fig.set_size_inches(9, max(4, top_n * 0.35))
        plt.title("SHAP — Distribucion del impacto por feature", fontsize=12)
        return _fig_to_b64(fig)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("shap_beeswarm fallo: %s", exc)
        return None


def _build_waterfall_explanation(
    explainer: Any,
    row: pd.DataFrame,
    feature_names: List[str],
) -> shap.Explanation:
    """Construye un objeto shap.Explanation para una sola fila."""
    sv_single = explainer.shap_values(row)
    sv = sv_single[1][0] if isinstance(sv_single, list) else sv_single[0]
    base_val = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    return shap.Explanation(
        values=sv,
        base_values=base_val,
        data=row.values[0],
        feature_names=feature_names,
    )


def shap_waterfall_comparison(
    model: Any,
    val_preds: ValPredictions,
    feature_names: List[str],
    max_display: int = 15,
) -> Optional[str]:
    """Waterfall side-by-side: peor prediccion vs mejor prediccion.

    Peor: indice donde |proba - label| es maximo (mayor error de confianza).
    Mejor: indice donde |proba - label| es minimo Y la prediccion es correcta.

    Args:
        model: Estimador sklearn entrenado.
        val_preds: ValPredictions con X, y_true e y_proba de validacion.
        feature_names: Nombres de las features.
        max_display: Numero maximo de features por waterfall.

    Returns:
        Imagen PNG en base64 con dos subplots, o None si falla.
    """
    try:
        errors = np.abs(val_preds.y_proba - val_preds.y_true.values.astype(float))
        worst_idx = int(np.argmax(errors))

        y_pred_binary = (val_preds.y_proba >= 0.5).astype(int)
        correct_mask = y_pred_binary == val_preds.y_true.values.astype(int)
        if correct_mask.any():
            errors_correct = np.where(correct_mask, errors, np.inf)
            best_idx = int(np.argmin(errors_correct))
        else:
            best_idx = int(np.argmin(errors))

        explainer, _, _ = _build_explainer(model, val_preds.X)

        exp_worst = _build_waterfall_explanation(
            explainer, val_preds.X.iloc[[worst_idx]], feature_names
        )
        exp_best = _build_waterfall_explanation(
            explainer, val_preds.X.iloc[[best_idx]], feature_names
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, max(10, max_display * 0.8)))

        plt.sca(axes[0])
        shap.plots.waterfall(exp_worst, max_display=max_display, show=False)
        axes[0].set_title(
            f"Peor prediccion — "
            f"label={int(val_preds.y_true.iloc[worst_idx])}  proba={val_preds.y_proba[worst_idx]:.3f}",
            fontsize=10,
        )

        plt.sca(axes[1])
        shap.plots.waterfall(exp_best, max_display=max_display, show=False)
        axes[1].set_title(
            f"Mejor prediccion — "
            f"label={int(val_preds.y_true.iloc[best_idx])}  proba={val_preds.y_proba[best_idx]:.3f}",
            fontsize=10,
        )

        fig.suptitle("SHAP Waterfall — peor vs mejor prediccion", fontsize=12)
        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("shap_waterfall_comparison fallo: %s", exc)
        return None


def compute_shap_plots(
    model: Any,
    x_train: pd.DataFrame,
    val_preds: ValPredictions,
    feature_names: List[str],
) -> Dict[str, Optional[str]]:
    """Calcula todos los plots SHAP y los devuelve como base64.

    Punto de entrada unico para el pipeline de entrenamiento.

    Args:
        model: Estimador sklearn entrenado.
        x_train: Features de entrenamiento (para calcular SHAP values globales).
        val_preds: ValPredictions con X, y_true e y_proba de validacion.
        feature_names: Nombres de las features.

    Returns:
        Diccionario con claves 'summary_bar', 'beeswarm', 'waterfall_comparison'.
        Cada valor es una cadena base64 PNG o None si fallo.
    """
    print("  [SHAP] Calculando summary bar...")
    shap_bar = shap_summary_bar(model, x_train, feature_names)
    print("  [SHAP] Calculando beeswarm...")
    beeswarm = shap_beeswarm(model, x_train, feature_names)
    print("  [SHAP] Calculando waterfall (peor vs mejor prediccion)...")
    waterfall = shap_waterfall_comparison(model, val_preds, feature_names)
    return {"summary_bar": shap_bar, "beeswarm": beeswarm, "waterfall_comparison": waterfall}
