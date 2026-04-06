"""
Funciones de visualizacion para el reporte de entrenamiento.

Cada funcion recibe datos ya calculados y devuelve una figura Plotly.
El script 03_train.py las orquesta junto con HTMLReport.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def cv_accuracy_bar(cv_results: pd.DataFrame) -> go.Figure:
    """Barplot de accuracy CV (media ± std) para todos los modelos.

    Args:
        cv_results: DataFrame con indice de nombres de modelo y columnas
            'cv_accuracy_mean' y 'cv_accuracy_std'.

    Returns:
        Figura Plotly.
    """
    fig = go.Figure(
        go.Bar(
            x=cv_results.index.tolist(),
            y=cv_results["cv_accuracy_mean"].tolist(),
            error_y={
                "type": "data",
                "array": cv_results["cv_accuracy_std"].tolist(),
                "visible": True,
            },
            marker_color="steelblue",
            text=[f"{v:.4f}" for v in cv_results["cv_accuracy_mean"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="CV Accuracy por modelo (media +/- std, 5-fold)",
        xaxis_title="Modelo",
        yaxis_title="Accuracy",
        yaxis={"range": [0, 1]},
        height=450,
    )
    return fig


def validation_metrics_bar(
    val_models: List[str],
    val_acc: List[float],
    val_roc: List[float],
) -> go.Figure:
    """Barplot agrupado de accuracy y ROC-AUC en validacion.

    Args:
        val_models: Nombres de los modelos a comparar.
        val_acc: Accuracy de validacion por modelo.
        val_roc: ROC-AUC de validacion por modelo.

    Returns:
        Figura Plotly.
    """
    fig = go.Figure(
        [
            go.Bar(
                name="val_accuracy",
                x=val_models,
                y=val_acc,
                text=[f"{v:.4f}" for v in val_acc],
                textposition="outside",
                marker_color="steelblue",
            ),
            go.Bar(
                name="val_roc_auc",
                x=val_models,
                y=val_roc,
                text=[f"{v:.4f}" for v in val_roc],
                textposition="outside",
                marker_color="darkorange",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Metricas de Validacion: Tuneado vs Stacking",
        yaxis={"range": [0, 1.1]},
        height=400,
    )
    return fig


def feature_importance_bar(
    model: object,
    feature_names: List[str],
    top_n: int = 20,
) -> Optional[go.Figure]:
    """Barplot horizontal de feature importance del modelo ganador.

    Soporta modelos con atributo `feature_importances_` o `coef_`.
    Devuelve None si el modelo no tiene ninguno de los dos.

    Args:
        model: Estimador sklearn entrenado.
        feature_names: Lista de nombres de features en el mismo orden.
        top_n: Numero maximo de features a mostrar.

    Returns:
        Figura Plotly o None.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig = go.Figure(
        go.Bar(
            x=fi["importance"].tolist(),
            y=fi["feature"].tolist(),
            orientation="h",
            marker_color="seagreen",
        )
    )
    fig.update_layout(
        title="Feature Importance — modelo ganador",
        xaxis_title="Importancia",
        yaxis={"autorange": "reversed"},
        height=500,
    )
    return fig
