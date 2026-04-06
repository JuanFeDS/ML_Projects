"""
Funciones de visualizacion para el script de feature engineering.

Cada funcion recibe datos ya calculados y devuelve una figura Plotly.
El script 02_features.py las orquesta junto con HTMLReport.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def derived_feature_double_bar(
    summary: pd.DataFrame,
    col_name: str,
    label: str,
) -> go.Figure:
    """Doble barplot de frecuencia y tasa de Transported para una feature derivada.

    Args:
        summary: DataFrame con columnas [col_name, 'count', 'tasa'].
        col_name: Nombre de la columna en summary (eje X).
        label: Titulo descriptivo de la feature para el titulo del grafico.

    Returns:
        Figura Plotly con dos paneles.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{label} — frecuencia",
            f"{label} — tasa de Transported",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=summary[col_name],
            y=summary["count"],
            name="Frecuencia",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary[col_name],
            y=summary["tasa"],
            name="Tasa",
            marker_color="#e94560",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title_text=label, height=400)
    return fig


def total_spending_compare(
    df_raw: pd.DataFrame,
    target: str,
) -> go.Figure:
    """Comparativa TotalSpending crudo vs log-transformado por clase.

    Reconstruye TotalSpending y TotalSpending_Log desde el DataFrame crudo.

    Args:
        df_raw: DataFrame con columnas de gasto y target.
        target: Nombre de la columna objetivo.

    Returns:
        Figura Plotly con dos paneles.
    """
    spending_cols = [
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ]
    import numpy as np  # pylint: disable=import-outside-toplevel

    df_tmp = df_raw.copy()
    df_tmp["TotalSpending"] = df_tmp[spending_cols].fillna(0).sum(axis=1)
    df_tmp["TotalSpending_Log"] = np.log1p(df_tmp["TotalSpending"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["TotalSpending (crudo)", "TotalSpending_Log"],
    )
    for col, c_idx, first in [
        ("TotalSpending", 1, True),
        ("TotalSpending_Log", 2, False),
    ]:
        fig.add_trace(
            go.Histogram(
                x=df_tmp.loc[~df_tmp[target], col],
                name="False",
                opacity=0.7,
                marker_color="#636EFA",
                nbinsx=60,
                showlegend=first,
            ),
            row=1,
            col=c_idx,
        )
        fig.add_trace(
            go.Histogram(
                x=df_tmp.loc[df_tmp[target], col],
                name="True",
                opacity=0.7,
                marker_color="#EF553B",
                nbinsx=60,
                showlegend=False,
            ),
            row=1,
            col=c_idx,
        )
    fig.update_layout(
        barmode="overlay",
        title_text="Comparativa TotalSpending vs TotalSpending_Log por clase",
        height=400,
    )
    return fig
