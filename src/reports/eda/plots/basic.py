"""
Visualizaciones de análisis básico: estructura, nulos, target y numéricas.
"""
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def unique_values_bar(dtypes_df: pd.DataFrame) -> go.Figure:
    return px.bar(
        dtypes_df, x="Columna", y="Valores unicos", color="Tipo",
        title="Valores únicos por columna", text="Valores unicos",
    )


def nulls_bar(nulls_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        nulls_df, x="Columna", y="% Nulos", text="% Nulos",
        title="Porcentaje de valores nulos por columna",
        color="% Nulos", color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return fig


def target_pie(balance_df: pd.DataFrame) -> go.Figure:
    return px.pie(
        balance_df, names="Clase", values="Conteo",
        title="Distribución del target Transported",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )


def numeric_stats_table(desc: pd.DataFrame) -> go.Figure:
    return go.Figure(data=[go.Table(
        header={"values": ["Feature"] + list(desc.columns), "fill_color": "#0f3460", "font": {"color": "white"}},
        cells={"values": [desc.index.tolist()] + [desc[c].tolist() for c in desc.columns],
               "fill_color": [["#f0f4f8", "#ffffff"] * (len(desc) // 2 + 1)]},
    )])


def numeric_histograms(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
    for i, col in enumerate(cols):
        row, c = divmod(i, 3)
        fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, showlegend=False), row=row + 1, col=c + 1)
    fig.update_layout(title_text="Histogramas — variables numéricas", height=500)
    return fig


def numeric_boxplots(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
    for i, col in enumerate(cols):
        row, c = divmod(i, 3)
        fig.add_trace(go.Box(y=df[col].dropna(), name=col, showlegend=False), row=row + 1, col=c + 1)
    fig.update_layout(title_text="Boxplots — variables numéricas", height=500)
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    return px.imshow(
        corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Matriz de correlación (numéricas + target)",
    )


def numeric_vs_target_hist(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    group_false: pd.Series, group_true: pd.Series,
    col: str, p_mw: float, r_pearson: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=group_false, name="Transported=False", opacity=0.7, nbinsx=40, marker_color="#636EFA"))
    fig.add_trace(go.Histogram(x=group_true, name="Transported=True", opacity=0.7, nbinsx=40, marker_color="#EF553B"))
    fig.update_layout(
        barmode="overlay",
        title_text=f"{col} — distribución por clase  |  MW p={p_mw:.2e}  |  r={r_pearson:.3f}",
        xaxis_title=col, yaxis_title="Frecuencia", height=380, legend={"x": 0.7, "y": 0.95},
    )
    return fig
