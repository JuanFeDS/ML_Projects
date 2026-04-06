"""
Funciones de visualizacion para el EDA de Spaceship Titanic.

Cada funcion recibe datos ya calculados y devuelve una figura Plotly.
El script 01_eda.py las orquesta junto con MarkdownReport y HTMLReport.
"""
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def unique_values_bar(dtypes_df: pd.DataFrame) -> go.Figure:
    """Barplot de valores unicos por columna, coloreado por tipo.

    Args:
        dtypes_df: DataFrame con columnas 'Columna', 'Tipo', 'Valores unicos'.

    Returns:
        Figura Plotly.
    """
    return px.bar(
        dtypes_df,
        x="Columna",
        y="Valores unicos",
        color="Tipo",
        title="Valores unicos por columna",
        text="Valores unicos",
    )


def nulls_bar(nulls_df: pd.DataFrame) -> go.Figure:
    """Barplot de porcentaje de nulos por columna.

    Args:
        nulls_df: DataFrame con columnas 'Columna' y '% Nulos'.

    Returns:
        Figura Plotly.
    """
    fig = px.bar(
        nulls_df,
        x="Columna",
        y="% Nulos",
        text="% Nulos",
        title="Porcentaje de valores nulos por columna",
        color="% Nulos",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return fig


def target_pie(balance_df: pd.DataFrame) -> go.Figure:
    """Grafico de torta con la distribucion del target.

    Args:
        balance_df: DataFrame con columnas 'Clase' y 'Conteo'.

    Returns:
        Figura Plotly.
    """
    return px.pie(
        balance_df,
        names="Clase",
        values="Conteo",
        title="Distribucion del target Transported",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )


def numeric_stats_table(desc: pd.DataFrame) -> go.Figure:
    """Tabla interactiva de estadisticas descriptivas.

    Args:
        desc: DataFrame resultado de df.describe().T.

    Returns:
        Figura Plotly.
    """
    return go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Feature"] + list(desc.columns),
                    "fill_color": "#0f3460",
                    "font": {"color": "white"},
                },
                cells={
                    "values": [desc.index.tolist()]
                    + [desc[c].tolist() for c in desc.columns],
                    "fill_color": [["#f0f4f8", "#ffffff"] * (len(desc) // 2 + 1)],
                },
            )
        ]
    )


def numeric_histograms(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    """Subplots con histogramas de variables numericas.

    Args:
        df: DataFrame con las columnas especificadas.
        cols: Lista de nombres de columnas a graficar.

    Returns:
        Figura Plotly con subplots 2×3.
    """
    fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
    for i, col in enumerate(cols):
        row, c = divmod(i, 3)
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
            row=row + 1,
            col=c + 1,
        )
    fig.update_layout(title_text="Histogramas — variables numericas", height=500)
    return fig


def numeric_boxplots(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    """Subplots con boxplots de variables numericas.

    Args:
        df: DataFrame con las columnas especificadas.
        cols: Lista de nombres de columnas a graficar.

    Returns:
        Figura Plotly con subplots 2×3.
    """
    fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
    for i, col in enumerate(cols):
        row, c = divmod(i, 3)
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col, showlegend=False),
            row=row + 1,
            col=c + 1,
        )
    fig.update_layout(title_text="Boxplots — variables numericas", height=500)
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Mapa de calor de la matriz de correlacion.

    Args:
        corr_matrix: DataFrame cuadrado de correlaciones.

    Returns:
        Figura Plotly.
    """
    return px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlacion (numericas + target)",
    )


def categorical_double_bar(
    summary: pd.DataFrame,
    col: str,
    chi2: float,
    p_val: float,
    dof: int,
) -> go.Figure:
    """Doble barplot de frecuencia y tasa de Transported para una variable categorica.

    Args:
        summary: DataFrame con columnas [col, 'count', 'pct', 'tasa_transported'].
        col: Nombre de la columna categorica.
        chi2: Estadistico chi-cuadrado.
        p_val: P-valor del test.
        dof: Grados de libertad.

    Returns:
        Figura Plotly con dos paneles.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{col}: distribucion de frecuencias",
            f"{col}: tasa de Transported por categoria",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=summary[col].astype(str),
            y=summary["count"],
            name="Frecuencia",
            text=summary["pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary[col].astype(str),
            y=summary["tasa_transported"],
            name="Tasa Transported",
            text=summary["tasa_transported"].apply(lambda x: f"{x:.2%}"),
            textposition="outside",
            marker_color="#e94560",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title_text=f"{col}  |  chi²={chi2:.1f}, p={p_val:.2e}, df={dof}",
        height=400,
    )
    return fig


def numeric_vs_target_hist(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    group_false: pd.Series,
    group_true: pd.Series,
    col: str,
    p_mw: float,
    r_pearson: float,
) -> go.Figure:
    """Histograma superpuesto por clase para una variable numerica.

    Args:
        group_false: Valores donde Transported=False.
        group_true: Valores donde Transported=True.
        col: Nombre de la variable.
        p_mw: P-valor Mann-Whitney U.
        r_pearson: Correlacion de Pearson con el target.

    Returns:
        Figura Plotly.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=group_false,
            name="Transported=False",
            opacity=0.7,
            nbinsx=40,
            marker_color="#636EFA",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=group_true,
            name="Transported=True",
            opacity=0.7,
            nbinsx=40,
            marker_color="#EF553B",
        )
    )
    fig.update_layout(
        barmode="overlay",
        title_text=(
            f"{col} — distribucion por clase  |  "
            f"MW p={p_mw:.2e}  |  r={r_pearson:.3f}"
        ),
        xaxis_title=col,
        yaxis_title="Frecuencia",
        height=380,
        legend={"x": 0.7, "y": 0.95},
    )
    return fig


def total_spending_compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    group_f_raw: pd.Series,
    group_t_raw: pd.Series,
    group_f_log: pd.Series,
    group_t_log: pd.Series,
    r_raw: float,
    r_log: float,
) -> go.Figure:
    """Comparativa TotalSpending crudo vs log-transformado por clase.

    Args:
        group_f_raw: TotalSpending de Transported=False.
        group_t_raw: TotalSpending de Transported=True.
        group_f_log: TotalSpending_Log de Transported=False.
        group_t_log: TotalSpending_Log de Transported=True.
        r_raw: Correlacion Pearson de TotalSpending con target.
        r_log: Correlacion Pearson de TotalSpending_Log con target.

    Returns:
        Figura Plotly con dos paneles.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["TotalSpending (crudo)", "TotalSpending_Log"],
    )
    fig.add_trace(
        go.Histogram(
            x=group_f_raw,
            name="Transported=False",
            opacity=0.7,
            marker_color="#636EFA",
            nbinsx=50,
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=group_t_raw,
            name="Transported=True",
            opacity=0.7,
            marker_color="#EF553B",
            nbinsx=50,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=group_f_log,
            name="Transported=False",
            opacity=0.7,
            marker_color="#636EFA",
            nbinsx=50,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=group_t_log,
            name="Transported=True",
            opacity=0.7,
            marker_color="#EF553B",
            nbinsx=50,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        barmode="overlay",
        title_text=f"TotalSpending: r={r_raw:.3f}  ->  TotalSpending_Log: r={r_log:.3f}",
        height=400,
    )
    return fig


def groupsize_bar(
    target_by_gs: pd.DataFrame, chi2_gs: float, p_gs: float
) -> go.Figure:
    """Barplot de tasa de Transported por GroupSize.

    Args:
        target_by_gs: DataFrame con columnas 'GroupSize', 'tasa_transported', 'n'.
        chi2_gs: Estadistico chi-cuadrado.
        p_gs: P-valor.

    Returns:
        Figura Plotly.
    """
    fig = px.bar(
        target_by_gs,
        x="GroupSize",
        y="tasa_transported",
        text=target_by_gs["tasa_transported"].apply(lambda x: f"{x:.2%}"),
        title=f"GroupSize — tasa de Transported  |  chi²={chi2_gs:.1f}, p={p_gs:.2e}",
        color="tasa_transported",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(textposition="outside")
    return fig


def spending_categories_bar(
    target_by_sc: pd.DataFrame, chi2_sc: float, p_sc: float
) -> go.Figure:
    """Barplot de tasa de Transported por SpendingCategories.

    Args:
        target_by_sc: DataFrame con columnas 'SpendingCategories', 'tasa_transported', 'n'.
        chi2_sc: Estadistico chi-cuadrado.
        p_sc: P-valor.

    Returns:
        Figura Plotly.
    """
    fig = px.bar(
        target_by_sc,
        x="SpendingCategories",
        y="tasa_transported",
        text=target_by_sc["tasa_transported"].apply(lambda x: f"{x:.2%}"),
        title=(
            f"SpendingCategories — tasa de Transported  |  "
            f"chi²={chi2_sc:.1f}, p={p_sc:.2e}"
        ),
        color="tasa_transported",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(textposition="outside")
    return fig


def decisions_table(decisions: pd.DataFrame) -> go.Figure:
    """Tabla coloreada de decisiones de features.

    Args:
        decisions: DataFrame con columnas 'Feature', 'Accion', 'Tipo', 'Justificacion'.

    Returns:
        Figura Plotly.
    """
    row_colors = [
        "#d4f7d4"
        if "MANTENER" in str(v)
        else "#ffd6d6"
        if v == "DESCARTAR"
        else "#fff3cd"
        for v in decisions["Accion"]
    ]
    return go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(decisions.columns),
                    "fill_color": "#0f3460",
                    "font": {"color": "white"},
                    "align": "left",
                },
                cells={
                    "values": [decisions[c].tolist() for c in decisions.columns],
                    "fill_color": [row_colors]
                    + [["#f8f9fa"] * len(decisions)]
                    * (len(decisions.columns) - 1),
                    "align": "left",
                },
            )
        ]
    )
