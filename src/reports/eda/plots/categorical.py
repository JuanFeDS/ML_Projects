"""
Visualizaciones para variables categóricas vs target.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def categorical_double_bar(
    summary: pd.DataFrame, col: str, chi2: float, p_val: float, dof: int,
) -> go.Figure:
    """Doble barplot: frecuencia y tasa de Transported para una variable categórica."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{col}: frecuencias", f"{col}: tasa Transported"],
    )
    fig.add_trace(go.Bar(
        x=summary[col].astype(str), y=summary["count"], name="Frecuencia",
        text=summary["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=summary[col].astype(str), y=summary["tasa_transported"], name="Tasa Transported",
        text=summary["tasa_transported"].apply(lambda x: f"{x:.2%}"), textposition="outside",
        marker_color="#e94560", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(title_text=f"{col}  |  chi²={chi2:.1f}, p={p_val:.2e}, df={dof}", height=400)
    return fig


def decisions_table(decisions: pd.DataFrame) -> go.Figure:
    """Tabla coloreada de decisiones de feature engineering."""
    row_colors = [
        "#d4f7d4" if "MANTENER" in str(v) else "#ffd6d6" if v == "DESCARTAR" else "#fff3cd"
        for v in decisions["Acción"]
    ]
    return go.Figure(data=[go.Table(
        header={"values": list(decisions.columns), "fill_color": "#0f3460",
                "font": {"color": "white"}, "align": "left"},
        cells={"values": [decisions[c].tolist() for c in decisions.columns],
               "fill_color": [row_colors] + [["#f8f9fa"] * len(decisions)] * (len(decisions.columns) - 1),
               "align": "left"},
    )])


def groupsize_bar(target_by_gs: pd.DataFrame, chi2_gs: float, p_gs: float) -> go.Figure:
    """Barplot de tasa de Transported por GroupSize."""
    import plotly.express as px
    fig = px.bar(
        target_by_gs, x="GroupSize", y="tasa_transported",
        text=target_by_gs["tasa_transported"].apply(lambda x: f"{x:.2%}"),
        title=f"GroupSize — tasa de Transported  |  chi²={chi2_gs:.1f}, p={p_gs:.2e}",
        color="tasa_transported", color_continuous_scale="RdYlGn",
    )
    fig.update_traces(textposition="outside")
    return fig
