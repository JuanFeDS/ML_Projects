"""
Visualizaciones de análisis de gasto: comparativa raw/log, por servicio y zero-inflation.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def total_spending_compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    group_f_raw: pd.Series, group_t_raw: pd.Series,
    group_f_log: pd.Series, group_t_log: pd.Series,
    r_raw: float, r_log: float,
) -> go.Figure:
    """Comparativa TotalSpending crudo vs log-transformado por clase."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["TotalSpending (crudo)", "TotalSpending_Log"])
    for grp, name, color, show in [
        (group_f_raw, "Transported=False", "#636EFA", True),
        (group_t_raw, "Transported=True", "#EF553B", False),
    ]:
        fig.add_trace(go.Histogram(x=grp, name=name, opacity=0.7, marker_color=color, nbinsx=50, showlegend=show), row=1, col=1)
    for grp, name, color in [
        (group_f_log, "Transported=False", "#636EFA"),
        (group_t_log, "Transported=True", "#EF553B"),
    ]:
        fig.add_trace(go.Histogram(x=grp, name=name, opacity=0.7, marker_color=color, nbinsx=50, showlegend=False), row=1, col=2)
    fig.update_layout(barmode="overlay", title_text=f"TotalSpending: r={r_raw:.3f}  →  TotalSpending_Log: r={r_log:.3f}", height=400)
    return fig


def spending_per_service_bar(per_service_df: pd.DataFrame) -> go.Figure:
    """Doble barplot: correlación con target y diferencia de tasa transported por servicio."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Correlación Pearson (log1p) con target", "Transported rate: gastadores vs no"],
    )
    fig.add_trace(go.Bar(
        x=per_service_df["Servicio"],
        y=per_service_df["r Pearson (log)"].abs(),
        text=per_service_df["r Pearson (log)"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        marker_color="#0f3460",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=per_service_df["Servicio"],
        y=per_service_df["Transported (no gasta)"],
        name="No gasta", marker_color="#636EFA",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=per_service_df["Servicio"],
        y=per_service_df["Transported (gasta)"],
        name="Gasta", marker_color="#EF553B",
    ), row=1, col=2)

    fig.update_layout(
        title_text="Análisis por servicio — poder discriminativo",
        barmode="group", height=420,
    )
    return fig


def zero_inflation_bar(per_service_zero_pct: pd.Series) -> go.Figure:
    """Barplot del porcentaje de ceros por columna de gasto."""
    df = per_service_zero_pct.reset_index()
    df.columns = ["Servicio", "% sin gasto"]
    fig = px.bar(
        df, x="Servicio", y="% sin gasto",
        text=df["% sin gasto"].apply(lambda x: f"{x:.1f}%"),
        title="Zero-inflation por servicio (% de pasajeros con gasto = 0)",
        color="% sin gasto", color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, 105])
    return fig
