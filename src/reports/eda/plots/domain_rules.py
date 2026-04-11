"""
Visualizaciones de validación de reglas físicas e imputación.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def violations_table(violations_df: pd.DataFrame) -> go.Figure:
    """Tabla de violaciones a reglas físicas, coloreada por severidad."""
    row_colors = [
        "#ffd6d6" if v > 0 else "#d4f7d4"
        for v in violations_df["Violaciones"]
    ]
    return go.Figure(data=[go.Table(
        header={
            "values": list(violations_df.columns),
            "fill_color": "#0f3460", "font": {"color": "white"}, "align": "left",
        },
        cells={
            "values": [violations_df[c].tolist() for c in violations_df.columns],
            "fill_color": [row_colors] + [["#f8f9fa"] * len(violations_df)] * (len(violations_df.columns) - 1),
            "align": "left",
        },
    )])


def imputation_opportunities_bar(opps_df: pd.DataFrame) -> go.Figure:
    """Barplot de NaN resolubles vs no resolubles por variable."""
    opps_df = opps_df.copy()
    opps_df["No resolubles"] = opps_df["NaN total"] - opps_df["Resolubles"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=opps_df["Variable"], y=opps_df["Resolubles"],
        name="Resolubles por inferencia", marker_color="#2ecc71",
        text=opps_df["% resolubles"].apply(lambda x: f"{x:.0f}%"),
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=opps_df["Variable"], y=opps_df["No resolubles"],
        name="Requieren imputación estadística", marker_color="#e74c3c",
    ))
    fig.update_layout(
        barmode="stack",
        title_text="NaN resolubles por inferencia de dominio vs imputación estadística",
        yaxis_title="Número de registros", height=400,
    )
    return fig
