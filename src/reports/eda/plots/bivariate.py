"""
Visualizaciones de análisis bivariado entre variables clave.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cryo_homeplanet_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Tasa de Transported por (HomePlanet × CryoSleep)."""
    return px.imshow(
        pivot,
        text_auto=".2%",
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=1,
        title="Tasa Transported — HomePlanet × CryoSleep (motivó GroupAllCryo en fs-013)",
        labels={"color": "Transported rate"},
    )


def deck_homeplanet_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Distribución de pasajeros por (HomePlanet × Deck)."""
    return px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Distribución HomePlanet × Deck — valida reglas físicas R4/R5",
        labels={"color": "Pasajeros"},
    )


def age_cryo_bar(age_cryo_df: pd.DataFrame) -> go.Figure:
    """Barplot de edad media, gasto medio y tasa transported por CryoSleep."""
    fig = go.Figure()
    metrics = [
        ("Edad media", "#3498db"),
        ("Gasto medio", "#e74c3c"),
        ("Transported rate", "#2ecc71"),
    ]
    for metric, color in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=age_cryo_df["CryoSleep"],
            y=age_cryo_df[metric],
            marker_color=color,
            text=age_cryo_df[metric].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title_text="Age, Gasto y Transported por estado CryoSleep",
        height=420,
    )
    return fig
