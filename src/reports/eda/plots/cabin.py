"""
Visualizaciones del análisis de Cabin: Deck, Side y distribución HomePlanet × Deck.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def deck_transport_rate_bar(summary: pd.DataFrame, chi2: float, p: float) -> go.Figure:
    """Tasa de Transported y frecuencia por Deck."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Frecuencia por Deck", "Tasa Transported por Deck"],
    )
    fig.add_trace(go.Bar(
        x=summary["Deck"].astype(str), y=summary["count"],
        text=summary["pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=summary["Deck"].astype(str), y=summary["tasa_transported"],
        text=summary["tasa_transported"].apply(lambda x: f"{x:.2%}"),
        textposition="outside", marker_color="#e94560", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(
        title_text=f"Deck  |  chi²={chi2:.1f}, p={p:.2e}", height=400,
    )
    return fig


def side_transport_rate_bar(summary: pd.DataFrame, chi2: float, p: float) -> go.Figure:
    """Tasa de Transported por Side (P vs S)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Frecuencia por Side", "Tasa Transported por Side"])
    fig.add_trace(go.Bar(
        x=summary["Side"].astype(str), y=summary["count"],
        text=summary["pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=summary["Side"].astype(str), y=summary["tasa_transported"],
        text=summary["tasa_transported"].apply(lambda x: f"{x:.2%}"),
        textposition="outside", marker_color="#e94560", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(title_text=f"Side  |  chi²={chi2:.1f}, p={p:.2e}", height=400)
    return fig


def deck_homeplanet_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Mapa de calor HomePlanet × Deck — valida reglas físicas R4/R5."""
    return px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Distribución HomePlanet × Deck (valida reglas físicas: A/B/C→Europa, G→Earth)",
        labels={"color": "Pasajeros"},
    )
