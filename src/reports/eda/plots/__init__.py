"""
Visualizaciones del EDA — re-exports para compatibilidad con imports existentes.
"""
from src.reports.eda.plots.basic import (
    correlation_heatmap,
    nulls_bar,
    numeric_boxplots,
    numeric_histograms,
    numeric_stats_table,
    numeric_vs_target_hist,
    target_pie,
    unique_values_bar,
)
from src.reports.eda.plots.bivariate import (
    age_cryo_bar,
    cryo_homeplanet_heatmap,
    deck_homeplanet_heatmap,
)
from src.reports.eda.plots.cabin import (
    deck_homeplanet_heatmap as cabin_deck_homeplanet_heatmap,
    deck_transport_rate_bar,
    side_transport_rate_bar,
)
from src.reports.eda.plots.categorical import (
    categorical_double_bar,
    decisions_table,
    groupsize_bar,
)
from src.reports.eda.plots.domain_rules import (
    imputation_opportunities_bar,
    violations_table,
)
from src.reports.eda.plots.spending import (
    spending_per_service_bar,
    total_spending_compare,
    zero_inflation_bar,
)

__all__ = [
    # basic
    "unique_values_bar", "nulls_bar", "target_pie", "numeric_stats_table",
    "numeric_histograms", "numeric_boxplots", "correlation_heatmap", "numeric_vs_target_hist",
    # categorical
    "categorical_double_bar", "decisions_table", "groupsize_bar",
    # spending
    "total_spending_compare", "spending_per_service_bar", "zero_inflation_bar",
    # cabin
    "deck_transport_rate_bar", "side_transport_rate_bar", "cabin_deck_homeplanet_heatmap",
    # domain_rules
    "violations_table", "imputation_opportunities_bar",
    # bivariate
    "cryo_homeplanet_heatmap", "deck_homeplanet_heatmap", "age_cryo_bar",
]
