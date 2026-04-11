"""
Orquestador de reportes EDA.

Recibe los resultados de todos los run_* y construye el reporte MD + HTML
llamando a las funciones de plots/ por sección. Cada sección es independiente
— si un análisis no está disponible en el dict, la sección se omite.
"""
from typing import Any, Dict

import pandas as pd

from src.config.settings import DOCS_DIR, REPORTS_DIR
from src.features.constants import RAW_NUMERIC, TARGET
from src.reports.builder import HTMLReport, MarkdownReport
from src.reports.eda.plots.basic import (
    correlation_heatmap, nulls_bar, numeric_boxplots, numeric_histograms,
    numeric_stats_table, numeric_vs_target_hist, target_pie, unique_values_bar,
)
from src.reports.eda.plots.bivariate import age_cryo_bar, cryo_homeplanet_heatmap, deck_homeplanet_heatmap
from src.reports.eda.plots.cabin import deck_transport_rate_bar, side_transport_rate_bar, deck_homeplanet_heatmap as cabin_deck_heatmap
from src.reports.eda.plots.categorical import categorical_double_bar, decisions_table, groupsize_bar
from src.reports.eda.plots.domain_rules import imputation_opportunities_bar, violations_table
from src.reports.eda.plots.spending import spending_per_service_bar, total_spending_compare, zero_inflation_bar
from src.reports.experiments.model_cards import write_data_quality_doc

# ---------------------------------------------------------------------------
# Textos por sección
# ---------------------------------------------------------------------------

_T = {
    "intro": (
        "Este reporte documenta el análisis exploratorio del dataset **Spaceship Titanic** (Kaggle). "
        "El objetivo es entender la estructura de los datos, validar reglas del dominio e identificar "
        "señales estadísticas que justifiquen las decisiones de feature engineering."
    ),
    "dtypes": (
        "El dataset combina identificadores (`PassengerId`, `Name`, `Cabin`), categóricas de baja "
        "cardinalidad (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`) y numéricas de gasto. "
        "`Cabin` tiene >6500 valores únicos — se descompone en Deck/Num/Side."
    ),
    "nulls": (
        "Los nulos son uniformes (~2% por columna), sin ninguna variable que supere el 2.5%. "
        "El patrón sugiere MCAR en la mayoría de casos, aunque la sección de reglas de dominio "
        "muestra que muchos NaN son **resolubles por inferencia**."
    ),
    "target": (
        "El dataset está prácticamente balanceado (50.36% vs 49.64%). Esto elimina la necesidad de "
        "rebalanceo y permite usar accuracy como métrica principal."
    ),
    "numeric": (
        "Las variables de gasto muestran distribuciones fuertemente sesgadas a la derecha (skewness > 6). "
        "La mediana es 0 en todos los servicios. `Age` es la única con distribución aproximadamente normal."
    ),
    "cat_intro": (
        "Test chi-cuadrado (χ²) para evaluar la asociación de cada categórica con el target. "
        "p < 0.05 indica dependencia estadísticamente significativa."
    ),
    "cabin": (
        "**Cabin** se descompone en Deck, CabinNumber y Side. Deck tiene chi²=392.3 y Side chi²=91.1 "
        "(ambos p<0.001). El heatmap HomePlanet × Deck confirma visualmente las reglas físicas: "
        "decks A/B/C son casi exclusivos de Europa; G de Earth."
    ),
    "spending_deep": (
        "Análisis por servicio individual. La mayoría de pasajeros no gasta en ningún servicio "
        "(zero-inflation). Los no transportados concentran casi todo el gasto: gastar en cualquier "
        "servicio reduce marcadamente la probabilidad de transporte."
    ),
    "domain_rules": (
        "Validación empírica de las 5 reglas físicas del dataset. Las violaciones son mínimas "
        "(<0.1%), confirmando que las reglas son casi perfectas. La tabla de oportunidades muestra "
        "cuántos NaN pueden resolverse por inferencia antes de recurrir a imputación estadística."
    ),
    "bivariate": (
        "Interacciones entre pares de variables. El heatmap HomePlanet × CryoSleep muestra que "
        "pasajeros europeos en CryoSleep tienen >86% de tasa de transporte — combinación que motivó "
        "las features `GroupAllCryo` y `GroupAnyCryo` en fs-013."
    ),
    "decisions": (
        "Tabla de decisiones de feature engineering derivadas de este análisis. "
        "Implementadas en `src/features/engineering/`."
    ),
}

_CHI2_INTERPRETATIONS = {
    "CryoSleep": (
        "**Variable más discriminativa.** CryoSleep=True → tasa de transporte >80%. "
        "Es la feature con mayor poder predictivo."
    ),
    "HomePlanet": "Asociación fuerte (χ²≈325). Europa tiene la mayor tasa de transporte.",
    "Destination": "Asociación moderada (χ²≈106). PSO J318.5-22 muestra tasas atípicas.",
    "VIP": "Asociación débil pero significativa (p=0.002). VIP → menor tasa de transporte.",
}
_CHI2_DEFAULT = "Asociación estadísticamente significativa. Ver gráfico para detalle."


# ---------------------------------------------------------------------------
# Secciones independientes
# ---------------------------------------------------------------------------

def _section_basic(md: MarkdownReport, html: HTMLReport, basic: dict, target: dict) -> None:
    md.add_section("Análisis Básico")
    html.add_section("Análisis Básico")
    md.add_text(_T["dtypes"])
    md.add_table(basic["dtypes"], index=False)
    html.add_figure(unique_values_bar(basic["dtypes"]), title="Tipos y Valores Únicos")
    md.add_subsection("Valores nulos")
    md.add_text(_T["nulls"])
    md.add_table(basic["nulls"], index=False)
    html.add_figure(nulls_bar(basic["nulls"]), title="Distribución de Nulos")

    md.add_section("Balance del Target")
    html.add_section("Balance del Target")
    md.add_text(_T["target"])
    md.add_table(target["balance_df"], index=False)
    html.add_figure(target_pie(target["balance_df"]), title="Proporción del Target")


def _section_numeric(md: MarkdownReport, html: HTMLReport, df: pd.DataFrame, stats: dict) -> None:
    md.add_section("Variables Numéricas")
    html.add_section("Variables Numéricas")
    md.add_text(_T["numeric"])
    md.add_table(stats["numeric_desc"], index=True)
    html.add_figure(numeric_stats_table(stats["numeric_desc"]), title="Estadísticas Descriptivas")
    html.add_figure(numeric_histograms(df, RAW_NUMERIC), title="Histogramas")
    html.add_figure(numeric_boxplots(df, RAW_NUMERIC), title="Boxplots")
    corr_df = df[RAW_NUMERIC + [TARGET]].copy()
    corr_df[TARGET] = corr_df[TARGET].astype(int)
    html.add_figure(correlation_heatmap(corr_df.corr().round(3)), title="Matriz de Correlación")


def _section_categorical(md: MarkdownReport, html: HTMLReport, stats: dict) -> None:
    md.add_section("Variables Categóricas vs Target")
    html.add_section("Variables Categóricas vs Target")
    md.add_text(_T["cat_intro"])
    for col, res in stats["cat_stats"].items():
        md.add_subsection(col)
        md.add_metric("χ²", f"{res['chi2']:.2f}")
        md.add_metric("p-valor", f"{res['p']:.2e}")
        md.add_text(_CHI2_INTERPRETATIONS.get(col, _CHI2_DEFAULT))
        html.add_figure(
            categorical_double_bar(res["summary"], col, res["chi2"], res["p"], res["dof"]),
            title=col,
        )


def _section_cabin(md: MarkdownReport, html: HTMLReport, cabin: dict) -> None:
    md.add_section("Análisis de Cabin")
    html.add_section("Análisis de Cabin")
    md.add_text(_T["cabin"])
    md.add_metric("χ² Deck", f"{cabin['deck']['chi2']:.1f}")
    md.add_metric("χ² Side", f"{cabin['side']['chi2']:.1f}")
    md.add_metric("Cabin con NaN", f"{cabin['cabin_null_pct']:.2f}%")
    html.add_figure(deck_transport_rate_bar(cabin["deck"]["summary"], cabin["deck"]["chi2"], cabin["deck"]["p"]), title="Deck vs Target")
    html.add_figure(side_transport_rate_bar(cabin["side"]["summary"], cabin["side"]["chi2"], cabin["side"]["p"]), title="Side vs Target")
    html.add_figure(cabin_deck_heatmap(cabin["deck_homeplanet"]), title="HomePlanet × Deck")


def _section_spending(md: MarkdownReport, html: HTMLReport, derived: dict, spending: dict) -> None:
    md.add_section("Análisis de Gasto")
    html.add_section("Análisis de Gasto")
    md.add_text(_T["spending_deep"])
    sp = derived["spending"]
    md.add_metric("r TotalSpending (crudo)", f"{sp['r_raw']:.3f}")
    md.add_metric("r TotalSpending (log1p)", f"{sp['r_log']:.3f}")
    html.add_figure(total_spending_compare(
        sp["group_f_raw"], sp["group_t_raw"], sp["group_f_log"], sp["group_t_log"],
        sp["r_raw"], sp["r_log"],
    ), title="TotalSpending: crudo vs log")
    html.add_figure(spending_per_service_bar(spending["per_service"]), title="Análisis por Servicio")
    html.add_figure(zero_inflation_bar(spending["zero_inflation"]["per_service_zero_pct"]), title="Zero-inflation por Servicio")
    md.add_table(spending["per_service"], index=False)


def _section_domain_rules(md: MarkdownReport, html: HTMLReport, domain: dict) -> None:
    md.add_section("Validación de Reglas de Dominio")
    html.add_section("Validación de Reglas de Dominio")
    md.add_text(_T["domain_rules"])
    md.add_table(domain["violations"], index=False)
    md.add_table(domain["imputation_opportunities"], index=False)
    html.add_figure(violations_table(domain["violations"]), title="Violaciones a Reglas Físicas")
    html.add_figure(imputation_opportunities_bar(domain["imputation_opportunities"]), title="NaN Resolubles por Inferencia")


def _section_bivariate(md: MarkdownReport, html: HTMLReport, bivariate: dict) -> None:
    md.add_section("Análisis Bivariado")
    html.add_section("Análisis Bivariado")
    md.add_text(_T["bivariate"])
    md.add_table(bivariate["age_cryo"], index=False)
    html.add_figure(cryo_homeplanet_heatmap(bivariate["cryo_homeplanet"]), title="HomePlanet × CryoSleep → Transported rate")
    html.add_figure(deck_homeplanet_heatmap(bivariate["deck_homeplanet"]), title="HomePlanet × Deck")
    html.add_figure(age_cryo_bar(bivariate["age_cryo"]), title="Age y Gasto por estado CryoSleep")


def _section_decisions(md: MarkdownReport, derived: dict, spending: dict) -> None:
    sp = derived["spending"]
    gs = derived["groupsize"]
    decisions = pd.DataFrame([
        {"Feature": "CryoSleep",     "Acción": "MANTENER",        "Justificación": "χ²=1861.75 — variable más discriminativa"},
        {"Feature": "HomePlanet",    "Acción": "MANTENER + TE",   "Justificación": "χ²=324.97, target encoding > OHE"},
        {"Feature": "Destination",   "Acción": "MANTENER + OHE",  "Justificación": "χ²=106.39, 3 categorías"},
        {"Feature": "VIP",           "Acción": "MANTENER",        "Justificación": "Señal débil pero significativa (p=0.002)"},
        {"Feature": "Cabin",         "Acción": "DESCOMPONER",     "Justificación": "Deck chi²=392.3, Side chi²=91.1 — target encoding"},
        {"Feature": "TotalSpending", "Acción": "CREAR + log1p",   "Justificación": f"r_log={sp['r_log']:.3f} vs r_raw={sp['r_raw']:.3f}"},
        {"Feature": "GroupSize",     "Acción": "CREAR",           "Justificación": f"χ²={gs['chi2']:.2f}, patrón no lineal"},
        {"Feature": "Age",           "Acción": "MANTENER",        "Justificación": "Normal (skew=0.42), imputar por grupo"},
        {"Feature": "Name",          "Acción": "DESCARTAR",       "Justificación": "Sin señal predictiva directa"},
        {"Feature": "PassengerId",   "Acción": "DESCARTAR (raw)", "Justificación": "Usar solo para extraer TravelGroup"},
    ])
    md.add_section("Decisiones de Feature Engineering")
    md.add_text(_T["decisions"])
    md.add_table(decisions, index=False)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def build_eda_report(df: pd.DataFrame, results: Dict[str, Any]) -> None:
    """Construye el reporte MD + HTML a partir de los resultados del análisis.

    Args:
        df: DataFrame original (para gráficos que requieren el raw).
        results: Dict con claves basic, target, stats, derived,
                 cabin, spending, domain_rules, bivariate.
    """
    md = MarkdownReport(title="EDA — Spaceship Titanic")
    html = HTMLReport(title="EDA — Spaceship Titanic")

    basic = results["basic"]
    target_res = results["target"]
    stats = results["stats"]
    derived = results["derived"]

    n_rows, n_cols = basic["shape"]
    html.add_intro(
        f"Análisis exploratorio de <b>Spaceship Titanic</b> ({n_rows:,} registros, {n_cols} variables)."
    )
    html.add_metrics_grid([
        (f"{n_rows:,}", "registros"), (n_cols, "variables"),
        (len(basic["nulls"]), "cols con nulos"), (basic["dupes"], "duplicados"),
        (f"{target_res['pcts'].get(True, 0):.1f}%", "tasa transported"),
    ])
    md.add_text(_T["intro"])

    _section_basic(md, html, basic, target_res)
    _section_numeric(md, html, df, stats)
    _section_categorical(md, html, stats)

    if "cabin" in results:
        _section_cabin(md, html, results["cabin"])

    if "spending" in results:
        _section_spending(md, html, derived, results["spending"])
    else:
        # Sección básica de gasto (backward compat si spending no está disponible)
        sp = derived["spending"]
        html.add_figure(total_spending_compare(
            sp["group_f_raw"], sp["group_t_raw"], sp["group_f_log"], sp["group_t_log"],
            sp["r_raw"], sp["r_log"],
        ), title="TotalSpending: crudo vs log")

    if "domain_rules" in results:
        _section_domain_rules(md, html, results["domain_rules"])

    if "bivariate" in results:
        _section_bivariate(md, html, results["bivariate"])

    _section_decisions(md, derived, results.get("spending", {"per_service": None}))

    out_dir = REPORTS_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    md.save(str(out_dir / "01_eda.md"))
    html.save(str(out_dir / "01_eda.html"))

    write_data_quality_doc(df, TARGET, str(DOCS_DIR / "data" / "data_quality.md"))
    print(f"[OK] Reportes generados en {out_dir}")
