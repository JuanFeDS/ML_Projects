"""
Generación de reportes para el EDA.

Este módulo encapsula la creación de los reportes Markdown y HTML
para el análisis exploratorio de datos.
"""
from typing import Dict, Any

import pandas as pd

from src.config.settings import DOCS_DIR, REPORTS_DIR
from src.features.constants import TARGET, RAW_NUMERIC
from src.reports.builder import HTMLReport, MarkdownReport
from src.reports.experiments.model_cards import write_data_quality_doc
from src.reports.eda.plots import (
    categorical_double_bar,
    correlation_heatmap,
    groupsize_bar,
    nulls_bar,
    numeric_boxplots,
    numeric_histograms,
    numeric_stats_table,
    target_pie,
    total_spending_compare,
    unique_values_bar,
)

# ---------------------------------------------------------------------------
# Textos explicativos por sección (hardcodeados con contexto del dominio)
# ---------------------------------------------------------------------------

_TEXT_INTRO = (
    "Este reporte documenta el análisis exploratorio inicial del dataset **Spaceship Titanic** "
    "(Kaggle, 2022). El objetivo es entender la estructura de los datos, identificar problemas "
    "de calidad y extraer señales estadísticas que guíen el feature engineering. "
    "Las conclusiones de este análisis son el punto de partida de NB02."
)

_TEXT_DTYPES = (
    "El dataset combina variables de distintos tipos: identificadores (`PassengerId`, `Name`, `Cabin`), "
    "variables categóricas de baja cardinalidad (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`) y "
    "variables numéricas continuas de gasto (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`). "
    "`Cabin` tiene cardinalidad muy alta (>6500 valores únicos) — requiere descomposición en componentes "
    "(Deck, Num, Side) en lugar de usarse directamente."
)

_TEXT_NULLS = (
    "Los nulos son **uniformes y bajos** (~2% por columna), sin ninguna variable que supere el 2.5%. "
    "Este patrón sugiere ausencia aleatoria (MCAR) en lugar de pérdida sistemática. "
    "Estrategia: imputar categóricas con `'Unknown'` y numéricas con `0` (ausencia de gasto), "
    "con excepción de `Age` donde se evaluará imputación por mediana o por grupos."
)

_TEXT_TARGET = (
    "El dataset está **prácticamente balanceado**: 50.36% transportados vs 49.64% no transportados. "
    "Esta condición elimina la necesidad de técnicas de rebalanceo (SMOTE, class_weight) y permite "
    "usar accuracy como métrica principal sin penalizar clases minoritarias. "
    "El AUC-ROC complementará la evaluación para capturar el poder discriminativo global."
)

_TEXT_NUMERIC_STATS = (
    "Las variables de gasto muestran distribuciones **fuertemente sesgadas a la derecha** "
    "(skewness > 6 en todas), con mediana = 0. Esto indica que la gran mayoría de pasajeros "
    "no realizó ningún gasto. `FoodCourt` y `VRDeck` presentan los valores máximos más extremos. "
    "Se recomienda aplicar transformación `log1p` para reducir el sesgo antes del escalado. "
    "`Age` es la única variable con distribución aproximadamente normal (skew = 0.42)."
)

_TEXT_NUMERIC_CORR = (
    "La matriz de correlación revela relaciones entre variables de gasto que se explorarán "
    "como features compuestas. Se calculó incluyendo el target (codificado como 0/1) para "
    "identificar las variables con mayor señal lineal directa."
)

_TEXT_CAT_INTRO = (
    "Se aplicó el test **chi-cuadrado** (χ²) para evaluar la asociación entre cada variable "
    "categórica y el target. Un p-valor < 0.05 indica dependencia estadísticamente significativa. "
    "Los grados de libertad (df) dependen del número de categorías de cada variable."
)

_CHI2_INTERPRETATIONS = {
    "CryoSleep": (
        "**Variable más discriminativa del dataset.** Los pasajeros en criosueño tienen una tasa "
        "de transporte radicalmente distinta (>80%) vs los activos (~30%). Es la feature con "
        "mayor poder predictivo y será central en el feature engineering."
    ),
    "HomePlanet": (
        "Asociación fuerte (χ²=324.97). Los pasajeros de Europa tienen la mayor tasa de transporte, "
        "mientras que los de Mars la menor. Indica diferencias sistemáticas por origen."
    ),
    "Destination": (
        "Asociación moderada-alta (χ²=106.39). `PSO J318.5-22` muestra tasas de transporte "
        "marcadamente distintas a los otros destinos."
    ),
    "VIP": (
        "Asociación débil (χ²=12.1, p=0.002). El estado VIP tiene correlación negativa con "
        "ser transportado — los VIP son transportados con menor frecuencia, posiblemente por "
        "mayor gasto en servicios que los mantiene 'activos'."
    ),
}

_TEXT_CAT_DEFAULT = (
    "Asociación estadísticamente significativa con el target. "
    "Ver gráfico para distribución de frecuencias y tasa de Transported por categoría."
)

_TEXT_SPENDING_DERIVED = (
    "`TotalSpending` = suma de los 5 servicios de a bordo. La distribución cruda es bimodal "
    "(muchos ceros + cola larga). Tras aplicar `log1p`, la separación entre clases mejora "
    "notablemente: los no transportados concentran mayor gasto. Esta transformación se incluirá "
    "como feature en el pipeline."
)

_TEXT_GROUPSIZE_DERIVED = (
    "`GroupSize` se extrae del prefijo de `PassengerId`. Los pasajeros que viajan solos (tamaño=1) "
    "tienen la tasa de transporte más baja, mientras que grupos medianos (2-4) muestran tasas "
    "más altas. El test chi-cuadrado confirma que la asociación es estadísticamente significativa."
)

_TEXT_DECISIONS_INTRO = (
    "La siguiente tabla resume las decisiones de ingeniería tomadas a partir de este análisis. "
    "Estas decisiones se implementarán en el **Notebook NB03 — Feature Engineering**."
)


def build_eda_report(df: pd.DataFrame, analysis_results: Dict[str, Any]):
    """Construye ambos reportes (MD y HTML) a partir de los resultados del análisis.

    Args:
        df: DataFrame original para generar los gráficos.
        analysis_results: Diccionario con los resultados de las funciones en eda_pipeline.py.
    """
    md = MarkdownReport(title="EDA — Spaceship Titanic")
    html = HTMLReport(title="EDA — Spaceship Titanic")

    basic = analysis_results["basic"]
    target = analysis_results["target"]
    stats = analysis_results["stats"]
    derived = analysis_results["derived"]

    n_rows, n_cols = basic["shape"]
    n_nulls_cols = len(basic["nulls"])
    pct_transported = target["pcts"].get(True, 0)

    # ------------------------------------------------------------------
    # Introducción
    # ------------------------------------------------------------------
    md.add_text(_TEXT_INTRO)
    md.add_text("")

    html.add_intro(
        f"Este reporte consolida el análisis exploratorio del dataset "
        f"<b>Spaceship Titanic</b> ({n_rows:,} registros, {n_cols} variables). "
        f"Se examinan distribuciones, valores nulos y la relación con el target."
    )
    html.add_metrics_grid([
        (f"{n_rows:,}", "registros"),
        (n_cols, "variables"),
        (n_nulls_cols, "cols con nulos"),
        (basic["dupes"], "duplicados"),
        (f"{pct_transported:.1f}%", "tasa transported"),
    ])

    # ------------------------------------------------------------------
    # 1. Análisis Básico
    # ------------------------------------------------------------------
    md.add_section("Análisis Básico")
    html.add_section("Análisis Básico")

    md.add_text(_TEXT_DTYPES)
    md.add_text("")
    md.add_table(basic["dtypes"], index=False)
    html.add_figure(unique_values_bar(basic["dtypes"]), title="Tipos y Valores Únicos")

    # Nulos
    md.add_subsection("Valores nulos")
    md.add_text(_TEXT_NULLS)
    md.add_text("")
    md.add_table(basic["nulls"], index=False)
    html.add_figure(nulls_bar(basic["nulls"]), title="Distribución de Nulos")

    # ------------------------------------------------------------------
    # 2. Balance del Target
    # ------------------------------------------------------------------
    md.add_section("Balance del Target")
    html.add_section("Balance del Target")

    md.add_text(_TEXT_TARGET)
    md.add_text("")
    md.add_table(target["balance_df"], index=False)
    html.add_figure(target_pie(target["balance_df"]), title="Proporción del Target")

    # ------------------------------------------------------------------
    # 3. Variables Numéricas
    # ------------------------------------------------------------------
    md.add_section("Variables Numéricas")
    html.add_section("Variables Numéricas")

    md.add_text(_TEXT_NUMERIC_STATS)
    md.add_text("")
    md.add_table(stats["numeric_desc"], index=True)
    html.add_figure(numeric_stats_table(stats["numeric_desc"]), title="Estadísticas Descriptivas")
    html.add_figure(numeric_histograms(df, RAW_NUMERIC), title="Histogramas de Distribución")
    html.add_figure(numeric_boxplots(df, RAW_NUMERIC), title="Boxplots (Outliers)")

    # Correlaciones
    md.add_subsection("Correlaciones")
    md.add_text(_TEXT_NUMERIC_CORR)
    corr_df = df[RAW_NUMERIC + [TARGET]].copy()
    corr_df[TARGET] = corr_df[TARGET].astype(int)
    corr_matrix = corr_df.corr().round(3)
    html.add_figure(correlation_heatmap(corr_matrix), title="Matriz de Correlación")

    # ------------------------------------------------------------------
    # 4. Variables Categóricas vs Target
    # ------------------------------------------------------------------
    md.add_section("Variables Categóricas vs Target")
    html.add_section("Variables Categóricas vs Target")

    md.add_text(_TEXT_CAT_INTRO)
    md.add_text("")

    for col, res in stats["cat_stats"].items():
        md.add_subsection(col)
        significance = "✅ Significativa" if res["p"] < 0.05 else "⚠️ No significativa"
        md.add_metric("χ²", f"{res['chi2']:.2f}")
        md.add_metric("p-valor", f"{res['p']:.2e}")
        md.add_metric("Significancia", significance)
        interpretation = _CHI2_INTERPRETATIONS.get(col, _TEXT_CAT_DEFAULT)
        md.add_text("")
        md.add_text(interpretation)
        md.add_text("")
        html.add_figure(
            categorical_double_bar(res["summary"], col, res["chi2"], res["p"], res["dof"]),
            title=col,
        )

    # ------------------------------------------------------------------
    # 5. Análisis de Features Derivadas
    # ------------------------------------------------------------------
    md.add_section("Análisis de Features Derivadas")
    html.add_section("Análisis de Features Derivadas")

    # TotalSpending
    md.add_subsection("TotalSpending")
    md.add_text(_TEXT_SPENDING_DERIVED)
    sp = derived["spending"]
    md.add_metric("r (crudo)", f"{sp['r_raw']:.3f}")
    md.add_metric("r (log1p)", f"{sp['r_log']:.3f}")
    md.add_text(
        f"> La transformación log mejora la correlación de "
        f"`{sp['r_raw']:.3f}` → `{sp['r_log']:.3f}` con el target."
    )
    md.add_text("")
    html.add_figure(
        total_spending_compare(
            sp["group_f_raw"], sp["group_t_raw"],
            sp["group_f_log"], sp["group_t_log"],
            sp["r_raw"], sp["r_log"],
        ),
        title="Impacto de TotalSpending (Raw vs Log)",
    )

    # GroupSize
    md.add_subsection("GroupSize")
    md.add_text(_TEXT_GROUPSIZE_DERIVED)
    gs = derived["groupsize"]
    md.add_metric("χ²", f"{gs['chi2']:.2f}")
    md.add_metric("p-valor", f"{gs['p']:.2e}")
    md.add_text("")
    html.add_figure(
        groupsize_bar(gs["summary"], gs["chi2"], gs["p"]),
        title="Influencia del Tamaño de Grupo",
    )

    # ------------------------------------------------------------------
    # 6. Decisiones y Próximos Pasos
    # ------------------------------------------------------------------
    md.add_section("Decisiones y Próximos Pasos")

    md.add_text(_TEXT_DECISIONS_INTRO)
    md.add_text("")

    decisions = pd.DataFrame([
        {"Feature":      "CryoSleep",     "Acción": "MANTENER",          "Justificación": f"χ²=1861.75 — variable más discriminativa"},
        {"Feature":      "HomePlanet",    "Acción": "MANTENER + OHE",    "Justificación": f"χ²=324.97, 3 categorías"},
        {"Feature":      "Destination",   "Acción": "MANTENER + OHE",    "Justificación": f"χ²=106.39, 3 categorías"},
        {"Feature":      "VIP",           "Acción": "MANTENER",          "Justificación": "Señal débil pero significativa (p=0.002)"},
        {"Feature":      "Cabin",         "Acción": "DESCOMPONER",       "Justificación": "Extraer Deck, Num, Side (cardinalidad ~6500)"},
        {"Feature":      "TotalSpending", "Acción": "CREAR + log1p",     "Justificación": f"r_log={sp['r_log']:.3f}, mejor que r_raw={sp['r_raw']:.3f}"},
        {"Feature":      "GroupSize",     "Acción": "CREAR",             "Justificación": f"χ²={gs['chi2']:.2f}, asociación significativa"},
        {"Feature":      "Age",           "Acción": "MANTENER",          "Justificación": "Distribución normal, skew=0.42"},
        {"Feature":      "Name",          "Acción": "DESCARTAR",         "Justificación": "Identificador, sin señal predictiva directa"},
        {"Feature":      "PassengerId",   "Acción": "DESCARTAR (raw)",   "Justificación": "Usar solo para extraer GroupId y GroupSize"},
    ])
    md.add_table(decisions, index=False)
    md.add_text("")
    md.add_text(
        "**Próximo paso:** Implementar estas transformaciones en `src/features/engineering.py` "
        "y orquestarlas desde `notebooks/exploratory/03.feature_engineering.ipynb`."
    )

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    out_dir = REPORTS_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    md.save(str(out_dir / "01_eda.md"))
    html.save(str(out_dir / "01_eda.html"))

    write_data_quality_doc(df, TARGET, str(DOCS_DIR / "data" / "data_quality.md"))

    print(f"[OK] Reportes generados en {out_dir}")
