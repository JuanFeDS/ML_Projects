"""
Generación de reportes para el EDA.

Este módulo encapsula la creación de los reportes Markdown y HTML
para el análisis exploratorio de datos.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.config.settings import DOCS_DIR, REPORTS_DIR, TRAIN_RAW
from src.features.constants import TARGET, RAW_NUMERIC, RAW_CATEGORICAL, SPENDING_COLS
from src.reports.builder import HTMLReport, MarkdownReport
from src.reports.model_cards import write_data_quality_doc
from src.reports.eda_plots import (
    categorical_double_bar,
    correlation_heatmap,
    decisions_table,
    groupsize_bar,
    nulls_bar,
    numeric_boxplots,
    numeric_histograms,
    numeric_stats_table,
    numeric_vs_target_hist,
    target_pie,
    total_spending_compare,
    unique_values_bar,
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
    
    # ------------------------------------------------------------------
    # 1. Introducción y Métricas
    # ------------------------------------------------------------------
    html.add_intro(
        f"Este reporte consolida el análisis exploratorio del dataset "
        f"<b>Spaceship Titanic</b> ({basic['shape'][0]:,} registros, {basic['shape'][1]} variables). "
        f"Se examinan distribuciones, valores nulos y la relación con el target."
    )
    html.add_metrics_grid([
        (f"{basic['shape'][0]:,}", "registros"),
        (basic['shape'][1], "variables"),
        (len(basic['nulls']), "cols con nulos"),
        (basic['dupes'], "duplicados"),
        (f"{target['pcts'].get(True, 0):.1f}%", "tasa transported"),
    ])
    
    # ------------------------------------------------------------------
    # 2. Análisis Básico
    # ------------------------------------------------------------------
    md.add_section("Análisis Básico")
    html.add_section("Análisis Básico")
    md.add_table(basic['dtypes'], index=False)
    html.add_figure(unique_values_bar(basic['dtypes']), title="Tipos y Valores Únicos")
    
    md.add_subsection("Valores nulos")
    md.add_table(basic['nulls'], index=False)
    html.add_figure(nulls_bar(basic['nulls']), title="Distribución de Nulos")
    
    # ------------------------------------------------------------------
    # 3. Target
    # ------------------------------------------------------------------
    md.add_section("Balance del Target")
    html.add_section("Balance del Target")
    md.add_table(target['balance_df'], index=False)
    html.add_figure(target_pie(target['balance_df']), title="Proporción del Target")
    
    # ------------------------------------------------------------------
    # 4. Variables Numéricas
    # ------------------------------------------------------------------
    md.add_section("Variables Numéricas")
    html.add_section("Variables Numéricas")
    md.add_table(stats['numeric_desc'], index=True)
    html.add_figure(numeric_stats_table(stats['numeric_desc']), title="Estadísticas Descriptivas")
    html.add_figure(numeric_histograms(df, RAW_NUMERIC), title="Histogramas de Distribución")
    html.add_figure(numeric_boxplots(df, RAW_NUMERIC), title="Boxplots (Outliers)")
    
    # Correlaciones
    corr_df = df[RAW_NUMERIC + [TARGET]].copy()
    corr_df[TARGET] = corr_df[TARGET].astype(int)
    corr_matrix = corr_df.corr().round(3)
    html.add_figure(correlation_heatmap(corr_matrix), title="Matriz de Correlación")

    # ------------------------------------------------------------------
    # 5. Variables Categóricas vs Target
    # ------------------------------------------------------------------
    md.add_section("Variables Categóricas vs Target")
    html.add_section("Variables Categóricas vs Target")
    for col, res in stats['cat_stats'].items():
        md.add_subsection(col)
        md.add_metric("chi2", res["chi2"])
        md.add_metric("p-valor", f"{res['p']:.2e}")
        html.add_figure(categorical_double_bar(res["summary"], col, res["chi2"], res["p"], res["dof"]), title=col)

    # ------------------------------------------------------------------
    # 6. Features Derivadas
    # ------------------------------------------------------------------
    md.add_section("Análisis de Features Derivadas")
    html.add_section("Análisis de Features Derivadas")
    
    # Spending
    sp = derived["spending"]
    html.add_figure(total_spending_compare(
        sp["group_f_raw"], sp["group_t_raw"], sp["group_f_log"], sp["group_t_log"], sp["r_raw"], sp["r_log"]
    ), title="Impacto de TotalSpending (Raw vs Log)")
    
    # GroupSize
    gs = derived["groupsize"]
    html.add_figure(groupsize_bar(gs["summary"], gs["chi2"], gs["p"]), title="Influencia del Tamaño de Grupo")

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md.save(str(REPORTS_DIR / "01_eda.md"))
    html.save(str(REPORTS_DIR / "01_eda.html"))
    
    # Documento de calidad de datos
    write_data_quality_doc(df, TARGET, str(DOCS_DIR / "data" / "data_quality.md"))
    
    print(f"✅ Reportes generados en {REPORTS_DIR}")
