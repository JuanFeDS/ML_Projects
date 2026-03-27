"""
Script de Analisis Exploratorio de Datos — Spaceship Titanic.

Replica la logica de NB01 (exploracion inicial) y NB02 (analisis vs target).
Genera Reports/01_eda.md y Reports/01_eda.html.

Ejecutar desde la raiz del proyecto:
    python scripts/01_eda.py
"""
import sys
import warnings

sys.path.insert(0, ".")  # scripts run from project root
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
# pylint: disable=wrong-import-position

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import DOCS_DIR, REPORTS_DIR, TRAIN_RAW
from src.features.constants import (
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    SPENDING_COLS,
    TARGET,
)
from src.reports.builder import HTMLReport, MarkdownReport, write_data_quality_doc
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
    spending_categories_bar,
    target_pie,
    total_spending_compare,
    unique_values_bar,
)


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total * 100:.2f}%)"


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Ejecuta el pipeline completo de EDA y genera los reportes."""
    print("=" * 60)
    print("📊 01_eda.py — Analisis Exploratorio de Datos")
    print("=" * 60)

    df = pd.read_csv(TRAIN_RAW)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    md = MarkdownReport(title="EDA — Spaceship Titanic")
    html = HTMLReport(title="EDA — Spaceship Titanic")

    # Calculos previos para el resumen de cabecera
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    nulls_df = (
        pd.DataFrame(
            {
                "Columna": null_counts.index,
                "Nulos": null_counts.values,
                "% Nulos": null_pct.values,
            }
        )
        .query("Nulos > 0")
        .reset_index(drop=True)
    )
    n_dupes = int(df.duplicated().sum())
    n_cols_nulls = len(nulls_df)
    transported_pct = df[TARGET].mean() * 100

    html.add_intro(
        f"Este reporte consolida el analisis exploratorio del dataset "
        f"<b>Spaceship Titanic</b> ({df.shape[0]:,} registros, {df.shape[1]} variables). "
        f"Se examinan distribuciones, valores nulos, la relacion de cada variable con el "
        f"target <code>Transported</code> mediante tests estadisticos (chi-square para "
        f"categoricas, Mann-Whitney U para numericas) y se validan features derivadas "
        f"que mejoran la separabilidad. Las decisiones documentadas al final alimentan "
        f"directamente el pipeline de feature engineering (<code>02_features.py</code>)."
    )
    html.add_metrics_grid([
        (f"{df.shape[0]:,}", "registros"),
        (df.shape[1], "variables"),
        (n_cols_nulls, "cols con nulos"),
        (n_dupes, "duplicados exactos"),
        (f"{transported_pct:.1f}%", "tasa transported"),
    ])

    # ------------------------------------------------------------------
    # 1. Analisis basico
    # ------------------------------------------------------------------
    print("\n🔍 [1] Analisis basico...")
    md.add_section("Analisis Basico del Dataset")
    html.add_section("Analisis Basico del Dataset")

    md.add_metric("Filas", f"{df.shape[0]:,}")
    md.add_metric("Columnas", df.shape[1])
    html.add_text(
        f"<b>Dimensiones:</b> {df.shape[0]:,} filas x {df.shape[1]} columnas"
    )

    dtypes_df = pd.DataFrame(
        {
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "Valores unicos": [df[c].nunique() for c in df.columns],
        }
    )
    md.add_subsection("Tipos de datos")
    md.add_table(dtypes_df, index=False)
    html.add_section("Tipos de datos", level=3)
    html.add_figure(unique_values_bar(dtypes_df), title="")

    print(f"  ⚠️  Columnas con nulos: {n_cols_nulls}")
    md.add_subsection("Valores nulos")
    md.add_table(nulls_df, index=False)
    html.add_section("Valores Nulos", level=3)
    html.add_figure(nulls_bar(nulls_df), title="")

    md.add_metric("Duplicados exactos", _pct(n_dupes, len(df)))
    html.add_text(f"<b>Duplicados exactos:</b> {_pct(n_dupes, len(df))}")

    # ------------------------------------------------------------------
    # 2. Balance del target
    # ------------------------------------------------------------------
    print("\n🎯 [2] Balance del target...")
    md.add_section("Balance del Target (Transported)")
    html.add_section("Balance del Target (Transported)")

    target_counts = df[TARGET].value_counts()
    target_pct = df[TARGET].value_counts(normalize=True) * 100
    balance_df = pd.DataFrame(
        {
            "Clase": target_counts.index.astype(str),
            "Conteo": target_counts.values,
            "% del total": target_pct.values.round(2),
        }
    )
    md.add_table(balance_df, index=False)
    html.add_figure(target_pie(balance_df), title="Distribucion del Target")
    true_n = target_counts.get(True, 0)
    false_n = target_counts.get(False, 0)
    print(f"  True: {true_n:,}  False: {false_n:,}")
    true_pct = target_pct.get(True, 0)
    false_pct = target_pct.get(False, 0)
    html.add_callout(
        f"El dataset esta <b>practicamente balanceado</b>: "
        f"{true_n:,} pasajeros transportados ({true_pct:.1f}%) vs "
        f"{false_n:,} no transportados ({false_pct:.1f}%). "
        f"No se requieren tecnicas de rebalanceo (SMOTE, class_weight).",
        kind="success",
    )

    # ------------------------------------------------------------------
    # 3. Variables numericas
    # ------------------------------------------------------------------
    print("\n🔢 [3] Variables numericas...")
    md.add_section("Variables Numericas")
    html.add_section("Variables Numericas")

    desc = df[RAW_NUMERIC].describe().T.round(2)
    md.add_subsection("Estadisticas descriptivas")
    md.add_table(desc, index=True)
    html.add_section("Estadisticas Descriptivas", level=3)
    html.add_figure(numeric_stats_table(desc), title="")

    html.add_section("Histogramas", level=3)
    html.add_figure(numeric_histograms(df, RAW_NUMERIC), title="")

    html.add_section("Boxplots", level=3)
    html.add_figure(numeric_boxplots(df, RAW_NUMERIC), title="")

    corr_df = df[RAW_NUMERIC + [TARGET]].copy()
    corr_df[TARGET] = corr_df[TARGET].astype(int)
    corr_matrix = corr_df.corr().round(3)

    corr_with_target = (
        corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    )
    corr_target_df = pd.DataFrame(
        {"Feature": corr_with_target.index, "Pearson r": corr_with_target.values}
    )
    md.add_subsection("Correlaciones con el target")
    md.add_table(corr_target_df, index=False)
    html.add_section("Matriz de Correlacion", level=3)
    html.add_figure(correlation_heatmap(corr_matrix), title="")

    # ------------------------------------------------------------------
    # 4. Variables categoricas
    # ------------------------------------------------------------------
    print("\n🏷️  [4] Variables categoricas...")
    md.add_section("Variables Categoricas")
    html.add_section("Variables Categoricas")

    for col in RAW_CATEGORICAL:
        print(f"  {col}...")
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = [col, "count"]
        freq["pct"] = (freq["count"] / len(df) * 100).round(2)

        target_rate = (
            df.groupby(col, dropna=False)[TARGET]
            .mean()
            .reset_index()
            .rename(columns={TARGET: "tasa_transported"})
        )
        target_rate["tasa_transported"] = target_rate["tasa_transported"].round(4)
        summary = freq.merge(target_rate, on=col, how="left")

        contingency = pd.crosstab(df[col].fillna("NaN"), df[TARGET])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

        md.add_subsection(col)
        md.add_metric("chi2", round(chi2, 2))
        md.add_metric("p-valor", f"{p_val:.2e}")
        md.add_metric("Grados de libertad", dof)
        md.add_table(summary, index=False)

        html.add_section(col, level=3)
        html.add_text(f"chi2={chi2:.2f} | p-valor={p_val:.2e} | df={dof}")
        html.add_figure(
            categorical_double_bar(summary, col, chi2, p_val, dof), title=""
        )

    # ------------------------------------------------------------------
    # 5. Variables numericas vs target (Mann-Whitney + Pearson)
    # ------------------------------------------------------------------
    print("\n📈 [5] Variables numericas vs target...")
    md.add_section("Variables Numericas vs Target")
    html.add_section("Variables Numericas vs Target")

    num_rows = []
    for col in RAW_NUMERIC:
        print(f"  {col}...")
        group_true = df.loc[df[TARGET], col].dropna()
        group_false = df.loc[~df[TARGET], col].dropna()

        stat_mw, p_mw = stats.mannwhitneyu(
            group_true, group_false, alternative="two-sided"
        )
        mask = df[col].notna() & df[TARGET].notna()
        r_p, _ = stats.pearsonr(
            df.loc[mask, col], df.loc[mask, TARGET].astype(int)
        )

        num_rows.append(
            {
                "Feature": col,
                "MannWhitney stat": round(stat_mw, 0),
                "MW p-valor": f"{p_mw:.2e}",
                "Pearson r": round(r_p, 4),
                "Media (True)": round(group_true.mean(), 3),
                "Media (False)": round(group_false.mean(), 3),
            }
        )

        html.add_section(col, level=3)
        html.add_figure(
            numeric_vs_target_hist(group_false, group_true, col, p_mw, r_p),
            title="",
        )

    md.add_table(pd.DataFrame(num_rows), index=False)

    # ------------------------------------------------------------------
    # 6. Features derivadas
    # ------------------------------------------------------------------
    print("\n⚙️  [6] Features derivadas...")
    md.add_section("Features Derivadas")
    html.add_section("Features Derivadas")

    df["TotalSpending"] = df[SPENDING_COLS].fillna(0).sum(axis=1)
    df["TotalSpending_Log"] = np.log1p(df["TotalSpending"])
    df["TravelGroup"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("TravelGroup")["TravelGroup"].transform("count")
    df["SpendingCategories"] = (df[SPENDING_COLS].fillna(0) > 0).sum(axis=1)

    mask = df["TotalSpending"].notna() & df[TARGET].notna()
    r_raw, _ = stats.pearsonr(
        df.loc[mask, "TotalSpending"], df.loc[mask, TARGET].astype(int)
    )
    r_log, _ = stats.pearsonr(
        df.loc[mask, "TotalSpending_Log"], df.loc[mask, TARGET].astype(int)
    )

    md.add_subsection("TotalSpending vs TotalSpending_Log")
    md.add_metric("r(TotalSpending, Target)", round(r_raw, 4))
    md.add_metric("r(TotalSpending_Log, Target)", round(r_log, 4))
    md.add_text(
        f"La transformacion log mejora la correlacion absoluta de "
        f"{abs(r_raw):.3f} a {abs(r_log):.3f}."
    )

    group_f_raw = df.loc[~df[TARGET], "TotalSpending"]
    group_t_raw = df.loc[df[TARGET], "TotalSpending"]
    group_f_log = df.loc[~df[TARGET], "TotalSpending_Log"]
    group_t_log = df.loc[df[TARGET], "TotalSpending_Log"]

    html.add_section("TotalSpending vs TotalSpending_Log", level=3)
    html.add_figure(
        total_spending_compare(
            group_f_raw, group_t_raw, group_f_log, group_t_log, r_raw, r_log
        ),
        title="",
    )

    contingency_gs = pd.crosstab(df["GroupSize"], df[TARGET])
    chi2_gs, p_gs, _, _ = stats.chi2_contingency(contingency_gs)
    target_by_gs = (
        df.groupby("GroupSize")[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "tasa_transported", "count": "n"})
    )
    md.add_subsection("GroupSize")
    md.add_metric("chi2", round(chi2_gs, 2))
    md.add_metric("p-valor", f"{p_gs:.2e}")
    md.add_table(target_by_gs, index=False)
    html.add_section("GroupSize", level=3)
    html.add_figure(groupsize_bar(target_by_gs, chi2_gs, p_gs), title="")

    contingency_sc = pd.crosstab(df["SpendingCategories"], df[TARGET])
    chi2_sc, p_sc, _, _ = stats.chi2_contingency(contingency_sc)
    target_by_sc = (
        df.groupby("SpendingCategories")[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "tasa_transported", "count": "n"})
    )
    md.add_subsection("SpendingCategories")
    md.add_metric("chi2", round(chi2_sc, 2))
    md.add_metric("p-valor", f"{p_sc:.2e}")
    md.add_table(target_by_sc, index=False)
    html.add_section("SpendingCategories", level=3)
    html.add_figure(
        spending_categories_bar(target_by_sc, chi2_sc, p_sc), title=""
    )

    # ------------------------------------------------------------------
    # 7. Tabla de decisiones
    # ------------------------------------------------------------------
    print("\n📋 [7] Tabla de decisiones...")
    md.add_section("Tabla de Decisiones")
    html.add_section("Tabla de Decisiones")

    decisions = pd.DataFrame(
        [
            {
                "Feature": "Age",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "MW p<0.001; categorizar en AgeCategory",
            },
            {
                "Feature": "RoomService",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "Correlacion con target via TotalSpending_Log",
            },
            {
                "Feature": "FoodCourt",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "Idem RoomService; parte de TotalSpending",
            },
            {
                "Feature": "ShoppingMall",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "Idem",
            },
            {
                "Feature": "Spa",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "Idem",
            },
            {
                "Feature": "VRDeck",
                "Accion": "MANTENER",
                "Tipo": "Numerica",
                "Justificacion": "Idem",
            },
            {
                "Feature": "TotalSpending_Log",
                "Accion": "MANTENER (derivada)",
                "Tipo": "Numerica",
                "Justificacion": f"r={round(r_log, 3)} vs r={round(r_raw, 3)} crudo",
            },
            {
                "Feature": "TotalSpending",
                "Accion": "DESCARTAR",
                "Tipo": "Numerica",
                "Justificacion": "Reemplazada por TotalSpending_Log",
            },
            {
                "Feature": "GroupSize",
                "Accion": "MANTENER (derivada)",
                "Tipo": "Numerica/Cat",
                "Justificacion": f"chi2={round(chi2_gs, 1)}, p<0.001",
            },
            {
                "Feature": "SpendingCategories",
                "Accion": "MANTENER (derivada)",
                "Tipo": "Numerica",
                "Justificacion": f"chi2={round(chi2_sc, 1)}, p<0.001",
            },
            {
                "Feature": "Cabin",
                "Accion": "TRANSFORMAR",
                "Tipo": "Categorica",
                "Justificacion": "Extraer Deck (chi2~392), CabinNumber, Side",
            },
            {
                "Feature": "Deck",
                "Accion": "MANTENER (derivada)",
                "Tipo": "Categorica",
                "Justificacion": "Alta discriminacion, chi2>392",
            },
            {
                "Feature": "Side",
                "Accion": "DESCARTAR",
                "Tipo": "Categorica",
                "Justificacion": "Senal cubierta por Deck",
            },
            {
                "Feature": "CabinNumber",
                "Accion": "MANTENER (derivada)",
                "Tipo": "Numerica",
                "Justificacion": "Senal de posicion a bordo",
            },
            {
                "Feature": "HomePlanet",
                "Accion": "MANTENER",
                "Tipo": "Categorica",
                "Justificacion": "chi2 significativo",
            },
            {
                "Feature": "Destination",
                "Accion": "MANTENER",
                "Tipo": "Categorica",
                "Justificacion": "chi2 significativo",
            },
            {
                "Feature": "CryoSleep",
                "Accion": "MANTENER (encoded)",
                "Tipo": "Binaria",
                "Justificacion": "Alta correlacion con no-gasto; label encode",
            },
            {
                "Feature": "VIP",
                "Accion": "DESCARTAR",
                "Tipo": "Binaria",
                "Justificacion": "Senal debil (corr~-0.037)",
            },
            {
                "Feature": "Name",
                "Accion": "DESCARTAR",
                "Tipo": "Texto",
                "Justificacion": "Sin poder predictivo directo",
            },
            {
                "Feature": "PassengerId",
                "Accion": "DESCARTAR",
                "Tipo": "ID",
                "Justificacion": "Solo ID; info util ya extraida en GroupSize",
            },
        ]
    )
    md.add_table(decisions, index=False)
    html.add_figure(
        decisions_table(decisions),
        title="Tabla de decisiones — que mantener, descartar o transformar",
    )

    # ------------------------------------------------------------------
    # 8. Guardar reportes + documentacion
    # ------------------------------------------------------------------
    print("\n💾 [8] Guardando reportes...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md.save(str(REPORTS_DIR / "01_eda.md"))
    html.save(str(REPORTS_DIR / "01_eda.html"))
    write_data_quality_doc(
        df=pd.read_csv(TRAIN_RAW),
        target_col=TARGET,
        path=str(DOCS_DIR / "data" / "data_quality.md"),
    )
    print("\n✅ EDA completado.")


if __name__ == "__main__":
    main()
