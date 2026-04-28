"""Orquestador del reporte HTML de seguimiento de experimentos.

Coordina la extracción de datos (tracking_loader), las visualizaciones (charts)
y el renderizado de tarjetas (cards_renderer) para producir el HTML final.
"""

from __future__ import annotations

from src.models.tracking import load_experiment_data
from src.reports.builder import HTMLReport
from src.reports.experiments.cards_renderer import _CARD_CSS, render_experiment_card
from src.reports.experiments.charts import (
    plot_accuracy_progression,
    plot_cv_vs_val,
    plot_leaderboard,
    plot_optuna_trials,
    plot_submission_rates,
)
from src.reports.experiments.descriptions import EXPERIMENT_DESCRIPTIONS


def build_tracking_report(
    tracking_uri: str,
    output_path: str,
    experiment_name: str = "spacechip_titanic",
) -> None:
    """Genera el reporte HTML de seguimiento de experimentos.

    Args:
        tracking_uri: URI de MLflow (ej. 'sqlite:///mlflow.db').
        output_path: Ruta del archivo HTML de salida.
        experiment_name: Nombre del experimento en MLflow.
    """
    print("[...] Consultando MLflow...")
    data = load_experiment_data(tracking_uri, experiment_name)
    df = data["training_df"]
    trials_df = data["trials_df"]
    pred_df = data["predictions_df"]
    summary = data["summary"]

    if df.empty:
        print("[WARN] No se encontraron runs de entrenamiento en MLflow.")
        return

    print(
        f"[OK] {summary['total_experiments']} experimentos, "
        f"{summary['total_trials']} trials Optuna, "
        f"{summary['total_predictions']} predicciones."
    )

    html = HTMLReport(title="Seguimiento de Experimentos — Spaceship Titanic")

    html.add_intro(
        f"Reporte de <b>{summary['total_experiments']} feature sets</b> evaluados "
        f"({summary['total_runs']} runs totales, {summary['total_trials']} trials Optuna). "
        f"Mejor resultado: <b>{summary['best_val_accuracy']:.4f} val_accuracy</b> "
        f"con <code>{summary['best_feature_set']}</code>."
    )
    html.add_metrics_grid(
        [
            (summary["total_experiments"], "Feature Sets"),
            (summary["total_runs"], "Runs totales"),
            (summary["total_trials"], "Trials Optuna"),
            (f"{summary['best_val_accuracy']:.4f}", "Mejor Val Acc"),
            (summary["best_feature_set"].split("_")[0], "Mejor FS"),
            (summary["total_predictions"], "Submissions"),
        ]
    )

    html.add_section("Leaderboard de Experimentos")
    html.add_text(
        "Tabla ordenada por <b>val_accuracy descendente</b>. "
        "El color de la columna Feature Set va de rojo (peor) a verde (mejor). "
        "Se muestra el mejor run por feature set."
    )
    if summary.get("n_anomalies", 0) > 0:
        anomaly_fs = df.loc[df["anomaly"], "feature_set"].tolist()
        html.add_callout(
            f"<b>Anomalia detectada ({summary['n_anomalies']} feature set(s)):</b> "
            f"<code>{', '.join(anomaly_fs)}</code> muestra una val_accuracy muy superior "
            "al resto del grupo (&gt;8 puntos sobre la mediana). Esto indica probable "
            "<b>fuga de datos</b> — las reglas de dominio codificaban el target de forma directa. "
            "La submission correspondiente confirma el problema: tasa de transporte anomalamente baja. "
            "Se excluye de las metricas de referencia del reporte.",
            kind="warning",
        )
    html.add_figure(plot_leaderboard(df), title="Ranking de Feature Sets")

    html.add_section("Progresion de Accuracy")
    html.add_text(
        "Evolucion del val_accuracy y cv_accuracy a lo largo de los experimentos, "
        "ordenados por numero de feature set. La linea discontinua marca el mejor resultado alcanzado."
    )
    html.add_figure(
        plot_accuracy_progression(df),
        title="Val Accuracy y CV Accuracy por Feature Set",
    )

    html.add_section("Diagnostico de Sobreajuste")
    html.add_text(
        "Puntos por encima de la diagonal (val > cv) pueden indicar split favorecido. "
        "Puntos muy por debajo indican sobreajuste. Un buen modelo se mantiene cerca de la diagonal."
    )
    html.add_figure(plot_cv_vs_val(df), title="CV Accuracy vs Val Accuracy")

    html.add_section("Busqueda de Hiperparametros (Optuna)")
    html.add_text(
        f"Distribucion de <b>trial_accuracy</b> en los {summary['total_trials']} trials de busqueda "
        "Bayesiana. Cada caja representa un experimento con tuning. "
        "Una distribucion estrecha y alta indica que Optuna convergio bien."
    )
    html.add_figure(
        plot_optuna_trials(trials_df, data["all_training_df"]),
        title="Distribucion de trial_accuracy por experimento",
    )

    html.add_section("Tasas de Prediccion (Submissions)")
    html.add_text(
        "Porcentaje de pasajeros predichos como <i>Transported=True</i> en cada submission. "
        "El dataset real tiene ~50% de balance — desviaciones grandes pueden indicar "
        "calibracion incorrecta del modelo."
    )
    html.add_figure(
        plot_submission_rates(pred_df), title="% Transported predicho por experimento"
    )

    html.add_section("Detalle por Experimento")
    html.add_text(
        "Tarjeta narrativa de cada feature set: que se hizo, hipotesis, resultado, "
        "hiperparametros encontrados por Optuna y curva de convergencia."
    )
    html.add_html(_CARD_CSS)

    baseline_row = df[df["feature_set"] == "fs-001_baseline"]
    baseline_val = (
        float(baseline_row["val_accuracy"].iloc[0])
        if not baseline_row.empty
        else 0.8285
    )

    for _, row in df.sort_values("fs_num").iterrows():
        fs = row["feature_set"]
        desc = EXPERIMENT_DESCRIPTIONS.get(
            fs,
            {
                "what": "Descripcion pendiente.",
                "hypothesis": "Pendiente.",
                "result": "Pendiente.",
                "tags": [],
            },
        )
        render_experiment_card(html, row, trials_df, baseline_val, desc)

    html.save(output_path)
    print(f"[OK] Reporte generado en {output_path}")
