"""
Generación de reportes de entrenamiento para el pipeline Spaceship Titanic.

Funciones para construir reports/03_training.md y reports/03_training.html
a partir de los resultados del pipeline de entrenamiento.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.config.settings import REPORTS_DIR
from src.reports.builder import HTMLReport, MarkdownReport
from src.reports.training.plots import (
    cv_accuracy_bar,
    feature_importance_bar,
    validation_metrics_bar,
)


@dataclass
class TrainingResults:
    """Resultados del pipeline de entrenamiento para generacion de reportes."""

    cv_results: pd.DataFrame
    best_name: str
    best_params: dict
    tuned_val: dict
    stacking_val: dict
    moe_val: dict
    winner_name: str
    winner_val: dict
    top_names: List[str]
    fs_name: str
    error_tables: Dict[str, Any]
    best_threshold: float
    threshold_acc: float
    exp_id: str = "000"
    winner_model: Any = None
    feature_names: List[str] = field(default_factory=list)
    shap_plots: Dict[str, Any] = field(default_factory=dict)


def build_training_md(results: TrainingResults) -> None:
    """Genera reports/03_training.md.

    Args:
        results: Resultados del pipeline de entrenamiento.
    """
    md = MarkdownReport("Reporte de Entrenamiento — Spaceship Titanic")

    md.add_section("Feature Set")
    md.add_metric("Nombre", results.fs_name)

    md.add_section("Resultados Cross-Validation (todos los modelos)")
    md.add_text(
        "Evaluacion con StratifiedKFold (5 folds). Ordenado por cv_accuracy_mean."
    )
    cv_display = results.cv_results.reset_index().rename(columns={"index": "Modelo"})
    md.add_table(cv_display, index=False)

    md.add_section(f"Mejor Modelo Seleccionado: {results.best_name}")
    md.add_text(
        "El modelo con mayor cv_accuracy_mean fue tuneado con "
        "Optuna TPE (n_iter=25)."
    )
    md.add_subsection("Mejores hiperparametros encontrados")
    params_str = "\n".join(f"{k}: {v}" for k, v in results.best_params.items())
    md.add_code(params_str, lang="")

    md.add_section("Evaluacion en Validacion")
    val_rows = [
        {
            "Modelo": f"{results.best_name}",
            "val_accuracy": results.tuned_val["val_accuracy"],
            "val_roc_auc": results.tuned_val["val_roc_auc"],
        }
    ]
    if results.stacking_val:
        val_rows.append(
            {
                "Modelo": "Stacking",
                "val_accuracy": results.stacking_val["val_accuracy"],
                "val_roc_auc": results.stacking_val["val_roc_auc"],
            }
        )
    if results.moe_val:
        val_rows.append(
            {
                "Modelo": "MoE (CatBoost x segmento)",
                "val_accuracy": results.moe_val["val_accuracy"],
                "val_roc_auc": results.moe_val["val_roc_auc"],
            }
        )
    md.add_table(pd.DataFrame(val_rows), index=False)

    md.add_subsection(f"Classification Report — {results.best_name}")
    md.add_code(results.tuned_val["classification_report"], lang="")

    if results.stacking_val:
        md.add_subsection("Classification Report — Stacking")
        md.add_code(results.stacking_val["classification_report"], lang="")

    md.add_section("Modelo Ganador Final")
    md.add_metric("Modelo", results.winner_name)
    md.add_metric("val_accuracy", results.winner_val["val_accuracy"])
    md.add_metric("val_roc_auc", results.winner_val["val_roc_auc"])
    md.add_text(
        "El modelo ganador fue re-entrenado sobre el conjunto completo de train "
        "y guardado en `models/production/best_model.pkl`."
    )

    md.add_section("Error Analysis — Tasa de error por segmento")
    md.add_text(
        "Porcentaje de errores del modelo ganador en el conjunto de validacion, "
        "desglosado por variable."
    )
    for seg, tbl in results.error_tables.items():
        md.add_subsection(seg)
        md.add_table(tbl, index=False)

    md.add_section("Threshold Optimization")
    md.add_metric("Umbral optimo", results.best_threshold)
    md.add_metric("val_accuracy con umbral optimo", results.threshold_acc)
    md.add_metric("val_accuracy con umbral 0.50", results.winner_val["val_accuracy"])
    md.add_metric(
        "Ganancia",
        round(results.threshold_acc - results.winner_val["val_accuracy"], 4),
    )

    out_dir = (
        REPORTS_DIR
        / "training"
        / f"exp-{results.exp_id}_{results.winner_name.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    md.save(str(out_dir / "03_training.md"))


def build_training_html(results: TrainingResults) -> None:
    """Genera reports/03_training.html.

    Args:
        results: Resultados del pipeline de entrenamiento.
    """
    html = HTMLReport("Reporte de Entrenamiento — Spaceship Titanic")

    all_accs = [results.tuned_val["val_accuracy"]]
    all_rocs = [results.tuned_val["val_roc_auc"]]
    if results.stacking_val:
        all_accs.append(results.stacking_val["val_accuracy"])
        all_rocs.append(results.stacking_val["val_roc_auc"])
    if results.moe_val:
        all_accs.append(results.moe_val["val_accuracy"])
        all_rocs.append(results.moe_val["val_roc_auc"])

    winner_acc = max(all_accs)
    winner_roc = max(all_rocs)
    n_models_evaluated = len(results.cv_results)

    html.add_intro(
        f"Se evaluaron <b>{n_models_evaluated} modelos</b> mediante StratifiedKFold (5 folds). "
        f"El modelo ganador alcanza una precision de <b>{winner_acc:.1%}</b> y un "
        f"ROC-AUC de <b>{winner_roc:.4f}</b> en el conjunto de validacion (hold-out 20%)."
    )
    metrics_grid = [
        (f"{winner_acc:.1%}", "val accuracy"),
        (f"{winner_roc:.4f}", "val ROC-AUC"),
        (f"{results.tuned_val['val_accuracy']:.1%}", results.best_name),
        (n_models_evaluated, "modelos evaluados"),
    ]
    if results.stacking_val:
        metrics_grid.append((f"{results.stacking_val['val_accuracy']:.1%}", "stacking"))
    if results.moe_val:
        metrics_grid.append((f"{results.moe_val['val_accuracy']:.1%}", "MoE"))
    html.add_metrics_grid(metrics_grid)

    html.add_section("Cross-Validation Accuracy (todos los modelos)")
    html.add_text(
        "Comparacion de todos los clasificadores con StratifiedKFold 5-fold sobre el "
        "conjunto de entrenamiento (80% del total). El Baseline (DummyClassifier) sirve "
        "como piso minimo de referencia."
    )
    html.add_figure(
        cv_accuracy_bar(results.cv_results), title="CV Accuracy — todos los modelos"
    )

    html.add_section("Validacion: comparacion de candidatos")
    val_models = [results.best_name]
    val_acc = [results.tuned_val["val_accuracy"]]
    val_roc = [results.tuned_val["val_roc_auc"]]
    if results.stacking_val:
        val_models.append("Stacking")
        val_acc.append(results.stacking_val["val_accuracy"])
        val_roc.append(results.stacking_val["val_roc_auc"])
    if results.moe_val:
        val_models.append("MoE")
        val_acc.append(results.moe_val["val_accuracy"])
        val_roc.append(results.moe_val["val_roc_auc"])
    html.add_figure(
        validation_metrics_bar(val_models, val_acc, val_roc),
        title="Val Accuracy y ROC-AUC",
    )

    fi_fig = feature_importance_bar(results.winner_model, results.feature_names)
    if fi_fig is not None:
        html.add_section("Feature Importance (top 20)")
        html.add_text(
            "Importancia de features del modelo ganador. Permite verificar coherencia "
            "con el analisis estadistico realizado en el EDA."
        )
        html.add_figure(fi_fig, title="Feature Importance")

    html.add_section("Error Analysis — Tasa de error por segmento")
    html.add_text(
        "Porcentaje de predicciones incorrectas del modelo ganador desglosado por "
        "variable. Permite identificar que segmentos son mas dificiles de clasificar."
    )
    for seg, tbl in results.error_tables.items():
        html.add_text(f"<b>{seg}</b>")
        html.add_text(tbl.to_html(index=False, border=0))

    html.add_section("Threshold Optimization")
    threshold_gain = round(results.threshold_acc - winner_acc, 4)
    html.add_text(
        f"Busqueda de umbral optimo en [0.30, 0.70] (200 puntos). "
        f"<b>Umbral optimo: {results.best_threshold:.4f}</b> → val_accuracy: "
        f"<b>{results.threshold_acc:.4f}</b> (vs {winner_acc:.4f} con 0.50, "
        f"ganancia: {threshold_gain:+.4f})."
    )

    if results.shap_plots:
        html.add_section("Analisis SHAP")
        html.add_text(
            "Los SHAP values (SHapley Additive exPlanations) miden la contribucion "
            "de cada feature a cada prediccion individual, con signo y magnitud."
        )
        if results.shap_plots.get("summary_bar"):
            html.add_image(
                results.shap_plots["summary_bar"],
                title="Importancia global (media |SHAP|)",
            )
        if results.shap_plots.get("beeswarm"):
            html.add_image(
                results.shap_plots["beeswarm"],
                title="Distribucion del impacto por feature",
            )
        if results.shap_plots.get("waterfall_comparison"):
            html.add_image(
                results.shap_plots["waterfall_comparison"],
                title="Waterfall — peor vs mejor prediccion",
            )

    out_dir = (
        REPORTS_DIR
        / "training"
        / f"exp-{results.exp_id}_{results.winner_name.replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    html.save(str(out_dir / "03_training.html"))
