"""ReportFactory y helpers de contexto para reportes de entrenamiento.

Separado de builder.py para evitar el ciclo de importacion:
  builder.py ← training/reports.py ← factory.py (unidireccional).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.reports.training.reports import TrainingResults, build_training_html, build_training_md


def _top_feature_names_for_insights(
    model: object, feature_names: List[str], k: int = 5
) -> List[str]:
    """Nombres de las k features con mayor importancia si el estimador las expone."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        order = np.argsort(imp)[::-1][:k]
        return [feature_names[i] for i in order]
    return []


def build_training_insights_context(results: Dict[str, Any]) -> Dict[str, Any]:
    """Construye el dict esperado por get_training_insights desde TrainingPipeline.run().

    Args:
        results: Salida de src.pipelines.training_pipeline.TrainingPipeline.run().

    Returns:
        Dict con métricas y metadatos para el reporte de insights.
    """
    cv_df = results["cv_results"]
    best_name = results["best_name"]
    cv_list = [
        {"model": str(idx), "cv_accuracy": float(row["cv_accuracy_mean"])}
        for idx, row in cv_df.iterrows()
    ]
    winner_val = results["winner_val"]
    tuned_val = results["tuned_val"]
    stacking_val = results["stacking_val"]
    moe_val = results["moe_val"]
    threshold_gain = float(results["threshold_acc"] - winner_val["val_accuracy"])
    return {
        "fs_name": results["fs_name"],
        "cv_results": cv_list,
        "best_model": best_name,
        "best_cv_accuracy": float(cv_df.loc[best_name, "cv_accuracy_mean"]),
        "tuned_val_accuracy": float(tuned_val["val_accuracy"]),
        "tuned_val_roc_auc": float(tuned_val["val_roc_auc"]),
        "stacking_val_accuracy": float(stacking_val["val_accuracy"]),
        "moe_val_accuracy": float(moe_val["val_accuracy"]),
        "winner_name": results["winner_name"],
        "winner_val_accuracy": float(winner_val["val_accuracy"]),
        "winner_val_roc_auc": float(winner_val["val_roc_auc"]),
        "best_threshold": float(results["best_threshold"]),
        "threshold_gain": threshold_gain,
        "top_features": _top_feature_names_for_insights(
            results["winner_model"], results["feature_names"], k=5
        ),
    }


class ReportFactory:  # pylint: disable=too-few-public-methods
    """Genera parejas MD/HTML de reportes operacionales a partir de resultados de pipeline."""

    @staticmethod
    def emit_training_reports(results: Dict[str, Any]) -> None:
        """Escribe reports/03_training.md y reports/03_training.html.

        Args:
            results: Salida de src.pipelines.training_pipeline.TrainingPipeline.run().
        """
        tr = TrainingResults(
            cv_results=results["cv_results"],
            best_name=results["best_name"],
            best_params=results["best_params"],
            tuned_val=results["tuned_val"],
            stacking_val=results["stacking_val"],
            moe_val=results["moe_val"],
            winner_name=results["winner_name"],
            winner_val=results["winner_val"],
            top_names=results["top_names"],
            fs_name=results["fs_name"],
            error_tables=results["error_tables"],
            best_threshold=results["best_threshold"],
            threshold_acc=results["threshold_acc"],
            exp_id=results["metadata"]["exp_id"],
            winner_model=results["winner_model"],
            feature_names=results["feature_names"],
            shap_plots=results.get("shap_plots", {}),
        )
        build_training_md(tr)
        build_training_html(tr)
