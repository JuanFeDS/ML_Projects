"""Tests del contexto de IA para reportes de entrenamiento."""
import pandas as pd

from src.reports.builder import build_training_insights_context


def _minimal_training_results():
    cv = pd.DataFrame(
        {
            "cv_accuracy_mean": [0.5, 0.82, 0.85],
            "cv_accuracy_std": [0.0, 0.02, 0.01],
        },
        index=["Baseline", "XGB", "CatBoost"],
    )
    val_common = {
        "val_accuracy": 0.81,
        "val_roc_auc": 0.88,
        "y_pred": [0, 1],
        "y_proba": [0.4, 0.6],
        "classification_report": "ok",
    }
    return {
        "fs_name": "fs-test",
        "cv_results": cv,
        "best_name": "CatBoost",
        "best_params": {"depth": 6},
        "tuned_val": dict(val_common),
        "stacking_val": dict(val_common),
        "moe_val": dict(val_common),
        "winner_name": "CatBoost (tuneado)",
        "winner_val": dict(val_common),
        "winner_model": object(),
        "feature_names": ["a", "b"],
        "error_tables": {},
        "best_threshold": 0.45,
        "threshold_acc": 0.83,
        "top_names": ["CatBoost", "XGB", "LGBM"],
        "metadata": {},
        "promoted": False,
    }


def test_build_training_insights_context_keys():
    results = _minimal_training_results()
    ctx = build_training_insights_context(results)
    assert ctx["fs_name"] == "fs-test"
    assert ctx["best_model"] == "CatBoost"
    assert "cv_results" in ctx
    assert len(ctx["cv_results"]) == 3
    assert ctx["winner_name"] == "CatBoost (tuneado)"
