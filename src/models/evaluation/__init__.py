"""Subpaquete de evaluacion — re-exporta la interfaz publica."""

from src.models.evaluation.errors import analyze_errors
from src.models.evaluation.evaluation import (
    compute_oof_metrics,
    evaluate_models,
    evaluate_on_validation,
    optimize_threshold,
)
from src.models.evaluation.stacking_oof import oof_027, oof_033, oof_tabpfn

__all__ = [
    "analyze_errors",
    "compute_oof_metrics",
    "evaluate_models",
    "evaluate_on_validation",
    "optimize_threshold",
    "oof_027",
    "oof_033",
    "oof_tabpfn",
]
