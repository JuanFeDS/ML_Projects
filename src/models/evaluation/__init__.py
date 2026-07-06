"""Subpaquete de evaluacion — re-exporta la interfaz publica."""

from src.models.evaluation.errors import analyze_errors
from src.models.evaluation.evaluation import (
    compute_oof_metrics,
    evaluate_models,
    evaluate_on_validation,
    optimize_threshold,
)

__all__ = [
    "analyze_errors",
    "compute_oof_metrics",
    "evaluate_models",
    "evaluate_on_validation",
    "optimize_threshold",
]
