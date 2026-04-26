"""Modulo de modelos — re-exports de la interfaz publica."""
from src.models.catalogue import MODELS, MOE_BASE_ESTIMATOR, PARAM_SPACES
from src.models.ensembles import build_moe, build_stacking
from src.models.errors import analyze_errors
from src.models.evaluation import (
    compute_oof_metrics,
    evaluate_models,
    evaluate_on_validation,
    optimize_threshold,
)
from src.models.pipeline_utils import add_model_prefix, make_fold_te_pipeline, prefix_param_space
from src.models.tracking import log_metrics_dict, log_params_dict, mlrun, setup_mlflow
from src.models.tuning import tune_model

__all__ = [
    "MODELS",
    "PARAM_SPACES",
    "MOE_BASE_ESTIMATOR",
    "evaluate_models",
    "evaluate_on_validation",
    "compute_oof_metrics",
    "optimize_threshold",
    "tune_model",
    "build_stacking",
    "build_moe",
    "analyze_errors",
    "make_fold_te_pipeline",
    "prefix_param_space",
    "add_model_prefix",
    "mlrun",
    "setup_mlflow",
    "log_metrics_dict",
    "log_params_dict",
]
