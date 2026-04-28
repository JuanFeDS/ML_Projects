"""Subpaquete de entrenamiento — re-exporta la interfaz publica."""

from src.models.training.catalogue import MODELS, MOE_BASE_ESTIMATOR, PARAM_SPACES
from src.models.training.catboost_native import (
    CatBoostTuneConfig,
    GroupKFoldConfig,
    make_pool,
    oof_proba,
    oof_proba_groupkfold,
    tune,
)
from src.models.training.pipeline_utils import (
    add_model_prefix,
    make_fold_te_pipeline,
    prefix_param_space,
)
from src.models.training.tuning import TuneConfig, tune_model

__all__ = [
    "MODELS",
    "MOE_BASE_ESTIMATOR",
    "PARAM_SPACES",
    "CatBoostTuneConfig",
    "GroupKFoldConfig",
    "make_pool",
    "oof_proba",
    "oof_proba_groupkfold",
    "tune",
    "add_model_prefix",
    "make_fold_te_pipeline",
    "prefix_param_space",
    "TuneConfig",
    "tune_model",
]
