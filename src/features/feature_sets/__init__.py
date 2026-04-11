"""
Feature sets para Spaceship Titanic.

Re-exporta la interfaz pública para mantener compatibilidad con imports existentes:
    from src.features.feature_sets import FEATURE_SETS, DEFAULT_FEATURE_SET, FeatureSetConfig
"""
from src.features.feature_sets.config import FeatureSetConfig
from src.features.feature_sets.registry import DEFAULT_FEATURE_SET, FEATURE_SETS

__all__ = ["FeatureSetConfig", "FEATURE_SETS", "DEFAULT_FEATURE_SET"]
