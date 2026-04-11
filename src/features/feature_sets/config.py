"""
Definición del dataclass FeatureSetConfig.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class FeatureSetConfig:
    """Configuración inmutable de un feature set.

    Attributes:
        description: Descripción del feature set y sus diferencias vs parent.
        pipeline: Función que transforma el DataFrame de train (elimina Age NaN).
        test_pipeline: Función que transforma el DataFrame de test (imputa Age NaN).
        numeric_features: Features numéricas a escalar con StandardScaler.
        categorical_cols: Columnas a codificar con One-Hot Encoding.
        features_to_drop: Columnas a eliminar antes del entrenamiento.
        target_encode_cols: Columnas a codificar con Target Encoding.
        parent: Nombre del feature set del que hereda (None si es el primero).
        deprecated: Si True, el feature set está retirado y no debe usarse en nuevos experimentos.
    """

    description: str
    pipeline: Callable[[pd.DataFrame], pd.DataFrame]
    test_pipeline: Callable[[pd.DataFrame], pd.DataFrame]
    numeric_features: List[str]
    categorical_cols: List[str]
    features_to_drop: List[str]
    target_encode_cols: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    deprecated: bool = False
