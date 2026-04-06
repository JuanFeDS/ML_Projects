"""
Mixture of Experts para clasificacion — Spaceship Titanic.

Implementa un MoE con gate determinista basado en CryoSleep_Encoded:
  - Segmento cryo  (CryoSleep_Encoded == 1): passengers en criostasis.
    Sus gastos son siempre 0 → clasificacion basada en demografia y cabina.
  - Segmento activo (CryoSleep_Encoded != 1): passengers activos/desconocidos.
    Los patrones de gasto dominan la senal.

Con drop_zero_variance=True (default), cada experto solo ve las columnas
con varianza positiva dentro de su segmento. Esto elimina automaticamente
las features de gasto del experto cryo (todas son 0 → varianza = 0) y
CryoSleep_Encoded del experto activo (varianza minima al ser constante por
segmento), dejando a cada experto trabajar solo con senal relevante.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class MixtureOfExperts(BaseEstimator, ClassifierMixin):
    """Hard-gated MoE con enrutamiento por CryoSleep_Encoded.

    Entrena dos estimadores independientes:
    - expert_cryo_: ajustado sobre passengers con CryoSleep=True (encoded=1).
    - expert_active_: ajustado sobre el resto (CryoSleep=False/Unknown).

    Con drop_zero_variance=True, detecta automaticamente en fit() que columnas
    tienen varianza cero dentro de cada segmento y las excluye de ese experto.
    En el segmento cryo esto elimina todas las features de gasto (siempre 0).

    La clase es compatible con sklearn: soporta cross_val_score, clone()
    y el protocolo fit/predict/predict_proba.

    Attributes:
        base_estimator: Estimador sklearn que se clona para cada experto.
        gate_col: Columna de enrutamiento (debe estar en X como DataFrame).
        min_segment_size: Minimo de muestras para crear un experto.
        drop_zero_variance: Si True, elimina columnas con var=0 por segmento.
        expert_cryo_: Estimador entrenado en el segmento cryo (post-fit).
        expert_active_: Estimador entrenado en el segmento activo (post-fit).
        cryo_cols_: Columnas usadas por expert_cryo_ (post-fit).
        active_cols_: Columnas usadas por expert_active_ (post-fit).
        classes_: Clases unicas del target.
    """

    def __init__(
        self,
        base_estimator,
        gate_col: str = "CryoSleep_Encoded",
        min_segment_size: int = 50,
        drop_zero_variance: bool = True,
    ):
        """
        Args:
            base_estimator: Estimador base sklearn (e.g. CatBoostClassifier).
                Se clona de forma independiente para cada experto.
            gate_col: Columna de enrutamiento. Debe estar presente en X.
            min_segment_size: Minimo de muestras para entrenar un experto.
            drop_zero_variance: Si True, cada experto solo ve columnas con
                varianza > 0 dentro de su segmento de entrenamiento.
        """
        self.base_estimator = base_estimator
        self.gate_col = gate_col
        self.min_segment_size = min_segment_size
        self.drop_zero_variance = drop_zero_variance

    def _cryo_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve mascara booleana True para el segmento cryo."""
        if not hasattr(X, "columns"):
            raise TypeError(
                f"X debe ser un DataFrame con columnas nombradas "
                f"(requiere '{self.gate_col}')."
            )
        return X[self.gate_col].values == 1

    def _nonzero_cols(self, X_seg: pd.DataFrame) -> List[str]:
        """Columnas con varianza positiva en el segmento dado."""
        return X_seg.columns[X_seg.var() > 1e-10].tolist()

    def fit(self, X, y):
        """Entrena un experto independiente por segmento.

        Args:
            X: Features DataFrame con self.gate_col.
            y: Target Series o array.

        Returns:
            self
        """
        self.classes_ = np.unique(y)
        cryo_mask = self._cryo_mask(X)
        active_mask = ~cryo_mask
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        self.expert_cryo_: Optional[object] = None
        self.expert_active_: Optional[object] = None
        self.cryo_cols_: List[str] = X.columns.tolist()
        self.active_cols_: List[str] = X.columns.tolist()

        x_cryo = X[cryo_mask]
        x_active = X[active_mask]

        if self.drop_zero_variance:
            self.cryo_cols_ = self._nonzero_cols(x_cryo) if cryo_mask.sum() > 0 else self.cryo_cols_
            self.active_cols_ = self._nonzero_cols(x_active) if active_mask.sum() > 0 else self.active_cols_

        if cryo_mask.sum() >= self.min_segment_size:
            self.expert_cryo_ = clone(self.base_estimator)
            self.expert_cryo_.fit(x_cryo[self.cryo_cols_], y_arr[cryo_mask])

        if active_mask.sum() >= self.min_segment_size:
            self.expert_active_ = clone(self.base_estimator)
            self.expert_active_.fit(x_active[self.active_cols_], y_arr[active_mask])

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidades predichas enrutando cada muestra a su experto.

        Si un segmento no tiene experto entrenado (demasiado pequeno al
        ajustar), usa el experto del otro segmento como fallback.

        Args:
            X: Features DataFrame.

        Returns:
            Array (n_samples, 2) con probabilidades [P(0), P(1)].
        """
        cryo_mask = self._cryo_mask(X)
        active_mask = ~cryo_mask
        probas = np.zeros((len(X), 2))

        if cryo_mask.sum() > 0:
            expert = self.expert_cryo_ or self.expert_active_
            cols = self.cryo_cols_ if self.expert_cryo_ else self.active_cols_
            if expert is not None:
                probas[cryo_mask] = expert.predict_proba(X[cryo_mask][cols])

        if active_mask.sum() > 0:
            expert = self.expert_active_ or self.expert_cryo_
            cols = self.active_cols_ if self.expert_active_ else self.cryo_cols_
            if expert is not None:
                probas[active_mask] = expert.predict_proba(X[active_mask][cols])

        return probas

    def predict(self, X) -> np.ndarray:
        """Predicciones binarias.

        Args:
            X: Features DataFrame.

        Returns:
            Array (n_samples,) con predicciones 0/1.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_segment_sizes(self, X) -> dict:
        """Tamano de cada segmento y features activas por experto.

        Args:
            X: Features DataFrame.

        Returns:
            Diccionario con conteos y columnas por segmento.
        """
        cryo_mask = self._cryo_mask(X)
        result = {
            "cryo": int(cryo_mask.sum()),
            "active": int((~cryo_mask).sum()),
        }
        if hasattr(self, "cryo_cols_"):
            result["cryo_features"] = len(self.cryo_cols_)
            result["active_features"] = len(self.active_cols_)
        return result
