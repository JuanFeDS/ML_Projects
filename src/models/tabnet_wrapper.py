"""
Wrapper sklearn-compatible para TabNetClassifier (pytorch-tabnet).

Permite usar TabNet con la misma interfaz que el resto del pipeline:
joblib.dump / load, predict_proba con DataFrames, etc.

El wrapper convierte DataFrames a numpy internamente y expone
get_params / set_params para compatibilidad con Optuna.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn para TabNetClassifier.

    Parametros de arquitectura:
        n_d: Dimension del embedding de decision (igual a n_a por defecto).
        n_a: Dimension del embedding de atencion.
        n_steps: Pasos secuenciales de atencion (profundidad).
        gamma: Coeficiente de reutilizacion de features entre pasos.
        n_independent: Capas FC independientes en cada paso.
        n_shared: Capas FC compartidas entre pasos.
        momentum: Momentum del batch normalization.

    Parametros de entrenamiento:
        max_epochs: Maximo de epocas de entrenamiento.
        patience: Epocas sin mejora antes de parar (early stopping).
        batch_size: Tamano de mini-batch.
        virtual_batch_size: Tamano de ghost batch normalization.
        learning_rate: Tasa de aprendizaje del optimizador Adam.
        seed: Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.02,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 1024,
        virtual_batch_size: int = 256,
        learning_rate: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def _to_numpy(self, X) -> np.ndarray:
        """Convierte DataFrame o array a numpy float32."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.array(X, dtype=np.float32)

    def fit(self, X, y, eval_set=None):
        """Entrena TabNet. eval_set = [(X_val, y_val)] para early stopping.

        Args:
            X: Features de entrenamiento (DataFrame o ndarray).
            y: Target de entrenamiento.
            eval_set: Lista de tuplas [(X_val, y_val)] para early stopping.
                Si None, entrena sin early stopping.

        Returns:
            self
        """
        from pytorch_tabnet.tab_model import TabNetClassifier  # pylint: disable=import-outside-toplevel

        x_np = self._to_numpy(X)
        y_np = np.array(y, dtype=int)

        self.model_ = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            momentum=self.momentum,
            optimizer_params={"lr": self.learning_rate},
            verbose=0,
            seed=self.seed,
        )

        fit_kwargs = {
            "X_train": x_np,
            "y_train": y_np,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
        }

        if eval_set is not None:
            x_val_np = self._to_numpy(eval_set[0][0])
            y_val_np = np.array(eval_set[0][1], dtype=int)
            fit_kwargs["eval_set"] = [(x_val_np, y_val_np)]
            fit_kwargs["eval_metric"] = ["accuracy"]

        self.model_.fit(**fit_kwargs)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X) -> np.ndarray:
        """Predice la clase.

        Args:
            X: Features (DataFrame o ndarray).

        Returns:
            Array de predicciones binarias.
        """
        x_np = self._to_numpy(X)
        return self.model_.predict(x_np)

    def predict_proba(self, X) -> np.ndarray:
        """Predice probabilidades de clase.

        Args:
            X: Features (DataFrame o ndarray).

        Returns:
            Array (n_samples, 2) con probabilidades [clase_0, clase_1].
        """
        x_np = self._to_numpy(X)
        return self.model_.predict_proba(x_np)
