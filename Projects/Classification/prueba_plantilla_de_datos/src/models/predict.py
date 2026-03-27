"""
Utilidades de prediccion para Spaceship Titanic.

Aplica el pipeline de preprocesamiento sobre datos de test
y genera predicciones con el modelo entrenado.
"""
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.feature_sets import FeatureSetConfig


_ENCODED_COLS = ["CryoSleep", "Side"]


def _encode_cryosleep(val) -> int:
    """Codifica CryoSleep a entero: True->1, False->0, Unknown->-1."""
    if val in (True, "True"):
        return 1
    if val in (False, "False"):
        return 0
    return -1


def _encode_side(val) -> int:
    """Codifica Side a entero: P->0, S->1, Unknown->-1."""
    if val == "P":
        return 0
    if val == "S":
        return 1
    return -1


def preprocess_test(
    df_test: pd.DataFrame,
    fs: FeatureSetConfig,
    feature_cols: List[str],
    scaler: StandardScaler,
    target_encoder: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Aplica el pipeline completo sobre datos de test.

    Usa fs.test_pipeline (que imputa Age NaN con la mediana en lugar de
    eliminar filas) para garantizar predicciones para todos los registros.

    Args:
        df_test: DataFrame crudo de test.csv.
        fs: FeatureSetConfig del modelo en produccion.
        feature_cols: Columnas esperadas por el modelo.
        scaler: StandardScaler ajustado sobre los datos de train.
        target_encoder: Mapa {columna: {categoria: media_target}} para
            feature sets con target_encode_cols. None si no aplica.

    Returns:
        DataFrame listo para prediccion con las mismas columnas que el modelo.
    """
    df = fs.test_pipeline(df_test)

    # Label encoding (siempre presente)
    df["CryoSleep_Encoded"] = df["CryoSleep"].apply(_encode_cryosleep)
    df["Side_Encoded"] = df["Side"].apply(_encode_side)

    # Target encoding (solo si el feature set lo requiere)
    if target_encoder:
        for col, mapping in target_encoder.items():
            global_mean = sum(mapping.values()) / len(mapping)
            encoded_col = f"{col}_TE"
            df[encoded_col] = df[col].map(mapping).fillna(global_mean)

    # One-Hot Encoding para columnas categoricas restantes
    if fs.categorical_cols:
        df = pd.get_dummies(df, columns=fs.categorical_cols, drop_first=False)

    # Drop de columnas
    cols_to_drop = fs.features_to_drop + [
        c for c in _ENCODED_COLS if c in df.columns
    ]
    if fs.target_encode_cols:
        cols_to_drop = cols_to_drop + list(fs.target_encode_cols)
    cols_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_existing)

    # Alinear con las columnas del modelo (rellena 0 si alguna OHE no aparece en test)
    x_test = df.reindex(columns=feature_cols, fill_value=0)

    bool_cols = x_test.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        x_test[bool_cols] = x_test[bool_cols].astype(int)

    numeric_active = [f for f in fs.numeric_features if f in x_test.columns]
    x_test[numeric_active] = scaler.transform(x_test[numeric_active])

    return x_test


def generate_submission(
    model: Any,
    x_test: pd.DataFrame,
    test_ids: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Genera el DataFrame de submission a partir de las predicciones del modelo.

    Args:
        model: Modelo entrenado con interfaz sklearn.
        x_test: Features de test preprocesadas.
        test_ids: Serie con los PassengerId originales de test.
        threshold: Umbral de clasificacion (default 0.5). Si difiere de 0.5,
            usa predict_proba en lugar de predict.

    Returns:
        DataFrame con columnas PassengerId y Transported (bool).
    """
    if abs(threshold - 0.5) < 1e-6:
        predictions = model.predict(x_test).astype(bool)
    else:
        y_proba = model.predict_proba(x_test)[:, 1]
        predictions = (y_proba >= threshold).astype(bool)
    return pd.DataFrame(
        {"PassengerId": test_ids.values, "Transported": predictions}
    )
