"""Utilidades para cargar y guardar artefactos de experimentos y submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    EXPERIMENTS_DIR,
    SUBMISSIONS_DIR,
    get_scaler_path,
    get_target_encoder_path,
    get_train_scaled,
)
from src.features.constants import TARGET


def find_model_path(exp_tag: str) -> Path:
    """Busca el artefacto .pkl de un experimento en models/experiments/.

    Args:
        exp_tag: Prefijo del experimento (ej. 'exp-027').

    Returns:
        Path al primer .pkl que coincide con el tag.

    Raises:
        FileNotFoundError: Si no existe ningun .pkl para el tag indicado.
    """
    candidates = list(EXPERIMENTS_DIR.glob(f"{exp_tag}_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontro modelo para '{exp_tag}' en {EXPERIMENTS_DIR}. "
            f"Disponibles: {[p.name for p in EXPERIMENTS_DIR.glob('*.pkl')]}"
        )
    return candidates[0]


def load_model(exp_tag: str) -> Any:
    """Carga el modelo pkl de un experimento.

    Args:
        exp_tag: Prefijo del experimento (ej. 'exp-027').

    Returns:
        Modelo deserializado.
    """
    return joblib.load(find_model_path(exp_tag))


def load_feature_set_artifacts(fs_name: str) -> Tuple[Any, Optional[Any]]:
    """Carga el scaler y el target encoder de un feature set.

    Args:
        fs_name: Nombre del feature set (ej. 'fs-017_lastname_te').

    Returns:
        Tupla (scaler, target_encoder). target_encoder es None si no existe.
    """
    scaler = joblib.load(get_scaler_path(fs_name))
    te_path = get_target_encoder_path(fs_name)
    target_encoder = joblib.load(te_path) if te_path.exists() else None
    return scaler, target_encoder


def load_val_set(fs_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Reconstruye el val set con el mismo split estratificado 80/20 del pipeline principal.

    Args:
        fs_name: Nombre del feature set escalado a usar.

    Returns:
        Tupla (x_val, y_val).
    """
    train_path = get_train_scaled(fs_name)
    df = pd.read_csv(train_path)
    y = df[TARGET]
    x = df.drop(columns=[TARGET])
    _, x_val, _, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    return x_val, y_val


def get_feature_cols(model: Any) -> Optional[List[str]]:
    """Extrae los nombres de features del modelo si los expone.

    Soporta sklearn (feature_names_in_) y LightGBM (feature_name_).

    Args:
        model: Modelo entrenado.

    Returns:
        Lista de nombres de features o None si el modelo no los expone.
    """
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_())
    if hasattr(model, "feature_names_in_"):
        return model.feature_names_in_.tolist()
    return None


def align_features(
    x: pd.DataFrame,
    model: Any,
    base_cols: List[str],
) -> pd.DataFrame:
    """Alinea columnas del DataFrame al esquema que el modelo espera.

    Normaliza ambos lados a underscores para resolver diferencias de nombrado
    entre librerias (LightGBM reemplaza espacios, XGBoost los preserva, etc.).

    Args:
        x: DataFrame con columnas en el esquema canonico.
        model: Modelo entrenado.
        base_cols: Columnas del esquema canonico.

    Returns:
        DataFrame con columnas renombradas al esquema del modelo.
    """
    if not hasattr(model, "feature_names_in_"):
        return x.reindex(columns=base_cols, fill_value=0)

    model_cols = model.feature_names_in_.tolist()
    base_norm = {c.replace(" ", "_"): c for c in base_cols}
    model_norm = {c.replace(" ", "_"): c for c in model_cols}

    rename_map = {
        base_col: model_norm[norm_key]
        for norm_key, base_col in base_norm.items()
        if norm_key in model_norm
    }
    return x.rename(columns=rename_map).reindex(columns=model_cols, fill_value=0)


def save_submission(
    predictions: np.ndarray,
    test_ids: pd.Series,
    exp_id: str,
) -> Path:
    """Guarda el archivo de submission en data/submissions/.

    Args:
        predictions: Array booleano de predicciones.
        test_ids: Series con los PassengerId del test set.
        exp_id: ID del experimento (ej. '034').

    Returns:
        Path al archivo guardado.
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame(
        {
            "PassengerId": test_ids.values,
            "Transported": predictions,
        }
    ).to_csv(sub_path, index=False)
    return sub_path


def load_production_feature_set(meta_prod: dict) -> tuple:
    """Resuelve el feature set del modelo en produccion desde su metadata.

    Si el feature set guardado no existe en el catalogo, usa fs-001_baseline.

    Args:
        meta_prod: Diccionario con el metadata del modelo en produccion.

    Returns:
        Tupla (fs_name, FeatureSetConfig).
    """
    from src.features.feature_sets import (
        FEATURE_SETS,
    )  # pylint: disable=import-outside-toplevel

    fs_name = meta_prod.get("feature_set_name", "fs-001_baseline")
    if fs_name not in FEATURE_SETS:
        print(f"  Feature set '{fs_name}' no encontrado, usando fs-001_baseline.")
        fs_name = "fs-001_baseline"
    return fs_name, FEATURE_SETS[fs_name]


def save_metadata(meta: dict, exp_id: str, name: str) -> Path:
    """Guarda metadata de experimento como JSON en models/experiments/.

    Args:
        meta: Diccionario con la metadata del experimento.
        exp_id: ID del experimento (ej. '034').
        name: Sufijo del nombre de archivo (ej. 'ensemble', 'stacking').

    Returns:
        Path al archivo guardado.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_{name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta_path
