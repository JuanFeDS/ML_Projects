"""
05_ensemble.py -- Soft Voting Ensemble

Promedia las probabilidades de prediccion de multiples modelos entrenados
sobre el mismo feature set. Diferente al stacking: no hay meta-learner,
solo el promedio simple (o ponderado) de las salidas de cada modelo.

Uso:
    python scripts/05_ensemble.py
    python scripts/05_ensemble.py --models exp-011 exp-012 exp-013
    python scripts/05_ensemble.py --models exp-011 exp-012 exp-013 --weights 1 1 2
    python scripts/05_ensemble.py --feature-set fs-004_target_encoding

Los modelos deben haber sido entrenados sobre el mismo feature set.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    EXPERIMENTS_DIR,
    SUBMISSIONS_DIR,
    TEST_RAW,
    TRAIN_RAW,
    get_scaler_path,
    get_target_encoder_path,
)
from src.features.feature_sets import FEATURE_SETS
from src.models.predict import preprocess_test

TARGET = "Transported"
DEFAULT_MODELS = ["exp-011", "exp-012", "exp-013"]
DEFAULT_FS = "fs-004_target_encoding"


def _find_model_path(exp_tag: str) -> Path:
    """Busca el .pkl del experimento en models/experiments/."""
    candidates = list(EXPERIMENTS_DIR.glob(f"{exp_tag}_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontro modelo para '{exp_tag}' en {EXPERIMENTS_DIR}. "
            f"Archivos disponibles: {list(EXPERIMENTS_DIR.glob('*.pkl'))}"
        )
    return candidates[0]


def _load_val_set(fs_name: str) -> tuple:
    """Reconstruye el val set con el mismo split que 03_train.py.

    Usa random_state=42 y test_size=0.2 estratificado, identico al pipeline
    de entrenamiento, para que el umbral optimo sea comparable.

    Args:
        fs_name: Nombre del feature set.

    Returns:
        Tupla (x_val, y_val) con las mismas columnas que el modelo.
    """
    from src.config.settings import get_train_scaled
    train_path = get_train_scaled(fs_name)
    df = pd.read_csv(train_path)
    y = df[TARGET]
    x = df.drop(columns=[TARGET])
    _, x_val, _, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    return x_val, y_val


def _find_optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Busca el umbral que maximiza accuracy en y_true.

    Args:
        y_true: Etiquetas reales (0/1).
        proba: Probabilidades del ensemble.

    Returns:
        Umbral optimo en [0.3, 0.7].
    """
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        acc = (proba >= thr).astype(int) == y_true.astype(int)
        acc_val = acc.mean()
        if acc_val > best_acc:
            best_acc = acc_val
            best_thr = thr
    return float(round(best_thr, 4))


def _align_to_model(x_base: pd.DataFrame, model, base_cols: list) -> pd.DataFrame:
    """Alinea las columnas del DataFrame al esquema que el modelo espera.

    Cada libreria puede usar una convencion distinta para nombres de columnas
    OHE (e.g. LightGBM reemplaza espacios por _, XGBoost los preserva).
    Esta funcion normaliza ambos lados para crear el mapeo correcto.

    Args:
        x_base: DataFrame con columnas en el esquema canonico del CSV.
        model: Modelo entrenado (LGBMClassifier, XGBClassifier, CatBoostClassifier...).
        base_cols: Lista de columnas canonicas (del CSV).

    Returns:
        DataFrame con columnas renombradas al esquema del modelo.
    """
    if not hasattr(model, "feature_names_in_"):
        # CatBoost: no tiene feature_names_in_, fue entrenado con las mismas
        # columnas del CSV. Devolver tal cual.
        return x_base.reindex(columns=base_cols, fill_value=0)

    model_cols = model.feature_names_in_.tolist()

    # Normalizar ambos lados a underscores para encontrar la correspondencia
    base_norm = {c.replace(" ", "_"): c for c in base_cols}
    model_norm = {c.replace(" ", "_"): c for c in model_cols}

    # Mapa: nombre_canonico -> nombre_del_modelo
    rename_map = {}
    for norm_key, base_col in base_norm.items():
        if norm_key in model_norm:
            rename_map[base_col] = model_norm[norm_key]

    x_aligned = x_base.rename(columns=rename_map)
    return x_aligned.reindex(columns=model_cols, fill_value=0)


def main() -> None:
    """Orquesta el ensemble de soft voting."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Tags de experimentos a ensamblar (e.g. exp-011 exp-012 exp-013).",
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FS,
        help="Feature set comun a todos los modelos.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Pesos para cada modelo (mismo orden que --models). Default: pesos iguales.",
    )
    args = parser.parse_args()

    fs_name = args.feature_set
    model_tags = args.models
    weights = args.weights

    if weights is not None and len(weights) != len(model_tags):
        raise ValueError("--weights debe tener el mismo numero de elementos que --models.")
    if weights is None:
        weights = [1.0] * len(model_tags)
    weights_arr = np.array(weights, dtype=float) / sum(weights)

    print("=" * 60)
    print("05_ensemble.py -- Soft Voting Ensemble")
    print(f"  Feature set : {fs_name}")
    print(f"  Modelos     : {model_tags}")
    print(f"  Pesos       : {weights_arr.tolist()}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cargar artefactos compartidos del feature set
    # ------------------------------------------------------------------
    fs = FEATURE_SETS[fs_name]
    scaler = joblib.load(get_scaler_path(fs_name))
    target_encoder = None
    te_path = get_target_encoder_path(fs_name)
    if te_path.exists():
        target_encoder = joblib.load(te_path)
        print(f"  Target encoder: {list(target_encoder.keys())}")

    # ------------------------------------------------------------------
    # 2. Cargar modelos
    # ------------------------------------------------------------------
    models = []
    feature_cols = None
    for tag in model_tags:
        path = _find_model_path(tag)
        model = joblib.load(path)
        print(f"  [OK] {tag}: {path.name} ({type(model).__name__})")
        # Tomar feature_cols del primer modelo que las tenga
        if feature_cols is None and hasattr(model, "feature_names_in_"):
            feature_cols = model.feature_names_in_.tolist()
        models.append(model)

    # feature_cols se usa mas abajo para preprocess_test; aqui no es critico
    print(f"  Features referencia: {len(feature_cols) if feature_cols else 'N/A (CatBoost)'}")

    # ------------------------------------------------------------------
    # 3. Optimizar umbral en val set (mismo split que 03_train.py)
    # ------------------------------------------------------------------
    print("\n[THR] Optimizando umbral en val set...")
    x_val, y_val = _load_val_set(fs_name)
    # Las columnas del CSV son el esquema canonico (get_dummies produce espacios).
    base_cols = x_val.columns.tolist()
    print(f"  Columnas CSV (base): {len(base_cols)}")

    ensemble_proba_val = np.zeros(len(x_val))
    for model, w in zip(models, weights_arr):
        x_aligned = _align_to_model(x_val, model, base_cols)
        proba = model.predict_proba(x_aligned)[:, 1]
        ensemble_proba_val += w * proba

    threshold = _find_optimal_threshold(y_val.values, ensemble_proba_val)
    val_acc = ((ensemble_proba_val >= threshold) == y_val.values).mean()
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    # ------------------------------------------------------------------
    # 4. Preprocesar test.csv
    # ------------------------------------------------------------------
    print("\n[PRED] Preprocesando test.csv...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    # Usar base_cols (esquema canonico del CSV) para que preprocess_test
    # produzca las mismas columnas que el val set.
    x_test = preprocess_test(
        df_test,
        fs=fs,
        feature_cols=base_cols,
        scaler=scaler,
        target_encoder=target_encoder,
    )
    print(f"  Shape test: {x_test.shape}")

    # ------------------------------------------------------------------
    # 5. Ensemble: promediar probabilidades y aplicar umbral
    # ------------------------------------------------------------------
    ensemble_proba_test = np.zeros(len(x_test))
    for model, w in zip(models, weights_arr):
        x_aligned = _align_to_model(x_test, model, base_cols)
        proba = model.predict_proba(x_aligned)[:, 1]
        ensemble_proba_test += w * proba

    predictions = (ensemble_proba_test >= threshold).astype(bool)

    # ------------------------------------------------------------------
    # 6. Guardar submission con numeracion de experimento
    # ------------------------------------------------------------------
    # Determinar el siguiente exp_id disponible
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SUBMISSIONS_DIR.glob("exp-*_submission.csv"))
    if existing:
        last_id = int(existing[-1].name.split("_")[0].replace("exp-", ""))
        exp_id = f"{last_id + 1:03d}"
    else:
        exp_id = "019"

    submission_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    submission = pd.DataFrame({
        "PassengerId": test_ids.values,
        "Transported": predictions,
    })
    submission.to_csv(submission_path, index=False)

    n_true = int(predictions.sum())
    n_false = len(predictions) - n_true
    print(f"\nDistribucion de predicciones:")
    print(f"  True  (transportado):    {n_true:>5} ({n_true/len(predictions)*100:.1f}%)")
    print(f"  False (no transportado): {n_false:>5} ({n_false/len(predictions)*100:.1f}%)")
    print(f"  Total:                   {len(predictions)}")
    models_str = " + ".join(model_tags)
    print(f"\n[OK] Submission guardado: {submission_path}")
    print(f"     Ensemble: {models_str} | umbral={threshold:.4f} | val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
