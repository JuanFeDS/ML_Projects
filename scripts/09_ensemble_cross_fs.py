"""
09_ensemble_cross_fs.py -- Soft Voting Ensemble cross-feature-set

Combina probabilidades de modelos sobre fs-017_lastname_te (CatBoost,
LightGBM, etc.) con exp-033 (CatBoost native con categoricas nativas),
promediando sus probabilidades sobre test.csv.

Cada pipeline corre de forma independiente — no asume feature set compartido.

Uso:
    python scripts/09_ensemble_cross_fs.py
    python scripts/09_ensemble_cross_fs.py --fs017-models exp-027 exp-029
    python scripts/09_ensemble_cross_fs.py --weights 1 1 1   # 027, 029, 033
    python scripts/09_ensemble_cross_fs.py --weights 2 1 2   # mas peso a CatBoosts
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.config.settings import (
    EXPERIMENTS_DIR,
    SUBMISSIONS_DIR,
    TEST_RAW,
    TRAIN_RAW,
    get_scaler_path,
    get_target_encoder_path,
    get_train_scaled,
)
from src.features.feature_sets.pipelines import _pipeline_fs017
from src.features.constants import TARGET
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets import FEATURE_SETS
from src.models.predict import preprocess_test
from src.models.training import optimize_threshold
from src.reports.experiments.log import get_next_exp_id

CAT_FEATURES_033 = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]
NUMERIC_FEATURES_033 = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]
ALL_FEATURES_033 = NUMERIC_FEATURES_033 + CAT_FEATURES_033

FS_017 = "fs-017_lastname_te"
EXP_033_TAG = "exp-033"
DEFAULT_FS017_MODELS = ["exp-027"]


def _find_model(tag: str):
    candidates = list(EXPERIMENTS_DIR.glob(f"{tag}_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro modelo para '{tag}' en {EXPERIMENTS_DIR}")
    return joblib.load(candidates[0])


def _load_fs017_artifacts():
    """Carga scaler y target encoder compartidos de fs-017."""
    scaler = joblib.load(get_scaler_path(FS_017))
    te_path = get_target_encoder_path(FS_017)
    target_encoder = joblib.load(te_path) if te_path.exists() else None
    return scaler, target_encoder


def _preprocess_fs017_test(df_test: pd.DataFrame, model, scaler, target_encoder) -> pd.DataFrame:
    """Pipeline de test para cualquier modelo sobre fs-017."""
    fs = FEATURE_SETS[FS_017]
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
    elif hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_.tolist()
    else:
        feature_cols = None
    return preprocess_test(df_test, fs, feature_cols, scaler, target_encoder)


def _preprocess_033_test(df_test: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de test para exp-033 (CatBoost native)."""
    fs = FEATURE_SETS[FS_017]
    out = fs.test_pipeline(df_test)
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES_033:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out[[c for c in ALL_FEATURES_033 if c in out.columns]]


def _val_probas_fs017(tags: list, df_train: pd.DataFrame, scaler, target_encoder) -> tuple:
    """Probabilidades del val set para todos los modelos fs-017.

    Returns:
        Lista de arrays de probabilidades, array de etiquetas verdaderas.
    """
    df_scaled = pd.read_csv(get_train_scaled(FS_017))
    y_all = df_scaled[TARGET]
    x_all = df_scaled.drop(columns=[TARGET])
    _, x_val, _, y_val = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    probas = []
    for tag in tags:
        model = _find_model(tag)
        if hasattr(model, "feature_name_"):
            feature_cols = list(model.feature_name_)
        elif hasattr(model, "feature_names_in_"):
            feature_cols = model.feature_names_in_.tolist()
        else:
            feature_cols = None
        x_aligned = x_val.reindex(columns=feature_cols, fill_value=0) if feature_cols else x_val
        probas.append(model.predict_proba(x_aligned)[:, 1])

    return probas, y_val.values


def _val_proba_033(df_train: pd.DataFrame) -> np.ndarray:
    """Probabilidades del val set para exp-033."""
    model = _find_model(EXP_033_TAG)
    out = _pipeline_fs017(df_train)
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES_033:
        out[col] = out[col].fillna("Unknown").astype(str)
    y = out[TARGET].astype(int)
    x = out[[c for c in ALL_FEATURES_033 if c in out.columns]]

    _, x_val, _, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    return model.predict_proba(Pool(x_val, cat_features=CAT_FEATURES_033))[:, 1]


def main() -> None:
    """Orquesta el ensemble cross-feature-set."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fs017-models", nargs="+", default=DEFAULT_FS017_MODELS,
        help="Tags de modelos sobre fs-017 (e.g. exp-027 exp-029). Default: exp-027.",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Pesos para cada modelo (fs-017 models en orden + exp-033 al final). "
             "Default: pesos iguales.",
    )
    args = parser.parse_args()

    fs017_tags = args.fs017_models
    all_tags = fs017_tags + [EXP_033_TAG]
    n_models = len(all_tags)

    weights = args.weights if args.weights else [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"--weights debe tener {n_models} valores ({', '.join(all_tags)}).")
    w_arr = np.array(weights) / sum(weights)

    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("09_ensemble_cross_fs.py -- Soft Voting cross-feature-set")
    for tag, w in zip(all_tags, w_arr):
        fs_label = "fs-017" if tag != EXP_033_TAG else "native"
        print(f"  {tag} ({fs_label}) peso={w:.3f}")
    print(f"  Exp ID : {exp_id}")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_RAW)
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()

    scaler, target_encoder = _load_fs017_artifacts()

    # --- Cargar modelos fs-017 y preprocesar test ---
    print("\n[LOAD] Cargando modelos y preprocesando test...")
    fs017_models = [_find_model(tag) for tag in fs017_tags]
    x_tests_fs017 = [_preprocess_fs017_test(df_test, m, scaler, target_encoder) for m in fs017_models]
    x_test_033 = _preprocess_033_test(df_test)
    model_033 = _find_model(EXP_033_TAG)
    for tag, x in zip(fs017_tags, x_tests_fs017):
        print(f"  {tag} test shape: {x.shape}")
    print(f"  {EXP_033_TAG} test shape: {x_test_033.shape}")

    # --- Umbral: optimizar en val set ---
    print("\n[THR] Calibrando umbral en val set (80/20 estratificado)...")
    probas_fs017_val, y_val = _val_probas_fs017(fs017_tags, df_train, scaler, target_encoder)
    proba_033_val = _val_proba_033(df_train)

    all_probas_val = probas_fs017_val + [proba_033_val]
    ensemble_val = sum(w * p for w, p in zip(w_arr, all_probas_val))
    threshold, val_acc = optimize_threshold(y_val, ensemble_val)
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    # --- Predicciones en test ---
    print("\n[PRED] Generando predicciones en test...")
    pool_test_033 = Pool(x_test_033, cat_features=CAT_FEATURES_033)
    probas_test = [m.predict_proba(x)[:, 1] for m, x in zip(fs017_models, x_tests_fs017)]
    probas_test.append(model_033.predict_proba(pool_test_033)[:, 1])
    ensemble_test = sum(w * p for w, p in zip(w_arr, probas_test))
    predictions = (ensemble_test >= threshold).astype(bool)

    # --- Guardar submission ---
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame({"PassengerId": test_ids.values, "Transported": predictions}).to_csv(sub_path, index=False)

    n_true = int(predictions.sum())
    models_str = " + ".join(all_tags)
    print(f"\n[OK] Submission guardado: {sub_path.name}")
    print(f"     Ensemble: {models_str} | umbral={threshold:.4f} | val_acc={val_acc:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")

    meta = {
        "exp_id": exp_id,
        "type": "ensemble_soft_voting",
        "models": all_tags,
        "weights": dict(zip(all_tags, w_arr.tolist())),
        "threshold": threshold,
        "val_accuracy": val_acc,
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_ensemble_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"     Metadata: {meta_path.name}")


if __name__ == "__main__":
    main()
