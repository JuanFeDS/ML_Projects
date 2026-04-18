"""
11_ensemble_tabpfn.py -- Soft Voting Ensemble: CatBoost (fs-017) + CatBoost (native) + TabPFN

Combina tres modelos con diversidad arquitectural maxima:
  - exp-027: CatBoost sobre fs-017_lastname_te (features numericas + TE)
  - exp-033: CatBoost con categoricas nativas
  - exp-047: TabPFN foundation model (cloud API)

Uso:
    python scripts/11_ensemble_tabpfn.py
    python scripts/11_ensemble_tabpfn.py --weights 1 1 2   # mas peso a TabPFN
    python scripts/11_ensemble_tabpfn.py --tabpfn-tag exp-047
"""
import argparse
import json

from dotenv import load_dotenv

# torch debe importarse ANTES que pandas en Windows (WinError 1114).
load_dotenv()
import torch  # noqa: E402
from tabpfn_client import TabPFNClassifier, set_access_token  # noqa: E402
import os  # noqa: E402

set_access_token(os.environ["TABPFN_TOKEN"])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config.settings import (  # noqa: E402
    EXPERIMENTS_DIR,
    SUBMISSIONS_DIR,
    TEST_RAW,
    TRAIN_RAW,
    get_scaler_path,
    get_target_encoder_path,
)
from src.features.constants import TARGET  # noqa: E402
from src.features.engineering import encode_cryosleep, encode_side  # noqa: E402
from src.features.feature_sets import FEATURE_SETS  # noqa: E402
from src.features.feature_sets.pipelines import _pipeline_fs017  # noqa: E402
from src.models.predict import preprocess_test  # noqa: E402
from src.models.training import optimize_threshold  # noqa: E402
from src.reports.experiments.log import get_next_exp_id  # noqa: E402

CAT_FEATURES_033 = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]
NUMERIC_FEATURES_047 = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]
ALL_FEATURES_047 = NUMERIC_FEATURES_047 + CAT_FEATURES_033

FS_017 = "fs-017_lastname_te"
EXP_027_TAG = "exp-027"
EXP_033_TAG = "exp-033"
DEFAULT_TABPFN_TAG = "exp-047"


def _find_model(tag: str):
    candidates = list(EXPERIMENTS_DIR.glob(f"{tag}_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro modelo para '{tag}' en {EXPERIMENTS_DIR}")
    return joblib.load(candidates[0])


def _load_fs017_artifacts():
    scaler = joblib.load(get_scaler_path(FS_017))
    te_path = get_target_encoder_path(FS_017)
    target_encoder = joblib.load(te_path) if te_path.exists() else None
    return scaler, target_encoder


def _preprocess_fs017_test(df_test: pd.DataFrame, model, scaler, target_encoder) -> pd.DataFrame:
    fs = FEATURE_SETS[FS_017]
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
    elif hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_.tolist()
    else:
        feature_cols = None
    return preprocess_test(df_test, fs, feature_cols, scaler, target_encoder)


def _preprocess_047(df: pd.DataFrame, is_test: bool = False):
    """Preprocesa con el mismo pipeline que 10_tabpfn_train.py."""
    fs = FEATURE_SETS[FS_017]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES_033:
        out[col] = out[col].fillna("Unknown").astype(str)

    if not is_test:
        y = out[TARGET].astype(int)
        x = out[[c for c in ALL_FEATURES_047 if c in out.columns]]
        return x, y
    return out[[c for c in ALL_FEATURES_047 if c in out.columns]]


def _val_proba_fs017(tag: str, scaler, target_encoder) -> tuple:
    """Probabilidades en val set (80/20) para un modelo fs-017."""
    from src.config.settings import get_train_scaled  # pylint: disable=import-outside-toplevel

    df_scaled = pd.read_csv(get_train_scaled(FS_017))
    y_all = df_scaled[TARGET]
    x_all = df_scaled.drop(columns=[TARGET])
    _, x_val, _, y_val = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    model = _find_model(tag)
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
    elif hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_.tolist()
    else:
        feature_cols = None
    x_aligned = x_val.reindex(columns=feature_cols, fill_value=0) if feature_cols else x_val
    return model.predict_proba(x_aligned)[:, 1], y_val.values


def _val_proba_033(df_train: pd.DataFrame) -> np.ndarray:
    """Probabilidades en val set para exp-033 (CatBoost native)."""
    model = _find_model(EXP_033_TAG)
    out = _pipeline_fs017(df_train)
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES_033:
        out[col] = out[col].fillna("Unknown").astype(str)
    y = out[TARGET].astype(int)
    x = out[[c for c in ALL_FEATURES_047 if c in out.columns]]
    _, x_val, _, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    return model.predict_proba(Pool(x_val, cat_features=CAT_FEATURES_033))[:, 1]


def _val_proba_047(df_train: pd.DataFrame, n_estimators: int = 8) -> np.ndarray:
    """Probabilidades en val set para TabPFN (fit en train split, predict en val split)."""
    x, y = _preprocess_047(df_train)
    x_tr, x_val, y_tr, _ = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  TabPFN val: fit en {len(x_tr):,} muestras, predict en {len(x_val):,}...")
    model = TabPFNClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_tr, y_tr)
    return model.predict_proba(x_val)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs=3, type=float, default=[1.0, 1.0, 1.0],
        metavar=("W_027", "W_033", "W_047"),
        help="Pesos para exp-027, exp-033 y tabpfn. Default: 1 1 1.",
    )
    parser.add_argument(
        "--tabpfn-tag", default=DEFAULT_TABPFN_TAG,
        help=f"Tag del modelo TabPFN. Default: {DEFAULT_TABPFN_TAG}.",
    )
    parser.add_argument(
        "--tabpfn-estimators", type=int, default=8,
        help="n_estimators para el forward pass TabPFN en val set. Default: 8.",
    )
    args = parser.parse_args()

    all_tags = [EXP_027_TAG, EXP_033_TAG, args.tabpfn_tag]
    w_arr = np.array(args.weights) / sum(args.weights)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("11_ensemble_tabpfn.py -- Soft Voting (3 modelos)")
    for tag, w in zip(all_tags, w_arr):
        print(f"  {tag}  peso={w:.3f}")
    print(f"  Exp ID : {exp_id}")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_RAW)
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()

    # --- Preprocesar test ---
    print("\n[LOAD] Preprocesando test para cada pipeline...")
    scaler, target_encoder = _load_fs017_artifacts()
    model_027 = _find_model(EXP_027_TAG)
    x_test_027 = _preprocess_fs017_test(df_test, model_027, scaler, target_encoder)
    model_033 = _find_model(EXP_033_TAG)
    x_test_033 = _preprocess_047(df_test, is_test=True)
    x_test_047 = _preprocess_047(df_test, is_test=True)
    print(f"  exp-027 test shape: {x_test_027.shape}")
    print(f"  exp-033 test shape: {x_test_033.shape}")
    print(f"  exp-047 test shape: {x_test_047.shape}")

    # --- Val set: calibrar umbral ---
    print("\n[THR] Calculando probas en val set (80/20 estratificado)...")
    p027_val, y_val = _val_proba_fs017(EXP_027_TAG, scaler, target_encoder)
    p033_val = _val_proba_033(df_train)
    p047_val = _val_proba_047(df_train, args.tabpfn_estimators)

    ensemble_val = w_arr[0] * p027_val + w_arr[1] * p033_val + w_arr[2] * p047_val
    threshold, val_acc = optimize_threshold(y_val, ensemble_val)
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    # --- Predicciones en test ---
    print("\n[PRED] Generando predicciones en test...")
    model_047 = _find_model(args.tabpfn_tag)
    p027_test = model_027.predict_proba(x_test_027)[:, 1]
    p033_test = model_033.predict_proba(Pool(x_test_033, cat_features=CAT_FEATURES_033))[:, 1]
    p047_test = model_047.predict_proba(x_test_047)[:, 1]

    ensemble_test = w_arr[0] * p027_test + w_arr[1] * p033_test + w_arr[2] * p047_test
    predictions = (ensemble_test >= threshold).astype(bool)

    # --- Guardar submission ---
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame({"PassengerId": test_ids.values, "Transported": predictions}).to_csv(
        sub_path, index=False
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission guardado: {sub_path.name}")
    print(f"     Ensemble: {' + '.join(all_tags)}")
    print(f"     Pesos: {dict(zip(all_tags, w_arr.tolist()))}")
    print(f"     Umbral={threshold:.4f} | val_acc={val_acc:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")

    meta = {
        "exp_id": exp_id,
        "type": "ensemble_soft_voting_tabpfn",
        "models": all_tags,
        "weights": dict(zip(all_tags, w_arr.tolist())),
        "threshold": threshold,
        "val_accuracy": val_acc,
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_ensemble_tabpfn_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"     Metadata: {meta_path.name}")


if __name__ == "__main__":
    main()
