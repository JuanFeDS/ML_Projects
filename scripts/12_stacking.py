"""
12_stacking.py -- Stacking Ensemble con meta-learner LogisticRegression

Nivel 0 — OOF 5-fold por modelo:
  - exp-027: CatBoost sobre fs-017 (Pipeline con TargetEncoder en LastName)
  - exp-033: CatBoost native (categoricas nativas)
  - exp-047: TabPFN cloud API

Nivel 1 — meta-learner LogisticRegression entrenado sobre las 3 OOF.

Uso:
    python scripts/12_stacking.py
    python scripts/12_stacking.py --n-estimators 4   # TabPFN mas rapido
"""
import argparse
import json

from dotenv import load_dotenv

# torch ANTES que pandas (WinError 1114 en Windows)
load_dotenv()
import torch  # noqa: E402
from tabpfn_client import TabPFNClassifier, set_access_token  # noqa: E402
import os  # noqa: E402

set_access_token(os.environ["TABPFN_TOKEN"])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

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

FS_017 = "fs-017_lastname_te"
CAT_FEATURES = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]
NUMERIC_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]
ALL_FEATURES_NATIVE = NUMERIC_FEATURES + CAT_FEATURES

EXP_033_PARAMS = {
    "iterations": 546, "depth": 7,
    "learning_rate": 0.07071005741224391,
    "l2_leaf_reg": 21.119964967140604,
    "bagging_temperature": 0.8011251001518414,
    "random_seed": 42, "verbose": 0,
    "allow_writing_files": False,
}


def _preprocess_native(df: pd.DataFrame, is_test: bool = False):
    """Pipeline compartido por exp-033 y exp-047 (categoricas nativas)."""
    fs = FEATURE_SETS[FS_017]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)
    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)
    if not is_test:
        y = out[TARGET].astype(int)
        x = out[[c for c in ALL_FEATURES_NATIVE if c in out.columns]]
        return x, y
    return out[[c for c in ALL_FEATURES_NATIVE if c in out.columns]]


def _build_027_pipeline():
    """Reconstruye el Pipeline de exp-027 desde cero (evita incompatibilidad sklearn 1.8→1.6)."""
    from sklearn.compose import ColumnTransformer  # pylint: disable=import-outside-toplevel
    from sklearn.pipeline import Pipeline as SKPipeline  # pylint: disable=import-outside-toplevel
    from sklearn.preprocessing import TargetEncoder  # pylint: disable=import-outside-toplevel

    ct = ColumnTransformer(
        transformers=[("te", TargetEncoder(random_state=42, target_type="binary"), ["LastName"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    cb = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.021652690184947157,
        l2_leaf_reg=16.121101290648085, bagging_temperature=0.003097017597283175,
        random_seed=42, verbose=0, allow_writing_files=False,
    )
    return SKPipeline([("preprocessor", ct), ("model", cb)])


def _oof_027(x_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    """OOF para exp-027: Pipeline reconstruido con cross_val_predict."""
    pipe = _build_027_pipeline()
    print("  [027] cross_val_predict (5-fold)...")
    proba = cross_val_predict(pipe, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    acc = accuracy_score(y_train, (proba >= 0.5).astype(int))
    print(f"  [027] OOF acc={acc:.4f}")
    return proba


def _oof_033(x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    """OOF para exp-033: CatBoost native con Pool."""
    proba = np.zeros(len(x))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(x, y), 1):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        pool_tr = Pool(x_tr, y_tr, cat_features=CAT_FEATURES)
        pool_val = Pool(x_val, cat_features=CAT_FEATURES)
        m = CatBoostClassifier(**EXP_033_PARAMS)
        m.fit(pool_tr)
        proba[val_idx] = m.predict_proba(pool_val)[:, 1]
        acc = accuracy_score(y_val, (proba[val_idx] >= 0.5).astype(int))
        print(f"  [033] Fold {fold}/5 acc={acc:.4f}")
    print(f"  [033] OOF acc={accuracy_score(y, (proba >= 0.5).astype(int)):.4f}")
    return proba


def _oof_047(x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, n_estimators: int) -> np.ndarray:
    """OOF para exp-047: TabPFN cloud API."""
    proba = np.zeros(len(x))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(x, y), 1):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = TabPFNClassifier(n_estimators=n_estimators, random_state=42)
        m.fit(x_tr, y_tr)
        proba[val_idx] = m.predict_proba(x_val)[:, 1]
        acc = accuracy_score(y_val, (proba[val_idx] >= 0.5).astype(int))
        print(f"  [047] Fold {fold}/5 acc={acc:.4f}")
    print(f"  [047] OOF acc={accuracy_score(y, (proba >= 0.5).astype(int)):.4f}")
    return proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=8,
                        help="n_estimators TabPFN. Default: 8.")
    args = parser.parse_args()

    exp_id = get_next_exp_id("docs/model/experimentation_log.md")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("12_stacking.py -- Stacking con meta-learner LR")
    print(f"  Base: exp-027 + exp-033 + exp-047 (n_est={args.n_estimators})")
    print(f"  Exp ID: {exp_id}")
    print("=" * 60)

    # --- Datos de entrenamiento ---
    df_raw = pd.read_csv(TRAIN_RAW)
    x_native, y = _preprocess_native(df_raw)

    from src.config.settings import get_train_scaled  # pylint: disable=import-outside-toplevel
    df_scaled = pd.read_csv(get_train_scaled(FS_017))
    y_027 = df_scaled[TARGET]
    x_027 = df_scaled.drop(columns=[TARGET])

    print(f"\nDataset: {len(y):,} muestras | x_native={x_native.shape[1]} feat | x_027={x_027.shape[1]} feat")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Nivel 0: OOF ---
    print("\n[L0] Generando OOF (5-fold) para los 3 modelos base...")
    p027 = _oof_027(x_027, y_027, cv)
    p033 = _oof_033(x_native, y, cv)
    p047 = _oof_047(x_native, y, cv, args.n_estimators)

    # --- Nivel 1: meta-learner ---
    print("\n[L1] Entrenando meta-learner LogisticRegression...")
    X_meta = np.column_stack([p027, p033, p047])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(X_meta, y)
    meta_proba_oof = meta.predict_proba(X_meta)[:, 1]
    threshold, meta_acc = optimize_threshold(y, meta_proba_oof)
    roc = roc_auc_score(y, meta_proba_oof)
    print(f"  Meta OOF acc={accuracy_score(y, (meta_proba_oof >= 0.5).astype(int)):.4f} "
          f"| acc(thr={threshold:.4f})={meta_acc:.4f} | ROC-AUC={roc:.4f}")
    print(f"  Coefs LR: 027={meta.coef_[0][0]:.3f}  033={meta.coef_[0][1]:.3f}  047={meta.coef_[0][2]:.3f}")

    # --- Test: predicciones base ---
    print("\n[TEST] Generando predicciones en test...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()

    scaler = joblib.load(get_scaler_path(FS_017))
    te_path = get_target_encoder_path(FS_017)
    target_encoder = joblib.load(te_path) if te_path.exists() else None

    model_027 = joblib.load(list(EXPERIMENTS_DIR.glob("exp-027_*.pkl"))[0])
    fs = FEATURE_SETS[FS_017]
    feat_cols = list(model_027.feature_names_in_)
    x_test_027 = preprocess_test(df_test, fs, feat_cols, scaler, target_encoder)
    p027_test = model_027.predict_proba(x_test_027)[:, 1]
    print(f"  [027] test shape: {x_test_027.shape}")

    model_033 = joblib.load(list(EXPERIMENTS_DIR.glob("exp-033_*.pkl"))[0])
    x_test_native = _preprocess_native(df_test, is_test=True)
    pool_test = Pool(x_test_native, cat_features=CAT_FEATURES)
    p033_test = model_033.predict_proba(pool_test)[:, 1]
    print(f"  [033] test shape: {x_test_native.shape}")

    model_047 = joblib.load(list(EXPERIMENTS_DIR.glob("exp-047_*.pkl"))[0])
    p047_test = model_047.predict_proba(x_test_native)[:, 1]
    print(f"  [047] test shape: {x_test_native.shape}")

    # --- Meta-learner en test ---
    X_meta_test = np.column_stack([p027_test, p033_test, p047_test])
    meta_proba_test = meta.predict_proba(X_meta_test)[:, 1]
    predictions = (meta_proba_test >= threshold).astype(bool)

    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame({"PassengerId": test_ids.values, "Transported": predictions}).to_csv(
        sub_path, index=False
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(f"     Meta OOF acc(thr)={meta_acc:.4f} | ROC-AUC={roc:.4f} | umbral={threshold:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")

    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_stacking_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "exp_id": exp_id,
            "type": "stacking_lr_meta",
            "base_models": ["exp-027", "exp-033", "exp-047"],
            "meta_learner": "LogisticRegression(C=1.0)",
            "lr_coefs": {"exp-027": meta.coef_[0][0], "exp-033": meta.coef_[0][1], "exp-047": meta.coef_[0][2]},
            "oof_acc_meta": meta_acc,
            "oof_roc_auc": roc,
            "threshold": threshold,
        }, f, indent=2)
    print(f"     Metadata: {meta_path.name}")


if __name__ == "__main__":
    main()
