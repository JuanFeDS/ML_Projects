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
import os

from dotenv import load_dotenv

load_dotenv()
import torch  # noqa: E402
from tabpfn_client import set_access_token  # noqa: E402

set_access_token(os.environ["TABPFN_TOKEN"])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW  # noqa: E402
from src.models.artifact_store import load_model, save_metadata, save_submission  # noqa: E402
from src.models.evaluation import optimize_threshold  # noqa: E402
from src.models.predict import preprocess_test  # noqa: E402
from src.models.stacking_oof import oof_027, oof_033, oof_tabpfn  # noqa: E402
from src.reports.experiments.log import get_next_exp_id  # noqa: E402
from src.scripts.common import CAT_FEATURES_NATIVE, FS_NATIVE, preprocess_native  # noqa: E402
from src.config.settings import get_scaler_path, get_target_encoder_path, get_train_scaled  # noqa: E402
from src.features.feature_sets import FEATURE_SETS  # noqa: E402
from catboost import Pool  # noqa: E402


def main() -> None:
    """Entrena el stacking ensemble y genera submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=8)
    args = parser.parse_args()

    exp_id = get_next_exp_id("docs/model/experimentation_log.md")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("12_stacking.py -- Stacking con meta-learner LR")
    print(f"  Base: exp-027 + exp-033 + exp-047 (n_est={args.n_estimators})")
    print(f"  Exp ID: {exp_id}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x_native, y = preprocess_native(df_raw)
    df_scaled = pd.read_csv(get_train_scaled(FS_NATIVE))
    y_027 = df_scaled["Transported"]
    x_027 = df_scaled.drop(columns=["Transported"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nDataset: {len(y):,} muestras | x_native={x_native.shape[1]} | x_027={x_027.shape[1]}")
    print("\n[L0] Generando OOF (5-fold)...")
    p027 = oof_027(x_027, y_027, cv)
    p033 = oof_033(x_native, y, cv)
    p047 = oof_tabpfn(x_native, y, cv, args.n_estimators)

    print("\n[L1] Entrenando meta-learner LogisticRegression...")
    x_meta = np.column_stack([p027, p033, p047])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(x_meta, y)
    meta_proba_oof = meta.predict_proba(x_meta)[:, 1]
    threshold, meta_acc = optimize_threshold(y, meta_proba_oof)
    roc = roc_auc_score(y, meta_proba_oof)
    print(f"  Meta OOF acc={accuracy_score(y, (meta_proba_oof >= 0.5).astype(int)):.4f} | "
          f"acc(thr={threshold:.4f})={meta_acc:.4f} | ROC-AUC={roc:.4f}")
    print(f"  Coefs LR: 027={meta.coef_[0][0]:.3f}  033={meta.coef_[0][1]:.3f}  047={meta.coef_[0][2]:.3f}")

    print("\n[TEST] Generando predicciones en test...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()

    scaler = joblib.load(get_scaler_path(FS_NATIVE))
    te_path = get_target_encoder_path(FS_NATIVE)
    target_encoder = joblib.load(te_path) if te_path.exists() else None

    model_027 = load_model("exp-027")
    fs = FEATURE_SETS[FS_NATIVE]
    feat_cols = list(model_027.feature_names_in_)
    x_test_027 = preprocess_test(df_test, fs, feat_cols, scaler, target_encoder)
    p027_test = model_027.predict_proba(x_test_027)[:, 1]

    x_test_native = preprocess_native(df_test, is_test=True)
    model_033 = load_model("exp-033")
    model_047 = load_model("exp-047")
    p033_test = model_033.predict_proba(Pool(x_test_native, cat_features=CAT_FEATURES_NATIVE))[:, 1]
    p047_test = model_047.predict_proba(x_test_native)[:, 1]

    meta_proba_test = meta.predict_proba(np.column_stack([p027_test, p033_test, p047_test]))[:, 1]
    predictions = (meta_proba_test >= threshold).astype(bool)

    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {"exp_id": exp_id, "type": "stacking_lr_meta",
         "base_models": ["exp-027", "exp-033", "exp-047"],
         "meta_learner": "LogisticRegression(C=1.0)",
         "lr_coefs": {"exp-027": meta.coef_[0][0], "exp-033": meta.coef_[0][1], "exp-047": meta.coef_[0][2]},
         "oof_acc_meta": meta_acc, "oof_roc_auc": roc, "threshold": threshold},
        exp_id, "stacking",
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(f"     Meta OOF acc(thr)={meta_acc:.4f} | ROC-AUC={roc:.4f} | umbral={threshold:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | {len(predictions)-n_true} False")


if __name__ == "__main__":
    main()
