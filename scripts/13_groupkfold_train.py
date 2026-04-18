"""
13_groupkfold_train.py -- CatBoost con GroupKFold por LastName (validacion honesta)

Corrige dos problemas del pipeline anterior:
  1. Split random rompia familias → LastName TE veia sus propios targets en val
  2. Threshold optimizado sobre val contaminado

Cambios respecto a exp-033:
  - CV: GroupKFold(LastName) en lugar de StratifiedKFold
  - Threshold: 0.5 fijo (sin optimizacion sobre val)
  - Modelo: CatBoost native (mismo, sin TE externo)

Uso:
    python scripts/13_groupkfold_train.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW
from src.features.constants import TARGET
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets import FEATURE_SETS
from src.features.feature_sets.pipelines import _pipeline_fs017
from src.reports.experiments.log import get_next_exp_id

CAT_FEATURES = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]
NUMERIC_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

CATBOOST_PARAMS = {
    "iterations": 546,
    "depth": 7,
    "learning_rate": 0.07071005741224391,
    "l2_leaf_reg": 21.119964967140604,
    "bagging_temperature": 0.8011251001518414,
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}


def _preprocess(df: pd.DataFrame, is_test: bool = False):
    """Pipeline fs-017 + encodings manuales. Retorna (X, y, groups) o (X, groups)."""
    fs = FEATURE_SETS["fs-017_lastname_te"]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)

    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)
    for col in CAT_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)

    groups = out["LastName"].values
    x = out[[c for c in ALL_FEATURES if c in out.columns]]

    if not is_test:
        y = out[TARGET].astype(int)
        return x, y, groups
    return x, groups


def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("13_groupkfold_train.py -- CatBoost + GroupKFold(LastName)")
    print(f"  Threshold: 0.5 (fijo)")
    print(f"  Exp ID   : {exp_id}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x, y, groups = _preprocess(df_raw)

    n_groups = len(set(groups))
    print(f"\nDataset: {len(y):,} muestras | {x.shape[1]} features | {n_groups:,} grupos (LastName)")

    # --- OOF con GroupKFold ---
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))

    print("\n[OOF] GroupKFold (5 folds) por LastName...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), 1):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pool_tr = Pool(x_tr, y_tr, cat_features=CAT_FEATURES)
        pool_val = Pool(x_val, cat_features=CAT_FEATURES)

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(pool_tr)
        oof_proba[val_idx] = model.predict_proba(pool_val)[:, 1]

        fold_acc = accuracy_score(y_val, (oof_proba[val_idx] >= 0.5).astype(int))
        # Verificar que no hay overlap de grupos entre train y val
        groups_tr = set(groups[tr_idx])
        groups_val = set(groups[val_idx])
        overlap = len(groups_tr & groups_val)
        print(f"  Fold {fold}/5 — acc={fold_acc:.4f} | val_groups={len(groups_val):,} | overlap={overlap}")

    oof_acc = accuracy_score(y, (oof_proba >= 0.5).astype(int))
    oof_roc = roc_auc_score(y, oof_proba)
    print(f"\n  OOF acc (thr=0.5) = {oof_acc:.4f} | ROC-AUC = {oof_roc:.4f}")

    # --- Modelo final sobre dataset completo ---
    print("\n[FIT] Entrenando modelo final sobre dataset completo...")
    pool_full = Pool(x, y, cat_features=CAT_FEATURES)
    final_model = CatBoostClassifier(**CATBOOST_PARAMS)
    final_model.fit(pool_full)

    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_CatBoost_groupkfold.pkl"
    joblib.dump(final_model, artifact)
    print(f"  Guardado: {artifact.name}")

    # --- Predicciones en test ---
    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test, _ = _preprocess(df_test, is_test=True)

    test_proba = final_model.predict_proba(Pool(x_test, cat_features=CAT_FEATURES))[:, 1]
    predictions = (test_proba >= 0.5).astype(bool)

    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame({"PassengerId": test_ids.values, "Transported": predictions}).to_csv(
        sub_path, index=False
    )

    n_true = int(predictions.sum())
    print(f"  Submission: {sub_path.name}")
    print(f"  Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")

    print(f"\n[OK] OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | threshold=0.5 (fijo)")

    meta = {
        "exp_id": exp_id,
        "model": "CatBoost_GroupKFold",
        "cv_strategy": "GroupKFold(n_splits=5, group=LastName)",
        "threshold": 0.5,
        "oof_acc": round(oof_acc, 4),
        "oof_roc_auc": round(oof_roc, 4),
        "cat_features": CAT_FEATURES,
        "feature_names": ALL_FEATURES,
        "n_train_samples": len(y),
        "n_groups": n_groups,
        "catboost_params": CATBOOST_PARAMS,
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_groupkfold_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"     Metadata: {meta_path.name}")


if __name__ == "__main__":
    main()
