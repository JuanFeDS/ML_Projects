"""
10_tabpfn_train.py -- TabPFN foundation model (cloud API via tabpfn-client)

TabPFN v2 es un transformer pre-entrenado para datos tabulares. No requiere
tuning de hiperparametros — hace inference directa sobre los datos de train/test
como "contexto" (in-context learning). Esta version usa la cloud API de PriorLabs,
evitando el bottleneck de inferencia local en CPU.

Usa las mismas 18 features que exp-033 (13 numericas + 5 categoricas nativas)
para maximizar la diversidad arquitectural respecto a los CatBoost del ensemble.

Uso:
    python scripts/10_tabpfn_train.py
    python scripts/10_tabpfn_train.py --n-estimators 16
"""
import argparse
import json
import os

from dotenv import load_dotenv

# torch debe importarse ANTES que pandas en Windows para evitar conflicto
# de inicializacion de DLLs (ERROR_DLL_INIT_FAILED / WinError 1114).
load_dotenv()
import torch  # noqa: E402
from tabpfn_client import TabPFNClassifier, set_access_token  # noqa: E402

set_access_token(os.environ["TABPFN_TOKEN"])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets.pipelines import _pipeline_fs017
from src.models.training import optimize_threshold
from src.reports.experiments.log import get_next_exp_id

CAT_FEATURES = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]
NUMERIC_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES


def _preprocess(df: pd.DataFrame, is_test: bool = False):
    """Preprocesa train o test con el pipeline fs-017 + encodings manuales.

    Args:
        df: DataFrame crudo (train.csv o test.csv).
        is_test: Si True usa test_pipeline y no retorna y.

    Returns:
        (X, y) si is_test=False, X si is_test=True.
    """
    from src.features.constants import TARGET  # pylint: disable=import-outside-toplevel
    from src.features.feature_sets import FEATURE_SETS  # pylint: disable=import-outside-toplevel

    fs = FEATURE_SETS["fs-017_lastname_te"]
    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)

    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)

    for col in CAT_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)

    if not is_test:
        y = out[TARGET].astype(int)
        x = out[[c for c in ALL_FEATURES if c in out.columns]]
        return x, y

    return out[[c for c in ALL_FEATURES if c in out.columns]]


def _oof_proba(clf_params: dict, x: pd.DataFrame, y: pd.Series,
               cv: StratifiedKFold) -> np.ndarray:
    """Genera probabilidades OOF con CV estratificado."""
    proba = np.zeros(len(x))
    for fold, (train_idx, val_idx) in enumerate(cv.split(x, y), 1):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        model = TabPFNClassifier(**clf_params)
        model.fit(x_tr, y_tr)
        proba[val_idx] = model.predict_proba(x_val)[:, 1]
        acc = accuracy_score(y.iloc[val_idx], (proba[val_idx] >= 0.5).astype(int))
        print(f"  Fold {fold}/5 — acc={acc:.4f}")
    return proba


def main() -> None:
    """Entrena TabPFN y genera submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-estimators", type=int, default=8,
        help="Numero de forward passes del ensemble TabPFN. Default: 8.",
    )
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("10_tabpfn_train.py -- TabPFN foundation model")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  Cat features: {CAT_FEATURES}")
    print(f"  Num features: {len(NUMERIC_FEATURES)}")
    print(f"  Exp ID      : {exp_id}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x, y = _preprocess(df_raw)
    print(f"\nDataset: {x.shape[0]:,} filas x {x.shape[1]} features")

    clf_params = {
        "n_estimators": args.n_estimators,
        "random_state": 42,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[OOF] Calculando predicciones out-of-fold (5 folds)...")
    oof_p = _oof_proba(clf_params, x, y, cv)
    oof_acc = round(float(accuracy_score(y, (oof_p >= 0.5).astype(int))), 4)
    oof_roc = round(float(roc_auc_score(y, oof_p)), 4)
    threshold, thr_acc = optimize_threshold(y, oof_p)
    print(f"  OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | thr={threshold:.4f} -> {thr_acc:.4f}")

    print("\n[FIT] Entrenando modelo final sobre dataset completo...")
    final_model = TabPFNClassifier(**clf_params)
    final_model.fit(x, y)

    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_TabPFN.pkl"
    joblib.dump(final_model, artifact)
    print(f"  Guardado: {artifact.name}")

    meta = {
        "exp_id": exp_id,
        "model": "TabPFN",
        "cat_features": CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_names": ALL_FEATURES,
        "n_estimators": args.n_estimators,
        "oof_acc": oof_acc,
        "oof_acc_with_threshold": thr_acc,
        "oof_roc_auc": oof_roc,
        "threshold": threshold,
        "n_train_samples": len(x),
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_tabpfn_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test = _preprocess(df_test, is_test=True)

    y_proba_test = final_model.predict_proba(x_test)[:, 1]
    predictions = (y_proba_test >= threshold).astype(bool)

    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    pd.DataFrame({"PassengerId": test_ids.values, "Transported": predictions}).to_csv(
        sub_path, index=False
    )

    n_true = int(predictions.sum())
    print(f"  Submission: {sub_path.name}")
    print(f"  Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")
    print(f"\n[OK] Completado. OOF={oof_acc:.4f} | ROC-AUC={oof_roc:.4f}")


if __name__ == "__main__":
    main()
