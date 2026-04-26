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

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW  # noqa: E402
from src.models.artifact_store import save_metadata, save_submission  # noqa: E402
from src.models.evaluation import optimize_threshold  # noqa: E402
from src.models.stacking_oof import oof_tabpfn  # noqa: E402
from src.reports.experiments.log import get_next_exp_id  # noqa: E402
from src.scripts.common import ALL_FEATURES_NATIVE, CAT_FEATURES_NATIVE, NUMERIC_FEATURES_NATIVE, preprocess_native  # noqa: E402


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
    print(f"  Cat features: {CAT_FEATURES_NATIVE}")
    print(f"  Num features: {len(NUMERIC_FEATURES_NATIVE)}")
    print(f"  Exp ID      : {exp_id}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x, y = preprocess_native(df_raw)
    print(f"\nDataset: {x.shape[0]:,} filas x {x.shape[1]} features")

    clf_params = {"n_estimators": args.n_estimators, "random_state": 42}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[OOF] Calculando predicciones out-of-fold (5 folds)...")
    oof_p = oof_tabpfn(x, y, cv, n_estimators=args.n_estimators)
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

    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test = preprocess_native(df_test, is_test=True)

    predictions = (final_model.predict_proba(x_test)[:, 1] >= threshold).astype(bool)
    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {"exp_id": exp_id, "model": "TabPFN",
         "cat_features": CAT_FEATURES_NATIVE, "numeric_features": NUMERIC_FEATURES_NATIVE,
         "feature_names": ALL_FEATURES_NATIVE, "n_estimators": args.n_estimators,
         "oof_acc": oof_acc, "oof_acc_with_threshold": thr_acc,
         "oof_roc_auc": oof_roc, "threshold": threshold, "n_train_samples": len(x)},
        exp_id, "tabpfn",
    )

    n_true = int(predictions.sum())
    print(f"  Submission: {sub_path.name}")
    print(f"  Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
          f"{len(predictions)-n_true} False")
    print(f"\n[OK] Completado. OOF={oof_acc:.4f} | ROC-AUC={oof_roc:.4f}")


if __name__ == "__main__":
    main()
