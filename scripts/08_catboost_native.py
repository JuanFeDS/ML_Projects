"""
08_catboost_native.py - CatBoost con categoricas nativas

En lugar de OHE + TE manual, pasa Deck, HomePlanet, Destination,
AgeCategory y LastName directamente a CatBoost como cat_features.
CatBoost usa Ordered Target Statistics internamente (fold-aware, sin leakage).

18 features: 13 numericas + 5 categoricas nativas.

Ejecutar:
    python scripts/08_catboost_native.py
    python scripts/08_catboost_native.py --n-iter 50
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW
from catboost import CatBoostClassifier

from src.models.evaluation import optimize_threshold
from src.models.inference import save_metadata, save_submission
from src.models.training import CatBoostTuneConfig, make_pool, oof_proba, tune
from src.reports.experiments.log import get_next_exp_id
from src.preprocessing.common import (
    ALL_FEATURES_NATIVE,
    CAT_FEATURES_NATIVE as CAT_FEATURES,
    NUMERIC_FEATURES_NATIVE,
    preprocess_native_test,
    preprocess_native_train,
)


def main() -> None:
    """Entrena CatBoost con categoricas nativas y genera submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("08_catboost_native.py -- CatBoost con categoricas nativas")
    print(f"  Cat features: {CAT_FEATURES}")
    print(f"  Num features: {len(NUMERIC_FEATURES_NATIVE)}")
    print(f"  Exp ID      : {exp_id}")
    print(f"  n_iter      : {args.n_iter}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x, y = preprocess_native_train(df_raw)
    print(f"\nDataset: {x.shape[0]:,} filas x {x.shape[1]} features")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    study_db = str(EXPERIMENTS_DIR / f"exp-{exp_id}_catboost_native_study.db")
    params_cache = EXPERIMENTS_DIR / f"exp-{exp_id}_catboost_native_best_params.json"

    if params_cache.exists():
        print(f"\n[TUNE] Cargando best_params desde cache: {params_cache.name}")
        with open(params_cache, encoding="utf-8") as f:
            best_params = json.load(f)
    else:
        print("\n[TUNE] Tuneando CatBoost con Optuna...")
        best_params = tune(x, y, cv, CatBoostTuneConfig(args.n_iter, study_db))
        with open(params_cache, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    print("\n[OOF] Calculando predicciones out-of-fold...")
    oof_p = oof_proba(best_params, x, y, cv)
    oof_acc = round(float(accuracy_score(y, (oof_p >= 0.5).astype(int))), 4)
    oof_roc = round(float(roc_auc_score(y, oof_p)), 4)
    threshold, thr_acc = optimize_threshold(y, oof_p)
    print(
        f"  OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | thr={threshold:.4f} -> {thr_acc:.4f}"
    )

    print("\n[FIT] Entrenando modelo final...")
    final_model = CatBoostClassifier(
        **best_params, verbose=0, allow_writing_files=False
    )
    final_model.fit(make_pool(x, y))
    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_CatBoost_native.pkl"
    joblib.dump(final_model, artifact)
    print(f"  Guardado: {artifact.name}")

    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test = preprocess_native_test(df_test)
    y_proba = final_model.predict_proba(make_pool(x_test))[:, 1]
    predictions = (y_proba >= threshold).astype(bool)

    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {
            "exp_id": exp_id,
            "model": "CatBoost_native",
            "cat_features": CAT_FEATURES,
            "numeric_features": NUMERIC_FEATURES_NATIVE,
            "feature_names": ALL_FEATURES_NATIVE,
            "oof_acc": oof_acc,
            "oof_acc_with_threshold": thr_acc,
            "oof_roc_auc": oof_roc,
            "threshold": threshold,
            "best_params": best_params,
            "n_train_samples": len(x),
        },
        exp_id,
        "catboost_native",
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(
        f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | {len(predictions)-n_true} False"
    )


if __name__ == "__main__":
    main()
