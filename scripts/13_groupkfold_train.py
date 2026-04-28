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

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW
from src.models.inference import save_metadata, save_submission
from src.models.training import make_pool, oof_proba_groupkfold
from src.reports.experiments.log import get_next_exp_id
from src.preprocessing.common import CAT_FEATURES_NATIVE, preprocess_native_test_groups, preprocess_native_train_groups

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


def main() -> None:
    """Entrena CatBoost con GroupKFold y genera submission."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("13_groupkfold_train.py -- CatBoost + GroupKFold(LastName)")
    print("  Threshold: 0.5 (fijo)")
    print(f"  Exp ID   : {exp_id}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    x, y, groups = preprocess_native_train_groups(df_raw)

    n_groups = len(set(groups))
    print(
        f"\nDataset: {len(y):,} muestras | {x.shape[1]} features | {n_groups:,} grupos (LastName)"
    )

    print("\n[OOF] GroupKFold (5 folds) por LastName...")
    oof_p = oof_proba_groupkfold(CATBOOST_PARAMS, x, y, groups)
    oof_acc = accuracy_score(y, (oof_p >= 0.5).astype(int))
    oof_roc = roc_auc_score(y, oof_p)
    print(f"\n  OOF acc (thr=0.5) = {oof_acc:.4f} | ROC-AUC = {oof_roc:.4f}")

    print("\n[FIT] Entrenando modelo final sobre dataset completo...")
    final_model = CatBoostClassifier(**CATBOOST_PARAMS)
    final_model.fit(make_pool(x, y))
    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_CatBoost_groupkfold.pkl"
    joblib.dump(final_model, artifact)
    print(f"  Guardado: {artifact.name}")

    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test, _ = preprocess_native_test_groups(df_test)

    predictions = (final_model.predict_proba(make_pool(x_test))[:, 1] >= 0.5).astype(
        bool
    )
    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {
            "exp_id": exp_id,
            "model": "CatBoost_GroupKFold",
            "cv_strategy": "GroupKFold(n_splits=5, group=LastName)",
            "threshold": 0.5,
            "oof_acc": round(oof_acc, 4),
            "oof_roc_auc": round(oof_roc, 4),
            "cat_features": CAT_FEATURES_NATIVE,
            "n_train_samples": len(y),
            "n_groups": n_groups,
            "catboost_params": CATBOOST_PARAMS,
        },
        exp_id,
        "groupkfold",
    )

    n_true = int(predictions.sum())
    print(f"  Submission: {sub_path.name}")
    print(
        f"  Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | "
        f"{len(predictions)-n_true} False"
    )
    print(
        f"\n[OK] OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | threshold=0.5 (fijo)"
    )


if __name__ == "__main__":
    main()
