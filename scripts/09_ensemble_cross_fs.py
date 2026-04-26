"""
09_ensemble_cross_fs.py -- Soft Voting Ensemble cross-feature-set

Combina probabilidades de modelos sobre fs-017_lastname_te (CatBoost, LightGBM)
con exp-033 (CatBoost native con categoricas nativas), promediando sus
probabilidades sobre test.csv. Cada pipeline corre de forma independiente.

Uso:
    python scripts/09_ensemble_cross_fs.py
    python scripts/09_ensemble_cross_fs.py --fs017-models exp-027 exp-029
    python scripts/09_ensemble_cross_fs.py --weights 2 1 2
"""
import argparse

import numpy as np
import pandas as pd
from catboost import Pool

from src.config.settings import TEST_RAW, TRAIN_RAW
from src.features.feature_sets import FEATURE_SETS
from src.models.artifact_store import (
    get_feature_cols,
    load_feature_set_artifacts,
    load_model,
    save_metadata,
    save_submission,
)
from src.models.evaluation import optimize_threshold
from src.models.predict import preprocess_test
from src.reports.experiments.log import get_next_exp_id
from src.scripts.common import CAT_FEATURES_NATIVE, FS_NATIVE, preprocess_native

FS_017 = FS_NATIVE
EXP_033_TAG = "exp-033"
DEFAULT_FS017_MODELS = ["exp-027"]


def _val_proba_fs017(tags: list, scaler, target_encoder) -> tuple:
    """Probabilidades en val set para los modelos fs-017."""
    from sklearn.model_selection import train_test_split  # pylint: disable=import-outside-toplevel

    from src.config.settings import get_train_scaled  # pylint: disable=import-outside-toplevel
    from src.features.constants import TARGET  # pylint: disable=import-outside-toplevel

    df = pd.read_csv(get_train_scaled(FS_017))
    y_all = df[TARGET]
    x_all = df.drop(columns=[TARGET])
    _, x_val, _, y_val = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    probas = []
    for tag in tags:
        m = load_model(tag)
        feat_cols = get_feature_cols(m)
        x_aligned = x_val.reindex(columns=feat_cols, fill_value=0) if feat_cols else x_val
        probas.append(m.predict_proba(x_aligned)[:, 1])
    return probas, y_val.values


def _val_proba_033(df_train: pd.DataFrame) -> np.ndarray:
    """Probabilidades en val set para exp-033 (CatBoost native)."""
    from sklearn.model_selection import train_test_split  # pylint: disable=import-outside-toplevel

    model = load_model(EXP_033_TAG)
    x, y = preprocess_native(df_train)
    _, x_val, _, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    return model.predict_proba(Pool(x_val, cat_features=CAT_FEATURES_NATIVE))[:, 1]


def main() -> None:
    """Orquesta el ensemble cross-feature-set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs017-models", nargs="+", default=DEFAULT_FS017_MODELS)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    all_tags = args.fs017_models + [EXP_033_TAG]
    weights = args.weights or [1.0] * len(all_tags)
    if len(weights) != len(all_tags):
        raise ValueError(f"--weights debe tener {len(all_tags)} valores.")
    w_arr = np.array(weights) / sum(weights)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("09_ensemble_cross_fs.py -- Soft Voting cross-feature-set")
    for tag, w in zip(all_tags, w_arr):
        print(f"  {tag} peso={w:.3f}")
    print(f"  Exp ID : {exp_id}")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_RAW)
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    scaler, target_encoder = load_feature_set_artifacts(FS_017)

    print("\n[THR] Calibrando umbral en val set...")
    probas_val, y_val = _val_proba_fs017(args.fs017_models, scaler, target_encoder)
    probas_val.append(_val_proba_033(df_train))
    ensemble_val = sum(w * p for w, p in zip(w_arr, probas_val))
    threshold, val_acc = optimize_threshold(y_val, ensemble_val)
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    print("\n[PRED] Generando predicciones en test...")
    fs017_models = [load_model(tag) for tag in args.fs017_models]
    x_tests = [
        preprocess_test(df_test, FEATURE_SETS[FS_017], get_feature_cols(m), scaler, target_encoder)
        for m in fs017_models
    ]
    x_test_native = preprocess_native(df_test, is_test=True)
    model_033 = load_model(EXP_033_TAG)

    probas_test = [m.predict_proba(x)[:, 1] for m, x in zip(fs017_models, x_tests)]
    probas_test.append(model_033.predict_proba(Pool(x_test_native, cat_features=CAT_FEATURES_NATIVE))[:, 1])
    predictions = (sum(w * p for w, p in zip(w_arr, probas_test)) >= threshold).astype(bool)

    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {"exp_id": exp_id, "type": "ensemble_soft_voting_cross_fs", "models": all_tags,
         "weights": dict(zip(all_tags, w_arr.tolist())),
         "threshold": threshold, "val_accuracy": val_acc},
        exp_id, "ensemble_cross_fs",
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(f"     {' + '.join(all_tags)} | umbral={threshold:.4f} | val_acc={val_acc:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | {len(predictions)-n_true} False")


if __name__ == "__main__":
    main()
