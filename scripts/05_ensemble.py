"""
05_ensemble.py -- Soft Voting Ensemble

Promedia las probabilidades de prediccion de multiples modelos entrenados
sobre el mismo feature set. Sin meta-learner: solo el promedio ponderado
de las salidas de cada modelo.

Uso:
    python scripts/05_ensemble.py
    python scripts/05_ensemble.py --models exp-011 exp-012 exp-013
    python scripts/05_ensemble.py --models exp-011 exp-012 exp-013 --weights 1 1 2
    python scripts/05_ensemble.py --feature-set fs-004_target_encoding
"""
import argparse

import numpy as np
import pandas as pd

from src.config.settings import TEST_RAW
from src.features.feature_sets import FEATURE_SETS
from src.models.artifact_store import (
    align_features,
    load_feature_set_artifacts,
    load_model,
    load_val_set,
    save_metadata,
    save_submission,
)
from src.models.evaluation import optimize_threshold
from src.models.predict import preprocess_test
from src.reports.experiments.log import get_next_exp_id

DEFAULT_MODELS = ["exp-011", "exp-012", "exp-013"]
DEFAULT_FS = "fs-004_target_encoding"


def main() -> None:
    """Orquesta el ensemble de soft voting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--feature-set", default=DEFAULT_FS)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    fs_name = args.feature_set
    model_tags = args.models
    weights = args.weights or [1.0] * len(model_tags)
    if len(weights) != len(model_tags):
        raise ValueError("--weights debe tener el mismo numero de elementos que --models.")
    w_arr = np.array(weights, dtype=float) / sum(weights)

    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("05_ensemble.py -- Soft Voting Ensemble")
    print(f"  Feature set : {fs_name}")
    print(f"  Modelos     : {model_tags}")
    print(f"  Pesos       : {w_arr.tolist()}")
    print(f"  Exp ID      : {exp_id}")
    print("=" * 60)

    scaler, target_encoder = load_feature_set_artifacts(fs_name)
    models = [load_model(tag) for tag in model_tags]
    for tag, m in zip(model_tags, models):
        print(f"  [OK] {tag}: {type(m).__name__}")

    print("\n[THR] Optimizando umbral en val set...")
    x_val, y_val = load_val_set(fs_name)
    base_cols = x_val.columns.tolist()
    ensemble_val = sum(
        w * m.predict_proba(align_features(x_val, m, base_cols))[:, 1]
        for m, w in zip(models, w_arr)
    )
    threshold, val_acc = optimize_threshold(y_val, ensemble_val)
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    print("\n[PRED] Preprocesando y prediciendo en test...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    x_test = preprocess_test(df_test, FEATURE_SETS[fs_name], base_cols, scaler, target_encoder)
    ensemble_test = sum(
        w * m.predict_proba(align_features(x_test, m, base_cols))[:, 1]
        for m, w in zip(models, w_arr)
    )
    predictions = (ensemble_test >= threshold).astype(bool)

    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {"exp_id": exp_id, "type": "ensemble_soft_voting", "models": model_tags,
         "weights": dict(zip(model_tags, w_arr.tolist())),
         "threshold": threshold, "val_accuracy": val_acc},
        exp_id, "ensemble",
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(f"     Ensemble: {' + '.join(model_tags)} | umbral={threshold:.4f} | val_acc={val_acc:.4f}")
    print(f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | {len(predictions)-n_true} False")


if __name__ == "__main__":
    main()
