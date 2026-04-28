"""
11_ensemble_tabpfn.py -- Soft Voting Ensemble: CatBoost (fs-017) + CatBoost (native) + TabPFN

Combina tres modelos con diversidad arquitectural maxima:
  - exp-027: CatBoost sobre fs-017_lastname_te (features numericas + TE)
  - exp-033: CatBoost con categoricas nativas
  - exp-047: TabPFN foundation model (cloud API)

Uso:
    python scripts/11_ensemble_tabpfn.py
    python scripts/11_ensemble_tabpfn.py --weights 1 1 2
    python scripts/11_ensemble_tabpfn.py --tabpfn-tag exp-047
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()
import torch  # noqa: E402
from tabpfn_client import TabPFNClassifier, set_access_token  # noqa: E402

set_access_token(os.environ["TABPFN_TOKEN"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import Pool  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config.settings import TEST_RAW, TRAIN_RAW, get_train_scaled  # noqa: E402
from src.features.constants import TARGET  # noqa: E402
from src.features.feature_sets import FEATURE_SETS  # noqa: E402
from src.models.evaluation import optimize_threshold  # noqa: E402
from src.models.inference import (  # noqa: E402
    get_feature_cols,
    load_feature_set_artifacts,
    load_model,
    preprocess_test,
    save_metadata,
    save_submission,
)
from src.reports.experiments.log import get_next_exp_id  # noqa: E402
from src.preprocessing.common import (
    CAT_FEATURES_NATIVE,
    FS_NATIVE,
    preprocess_native_test,
    preprocess_native_train,
)  # noqa: E402

FS_017 = FS_NATIVE
EXP_027_TAG = "exp-027"
EXP_033_TAG = "exp-033"
DEFAULT_TABPFN_TAG = "exp-047"


def _val_probas(
    tags_027: list,
    tabpfn_tag: str,
    tabpfn_estimators: int,
    scaler,
    target_encoder,
    df_train: pd.DataFrame,
) -> tuple:
    """Calcula probabilidades en val set para los 3 modelos."""
    df_scaled = pd.read_csv(get_train_scaled(FS_017))
    y_all = df_scaled[TARGET]
    x_all = df_scaled.drop(columns=[TARGET])
    _, x_val_scaled, _, y_val = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    probas = []
    for tag in tags_027:
        m = load_model(tag)
        feat_cols = get_feature_cols(m)
        x_aligned = (
            x_val_scaled.reindex(columns=feat_cols, fill_value=0)
            if feat_cols
            else x_val_scaled
        )
        probas.append(m.predict_proba(x_aligned)[:, 1])

    x_native, y_native = preprocess_native_train(df_train)
    _, x_val_native, _, _ = train_test_split(
        x_native, y_native, test_size=0.2, stratify=y_native, random_state=42
    )
    model_033 = load_model(EXP_033_TAG)
    probas.append(
        model_033.predict_proba(Pool(x_val_native, cat_features=CAT_FEATURES_NATIVE))[
            :, 1
        ]
    )

    print(
        f"  TabPFN val: fit en {len(x_native) - len(x_val_native):,}, predict en {len(x_val_native):,}..."
    )
    x_tr_native, x_val_native2, y_tr_native, _ = train_test_split(
        x_native, y_native, test_size=0.2, stratify=y_native, random_state=42
    )
    model_tabpfn_val = TabPFNClassifier(n_estimators=tabpfn_estimators, random_state=42)
    model_tabpfn_val.fit(x_tr_native, y_tr_native)
    probas.append(model_tabpfn_val.predict_proba(x_val_native2)[:, 1])

    return probas, y_val.values


def main() -> None:
    """Orquesta el ensemble de 3 modelos con soft voting."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("W_027", "W_033", "W_047"),
    )
    parser.add_argument("--tabpfn-tag", default=DEFAULT_TABPFN_TAG)
    parser.add_argument("--tabpfn-estimators", type=int, default=8)
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
    scaler, target_encoder = load_feature_set_artifacts(FS_017)

    print("\n[THR] Calibrando umbral en val set...")
    probas_val, y_val = _val_probas(
        [EXP_027_TAG],
        args.tabpfn_tag,
        args.tabpfn_estimators,
        scaler,
        target_encoder,
        df_train,
    )
    threshold, val_acc = optimize_threshold(
        y_val, sum(w * p for w, p in zip(w_arr, probas_val))
    )
    print(f"  Umbral optimo: {threshold:.4f} -> val_accuracy: {val_acc:.4f}")

    print("\n[PRED] Generando predicciones en test...")
    model_027 = load_model(EXP_027_TAG)
    x_test_027 = preprocess_test(
        df_test,
        FEATURE_SETS[FS_017],
        get_feature_cols(model_027),
        scaler,
        target_encoder,
    )
    x_test_native = preprocess_native_test(df_test)
    model_033 = load_model(EXP_033_TAG)
    model_tabpfn = load_model(args.tabpfn_tag)

    p027_test = model_027.predict_proba(x_test_027)[:, 1]
    p033_test = model_033.predict_proba(
        Pool(x_test_native, cat_features=CAT_FEATURES_NATIVE)
    )[:, 1]
    p047_test = model_tabpfn.predict_proba(x_test_native)[:, 1]
    ensemble_test = w_arr[0] * p027_test + w_arr[1] * p033_test + w_arr[2] * p047_test
    predictions = (ensemble_test >= threshold).astype(bool)

    sub_path = save_submission(predictions, test_ids, exp_id)
    save_metadata(
        {
            "exp_id": exp_id,
            "type": "ensemble_soft_voting_tabpfn",
            "models": all_tags,
            "weights": dict(zip(all_tags, w_arr.tolist())),
            "threshold": threshold,
            "val_accuracy": val_acc,
        },
        exp_id,
        "ensemble_tabpfn",
    )

    n_true = int(predictions.sum())
    print(f"\n[OK] Submission: {sub_path.name}")
    print(
        f"     {' + '.join(all_tags)} | umbral={threshold:.4f} | val_acc={val_acc:.4f}"
    )
    print(
        f"     Distribucion: {n_true} True ({100*n_true/len(predictions):.1f}%) | {len(predictions)-n_true} False"
    )


if __name__ == "__main__":
    main()
