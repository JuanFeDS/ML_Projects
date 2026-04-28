"""
03_train_tabnet.py -- Entrenamiento TabNet

TabNet usa una red neuronal con mecanismo de atencion secuencial.
No es compatible con cross_val_score de sklearn (lento + necesita eval_set
para early stopping), por lo que usa un split train/val directo.

Ejecutar desde la raiz del proyecto:
    python scripts/03_train_tabnet.py
    python scripts/03_train_tabnet.py --feature-set fs-004_target_encoding
    python scripts/03_train_tabnet.py --n-iter 30
"""

import argparse
import json
import shutil

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config.settings import (
    DOCS_DIR,
    EXPERIMENTS_DIR,
    MODEL_METADATA,
    MODEL_PATH,
    PRODUCTION_DIR,
    SCALER_PATH,
    get_scaler_path,
    get_train_scaled,
)
from src.config.vcs import create_git_tag
from src.features.constants import TARGET
from src.features.feature_sets import FEATURE_SETS
from src.models.evaluation import analyze_errors, optimize_threshold
from src.models.inference import save_metadata
from src.models.training.tabnet_training import TabNetConfig, train_tabnet, tune_tabnet
from src.reports.experiments.log import ExperimentContext, append_experiment_log, get_next_exp_id
from src.reports.experiments.model_cards import write_experiment_card, write_model_card

DEFAULT_FS = "fs-004_target_encoding"


def main() -> None:
    """Entrena y tunea TabNet sobre el feature set indicado."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set", default=DEFAULT_FS, choices=list(FEATURE_SETS.keys())
    )
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    fs_name = args.feature_set
    print("=" * 60)
    print("03_train_tabnet.py -- TabNet")
    print(f"  Feature set : {fs_name}")
    print(
        f"  n_iter      : {args.n_iter} | max_epochs: {args.max_epochs} | patience: {args.patience}"
    )
    print("=" * 60)

    df = pd.read_csv(get_train_scaled(fs_name))
    y = df[TARGET]
    x = df.drop(columns=[TARGET])
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train: {x_train.shape[0]:,} | Val: {x_val.shape[0]:,}")

    print(f"\n[TUNE] Tuneando TabNet (n_iter={args.n_iter})...")
    tune_cfg = TabNetConfig(n_iter=args.n_iter, max_epochs=args.max_epochs, patience=args.patience)
    best_params, best_val_acc = tune_tabnet(x_train, y_train, (x_val, y_val), tune_cfg)
    print(f"  Mejor val_acc: {best_val_acc:.4f} | Params: {best_params}")

    print("\n[EVAL] Entrenando modelo final con val set...")
    train_cfg = TabNetConfig(max_epochs=args.max_epochs, patience=args.patience)
    final_model = train_tabnet(best_params, x_train, y_train, (x_val, y_val), train_cfg)
    y_pred_val = final_model.predict(x_val)
    y_proba_val = final_model.predict_proba(x_val)[:, 1]
    val_acc = float(accuracy_score(y_val, y_pred_val))
    val_roc = float(roc_auc_score(y_val, y_proba_val))
    print(f"  TabNet -> val_acc={val_acc:.4f} | roc_auc={val_roc:.4f}")

    error_tables = analyze_errors(
        x_val, y_val, pd.Series(y_pred_val, index=y_val.index)
    )
    best_threshold, threshold_acc = optimize_threshold(y_val, y_proba_val)
    effective_acc = max(val_acc, threshold_acc)
    effective_threshold = best_threshold if threshold_acc > val_acc else 0.5

    print("\n[FIT] Re-entrenando sobre dataset completo...")
    final_model_full = train_tabnet(best_params, x, y, config=train_cfg)

    log_path = str(DOCS_DIR / "model" / "experimentation_log.md")
    exp_id = get_next_exp_id(log_path)
    metadata = {
        "exp_id": exp_id,
        "model_name": "TabNet",
        "feature_set_name": fs_name,
        "feature_set_description": FEATURE_SETS[fs_name].description,
        "feature_set_parent": FEATURE_SETS[fs_name].parent,
        "features_added": [],
        "features_removed": [],
        "numeric_features": FEATURE_SETS[fs_name].numeric_features,
        "val_accuracy": effective_acc,
        "val_accuracy_default_threshold": val_acc,
        "val_roc_auc": val_roc,
        "cv_accuracy": best_val_acc,
        "n_features": x.shape[1],
        "n_train_samples": x.shape[0],
        "best_params": best_params,
        "best_threshold": effective_threshold,
        "feature_names": x.columns.tolist(),
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_TabNet.pkl"
    joblib.dump(final_model_full, exp_artifact)
    print(f"  [SAVE] Artefacto: {exp_artifact}")

    current_best_acc = None
    promoted = False
    if MODEL_METADATA.exists():
        import json as _json  # pylint: disable=import-outside-toplevel

        with open(MODEL_METADATA, encoding="utf-8") as f:
            current_best_acc = _json.load(f).get("val_accuracy")
    promoted = current_best_acc is None or effective_acc > current_best_acc

    if promoted:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model_full, MODEL_PATH)
        scaler_pkl_path = get_scaler_path(fs_name)
        if scaler_pkl_path.exists():
            shutil.copy2(scaler_pkl_path, SCALER_PATH)
        with open(MODEL_METADATA, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        label = "NUEVO MEJOR MODELO" if current_best_acc else "primer modelo"
        print(f"  [PROD] [{label}] Promovido: {MODEL_PATH}")
        create_git_tag(exp_id, fs_name, effective_acc)
    else:
        print(
            f"  [--] No promovido -- {effective_acc:.4f} no supera {current_best_acc:.4f}"
        )

    cards_dir = str(DOCS_DIR / "model" / "cards")
    append_experiment_log(
        metadata=metadata,
        path=log_path,
        exp_id=exp_id,
        promoted=promoted,
        context=ExperimentContext(current_best_acc=current_best_acc),
    )
    write_experiment_card(metadata=metadata, cards_dir=cards_dir, promoted=promoted)
    if promoted:
        write_model_card(metadata=metadata, docs_dir=str(DOCS_DIR / "model"))

    print("\n[OK] Pipeline TabNet completado.")


if __name__ == "__main__":
    main()
