"""
03_train_tabnet.py -- Entrenamiento TabNet

TabNet usa una red neuronal con mecanismo de atencion secuencial.
No es compatible con cross_val_score de sklearn (lento + necesita eval_set
para early stopping), por lo que usa un split train/val directo
identico al del pipeline principal (test_size=0.2, random_state=42).

Ejecutar desde la raiz del proyecto:
    python scripts/03_train_tabnet.py
    python scripts/03_train_tabnet.py --feature-set fs-004_target_encoding
    python scripts/03_train_tabnet.py --n-iter 30
"""
import argparse
import json
import shutil

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
from src.features.constants import TARGET
from src.features.feature_sets import FEATURE_SETS
from src.models.tabnet_wrapper import TabNetWrapper
from src.models.training import analyze_errors, optimize_threshold
from src.pipelines.training_pipeline import _create_git_tag
from src.reports.experiments.log import (
    append_experiment_log,
    get_next_exp_id,
)
from src.reports.experiments.model_cards import write_experiment_card, write_model_card

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_FS = "fs-004_target_encoding"


def _tabnet_param_space(trial) -> dict:
    """Espacio de busqueda Optuna para TabNet."""
    n_d = trial.suggest_categorical("n_d", [16, 32, 64])
    return {
        "n_d": n_d,
        "n_a": n_d,  # n_a == n_d es la practica estandar
        "n_steps": trial.suggest_int("n_steps", 2, 6),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "n_independent": trial.suggest_int("n_independent", 1, 3),
        "n_shared": trial.suggest_int("n_shared", 1, 3),
        "momentum": trial.suggest_float("momentum", 0.01, 0.4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    }


def main() -> None:
    """Entrena y tunica TabNet sobre el feature set indicado."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FS,
        choices=list(FEATURE_SETS.keys()),
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Trials de Optuna (default: 25). Cada trial = 1 entrenamiento TabNet.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximo de epocas por entrenamiento (default: 200, con early stopping).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Paciencia del early stopping en epocas (default: 20).",
    )
    args = parser.parse_args()

    fs_name = args.feature_set

    print("=" * 60)
    print("03_train_tabnet.py -- TabNet")
    print(f"  Feature set : {fs_name}")
    print(f"  n_iter      : {args.n_iter}")
    print(f"  max_epochs  : {args.max_epochs}")
    print(f"  patience    : {args.patience}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cargar datos
    # ------------------------------------------------------------------
    print("\n[...] Cargando datos...")
    train_path = get_train_scaled(fs_name)
    df = pd.read_csv(train_path)
    print(f"  Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    y = df[TARGET]
    x = df.drop(columns=[TARGET])

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {x_train.shape[0]:,} | Val: {x_val.shape[0]:,}")

    # ------------------------------------------------------------------
    # 2. Tuning con Optuna (objetivo: val_accuracy con early stopping)
    # ------------------------------------------------------------------
    print(f"\n[TUNE] Tuneando TabNet (n_iter={args.n_iter})...")

    def objective(trial) -> float:
        params = _tabnet_param_space(trial)
        model = TabNetWrapper(
            max_epochs=args.max_epochs,
            patience=args.patience,
            virtual_batch_size=min(params["batch_size"] // 4, 256),
            **params,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        y_pred = model.predict(x_val)
        return float(accuracy_score(y_val, y_pred))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    with tqdm(total=args.n_iter, desc="Optuna (TabNet)", unit="trial") as pbar:
        def _cb(study, trial):  # pylint: disable=unused-argument
            pbar.update(1)
        study.optimize(objective, n_trials=args.n_iter, callbacks=[_cb])

    best_params = study.best_params
    best_val_acc_tuning = study.best_value
    print(f"  Mejor val_acc tuneado: {best_val_acc_tuning:.4f}")
    print(f"  Params: {best_params}")

    # ------------------------------------------------------------------
    # 3. Entrenar modelo final con los mejores params
    # ------------------------------------------------------------------
    print("\n[EVAL] Entrenando modelo final y evaluando en validacion...")
    final_model = TabNetWrapper(
        max_epochs=args.max_epochs,
        patience=args.patience,
        virtual_batch_size=min(best_params["batch_size"] // 4, 256),
        **best_params,
    )
    final_model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    y_pred_val = final_model.predict(x_val)
    y_proba_val = final_model.predict_proba(x_val)[:, 1]
    val_acc = float(accuracy_score(y_val, y_pred_val))
    val_roc = float(roc_auc_score(y_val, y_proba_val))
    print(f"  TabNet -> val_acc={val_acc:.4f} | roc_auc={val_roc:.4f}")

    # ------------------------------------------------------------------
    # 4. Error analysis y umbral optimo
    # ------------------------------------------------------------------
    print("\n[ERR] Analizando errores...")
    error_tables = analyze_errors(
        x_val, y_val, pd.Series(y_pred_val, index=y_val.index)
    )
    for seg, tbl in error_tables.items():
        print(f"\n  Error rate por {seg}:")
        print(tbl.to_string(index=False))

    print("\n[THR] Optimizando umbral de clasificacion...")
    best_threshold, threshold_acc = optimize_threshold(y_val, y_proba_val)
    gain = round(threshold_acc - val_acc, 4)
    print(f"  Umbral optimo: {best_threshold:.4f} -> val_accuracy: {threshold_acc:.4f} (ganancia: {gain:+.4f})")

    effective_acc = max(val_acc, threshold_acc)
    effective_threshold = best_threshold if threshold_acc > val_acc else 0.5

    # ------------------------------------------------------------------
    # 5. Re-entrenamiento sobre train completo
    # ------------------------------------------------------------------
    print("\n[FIT] Re-entrenando sobre x_train completo...")
    final_model_full = TabNetWrapper(
        max_epochs=args.max_epochs,
        patience=args.patience,
        virtual_batch_size=min(best_params["batch_size"] // 4, 256),
        **best_params,
    )
    # Re-entrenar con todo el train (sin eval_set para usar todas las epocas aprendidas)
    final_model_full.fit(x, y)

    # ------------------------------------------------------------------
    # 6. Artefactos y promocion
    # ------------------------------------------------------------------
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
        "cv_accuracy": best_val_acc_tuning,
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

    promoted = False
    current_best_acc = None
    if MODEL_METADATA.exists():
        with open(MODEL_METADATA, encoding="utf-8") as f:
            current_meta = json.load(f)
        current_best_acc = current_meta.get("val_accuracy")

    promoted = current_best_acc is None or effective_acc > current_best_acc

    scaler_pkl_path = get_scaler_path(fs_name)
    if promoted:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model_full, MODEL_PATH)
        if scaler_pkl_path.exists():
            shutil.copy2(scaler_pkl_path, SCALER_PATH)
        with open(MODEL_METADATA, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        label = "NUEVO MEJOR MODELO" if current_best_acc else "primer modelo"
        print(f"  [PROD] [{label}] Promovido a produccion: {MODEL_PATH}")
        _create_git_tag(exp_id, fs_name, effective_acc)
    else:
        print(
            f"  [--] No promovido -- val_accuracy {effective_acc:.4f} "
            f"no supera {current_best_acc:.4f} (produccion actual)"
        )

    append_experiment_log(
        metadata=metadata,
        path=log_path,
        exp_id=exp_id,
        promoted=promoted,
        current_best_acc=current_best_acc,
        cv_results=None,
        features_added=[],
        features_removed=[],
    )

    cards_dir = str(DOCS_DIR / "model" / "cards")
    write_experiment_card(metadata=metadata, cards_dir=cards_dir, promoted=promoted)
    if promoted:
        write_model_card(metadata=metadata, docs_dir=str(DOCS_DIR / "model"))

    print("\n[OK] Pipeline TabNet completado.")


if __name__ == "__main__":
    main()
