"""
Orquestador del pipeline de entrenamiento y seleccion de modelos.

Carga datos escalados, ejecuta CV / tuning / ensambles, gestiona artefactos
y documentacion en docs/. Los reportes operacionales (reports/) se emiten
via ReportFactory a partir del dict retornado.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, Optional

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config.settings import (
    DOCS_DIR,
    EXPERIMENTS_DIR,
    MODEL_METADATA,
    MODEL_PATH,
    PRODUCTION_DIR,
    REPORTS_DIR,
    SCALER_PATH,
    get_scaler_path,
    get_train_scaled,
)
from src.features.constants import TARGET
from src.features.feature_sets import FEATURE_SETS
from src.models.catalogue import MODELS, PARAM_SPACES
from src.models.tracking import mlrun
from src.models.training import (
    analyze_errors,
    build_moe,
    build_stacking,
    evaluate_models,
    evaluate_on_validation,
    optimize_threshold,
    tune_model,
)
from src.reports.experiments.log import (
    append_experiment_log,
    get_next_exp_id,
    is_duplicate_experiment,
)
from src.reports.experiments.model_cards import write_experiment_card, write_model_card


def _get_git_commit() -> str:
    """Retorna el hash corto del commit HEAD, o 'unknown' si falla."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pylint: disable=broad-except
        return "unknown"


def _create_git_tag(exp_id: str, fs_name: str, val_accuracy: float) -> None:
    """Crea un git tag para el experimento promovido."""
    tag = f"exp-{exp_id}_{fs_name}_{val_accuracy:.4f}"
    try:
        subprocess.run(
            ["git", "tag", tag], check=True, stderr=subprocess.DEVNULL
        )
        print(f"  [TAG] Git tag creado: {tag}")
    except subprocess.CalledProcessError:
        print(f"  [WARN] No se pudo crear git tag '{tag}' (ya existe?)")


def _log_mlflow_training_flat(fs_name: str, metadata: Dict[str, Any], winner_name: str) -> None:
    """Registra parametros y metricas escalares (MLflow no acepta dicts anidados)."""
    mlflow.set_tag("feature_set", fs_name)
    mlflow.set_tag("git_commit", _get_git_commit())
    mlflow.log_param("feature_set", fs_name)
    mlflow.log_param("winner_model", str(winner_name)[:250])
    mlflow.log_param("exp_id", str(metadata.get("exp_id", "")))
    mlflow.log_metric("val_accuracy", float(metadata.get("val_accuracy", 0.0)))
    mlflow.log_metric("val_roc_auc", float(metadata.get("val_roc_auc", 0.0)))
    mlflow.log_metric("cv_accuracy", float(metadata.get("cv_accuracy", 0.0)))
    bp = metadata.get("best_params") or {}
    for k, v in list(bp.items())[:40]:
        mlflow.log_param(f"bp_{k}", str(v)[:250])


def run_training_pipeline(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    fs_name: str,
    model_name: Optional[str] = None,
    tune: bool = True,
    build_stack: bool = True,
    build_moe_flag: bool = True,
    compute_shap: bool = False,
) -> Dict[str, Any]:
    """Ejecuta entrenamiento completo: modelos, artefactos, docs y tracking MLflow.

    Args:
        fs_name: Nombre del feature set (ej. fs-001_baseline).
        model_name: Si se indica, entrena solo ese modelo (sin comparacion multiple).
            Si es None, compara todos los modelos del catalogo y elige el mejor.
        tune: Si True, ejecuta tuning de hiperparametros con Optuna.
        build_stack: Si True, construye un StackingClassifier con los top-3 modelos.
        build_moe_flag: Si True, construye un Mixture of Experts.
        compute_shap: Si True, calcula SHAP values y agrega plots al reporte.

    Returns:
        Diccionario con modelos, metricas y tablas para ReportFactory y tests.
    """
    fs = FEATURE_SETS[fs_name]
    train_scaled_path = get_train_scaled(fs_name)
    scaler_pkl_path = get_scaler_path(fs_name)

    print("[...] Cargando datos...")
    df = pd.read_csv(train_scaled_path)
    print(f"  Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    y = df[TARGET]
    x_df = df.drop(columns=[TARGET])

    x_train, x_val, y_train, y_val = train_test_split(
        x_df, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {x_train.shape[0]} | Val: {x_val.shape[0]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlrun(f"03_Train_{fs_name}"):
        # ------------------------------------------------------------------
        # Seleccion de modelos a evaluar
        # ------------------------------------------------------------------
        _NOT_TUNABLE = {"Baseline"}

        models_to_eval = {model_name: MODELS[model_name]} if model_name else MODELS

        print(f"\n[CV] Evaluando modelos: {list(models_to_eval.keys())}")
        cv_results = evaluate_models(models_to_eval, x_train, y_train, cv)
        print(cv_results.to_string())

        best_name = model_name if model_name else next(
            n for n in cv_results.index
            if n not in _NOT_TUNABLE and n in PARAM_SPACES
        )
        best_cv_score = float(cv_results.loc[best_name, "cv_accuracy_mean"])
        print(f"  Modelo seleccionado: {best_name} | CV accuracy: {best_cv_score:.4f}")

        # ------------------------------------------------------------------
        # Tuning
        # ------------------------------------------------------------------
        best_params: dict = {}
        if tune and best_name in PARAM_SPACES:
            print(f"\n[TUNE] Tuneando {best_name} (n_iter=25)...")
            tuned_model, best_params, tuned_cv_score = tune_model(
                MODELS[best_name], PARAM_SPACES[best_name], x_train, y_train, cv, n_iter=25
            )
            print(f"  Mejor CV tuneado: {tuned_cv_score:.4f} | Params: {best_params}")
        else:
            tuned_model = MODELS[best_name]
            print(f"\n[SKIP] Tuning omitido -- usando params por defecto de {best_name}")

        # ------------------------------------------------------------------
        # Stacking (solo si se comparan multiples modelos)
        # ------------------------------------------------------------------
        stacking_model = None
        stacking_val: dict = {}
        top_names = [best_name]
        if build_stack and not model_name:
            print("\n[STACK] Construyendo Stacking...")
            top_names = [n for n in cv_results.index if n != "Baseline"][:3]
            print(f"  Base estimators: {top_names}")
            base_estimators = [(name, MODELS[name]) for name in top_names]
            stacking_model, stacking_cv_score = build_stacking(
                base_estimators, x_train, y_train, cv
            )
            stacking_val = evaluate_on_validation(
                stacking_model, x_train, y_train, x_val, y_val
            )
            print(f"  Stacking -> val_acc={stacking_val['val_accuracy']:.4f} | roc_auc={stacking_val['val_roc_auc']:.4f}")

        # ------------------------------------------------------------------
        # Mixture of Experts
        # ------------------------------------------------------------------
        moe_model = None
        moe_val: dict = {}
        if build_moe_flag:
            print("\n[MOE] Construyendo Mixture of Experts...")
            moe_model, moe_cv_score = build_moe(tuned_model, x_train, y_train, cv)
            sizes = moe_model.get_segment_sizes(x_train)
            moe_val = evaluate_on_validation(moe_model, x_train, y_train, x_val, y_val)
            print(f"  Segmento cryo: {sizes['cryo']:,} | activo: {sizes['active']:,}")
            print(f"  MoE -> val_acc={moe_val['val_accuracy']:.4f} | roc_auc={moe_val['val_roc_auc']:.4f}")

        # ------------------------------------------------------------------
        # Evaluacion en validacion del modelo principal (siempre)
        # ------------------------------------------------------------------
        print("\n[EVAL] Evaluando en validacion...")
        tuned_val = evaluate_on_validation(tuned_model, x_train, y_train, x_val, y_val)
        print(f"  {best_name} -> val_acc={tuned_val['val_accuracy']:.4f} | roc_auc={tuned_val['val_roc_auc']:.4f}")

        # ------------------------------------------------------------------
        # Seleccion del ganador
        # ------------------------------------------------------------------
        candidates = [(best_name, tuned_model, tuned_val)]
        if stacking_val:
            candidates.append(("Stacking", stacking_model, stacking_val))
        if moe_val:
            candidates.append(("MoE", moe_model, moe_val))

        winner_name, winner_model, winner_val = max(
            candidates, key=lambda t: t[2]["val_accuracy"]
        )
        print(f"\n[WIN] Modelo ganador: {winner_name} | val_acc={winner_val['val_accuracy']:.4f}")

        # ------------------------------------------------------------------
        # Analisis de errores y umbral optimo
        # ------------------------------------------------------------------
        print("\n[ERR] Analizando errores del modelo ganador...")
        y_pred_winner = winner_val["y_pred"]
        y_proba_winner = winner_val["y_proba"]
        error_tables = analyze_errors(
            x_val, y_val, pd.Series(y_pred_winner, index=y_val.index)
        )
        for seg, tbl in error_tables.items():
            print(f"\n  Error rate por {seg}:")
            print(tbl.to_string(index=False))

        print("\n[THR] Optimizando umbral de clasificacion...")
        best_threshold, threshold_acc = optimize_threshold(y_val, y_proba_winner)
        gain = round(threshold_acc - winner_val["val_accuracy"], 4)
        print(f"  Umbral optimo: {best_threshold:.4f} -> val_accuracy: {threshold_acc:.4f} (ganancia: {gain:+.4f})")

        # ------------------------------------------------------------------
        # Re-entrenamiento sobre train completo
        # ------------------------------------------------------------------
        print("\n[FIT] Re-entrenando sobre x_train completo...")
        winner_model.fit(x_train, y_train)

        # ------------------------------------------------------------------
        # Metadata del experimento
        # ------------------------------------------------------------------
        log_path = str(DOCS_DIR / "model" / "experimentation_log.md")
        exp_id = get_next_exp_id(log_path)
        feature_names = x_train.columns.tolist()

        features_added: list = []
        features_removed: list = []
        if fs.parent and fs.parent in FEATURE_SETS:
            parent_fs = FEATURE_SETS[fs.parent]
            current_cols = set(
                fs.numeric_features + fs.categorical_cols + list(fs.target_encode_cols)
            )
            parent_cols = set(
                parent_fs.numeric_features
                + parent_fs.categorical_cols
                + list(parent_fs.target_encode_cols)
            )
            features_added = sorted(current_cols - parent_cols)
            features_removed = sorted(parent_cols - current_cols)

        effective_acc = max(winner_val["val_accuracy"], threshold_acc)
        effective_threshold = (
            best_threshold if threshold_acc > winner_val["val_accuracy"] else 0.5
        )

        metadata = {
            "exp_id": exp_id,
            "model_name": winner_name,
            "feature_set_name": fs_name,
            "feature_set_description": fs.description,
            "feature_set_parent": fs.parent,
            "features_added": features_added,
            "features_removed": features_removed,
            "numeric_features": fs.numeric_features,
            "val_accuracy": effective_acc,
            "val_accuracy_default_threshold": winner_val["val_accuracy"],
            "val_roc_auc": winner_val["val_roc_auc"],
            "cv_accuracy": best_cv_score,
            "n_features": x_train.shape[1],
            "n_train_samples": x_train.shape[0],
            "best_params": best_params,
            "best_threshold": effective_threshold,
            "feature_names": feature_names,
        }

        # ------------------------------------------------------------------
        # Artefactos, log y card
        # ------------------------------------------------------------------
        promoted = False
        current_best_acc: float | None = None

        if is_duplicate_experiment(metadata, log_path) and MODEL_PATH.exists():
            print(
                "\n[SKIP] Experimento identico ya registrado -- "
                "se omiten artefacto, log y card."
            )
        else:
            EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = winner_name.replace(" ", "_").replace("(", "").replace(")", "")
            exp_artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_{safe_name}.pkl"
            joblib.dump(winner_model, exp_artifact)
            print(f"  [SAVE] Artefacto: {exp_artifact}")

            if MODEL_METADATA.exists():
                with open(MODEL_METADATA, encoding="utf-8") as f:
                    current_meta = json.load(f)
                current_best_acc = current_meta.get("val_accuracy")

            promoted = current_best_acc is None or effective_acc > current_best_acc

            if promoted:
                PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(winner_model, MODEL_PATH)
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

            cards_dir = str(DOCS_DIR / "model" / "cards")
            append_experiment_log(
                metadata=metadata,
                path=log_path,
                exp_id=exp_id,
                promoted=promoted,
                current_best_acc=current_best_acc,
                cv_results=cv_results,
                features_added=features_added,
                features_removed=features_removed,
            )
            write_experiment_card(
                metadata=metadata,
                feature_names=feature_names,
                exp_id=exp_id,
                cards_dir=cards_dir,
                promoted=promoted,
                current_best_acc=current_best_acc,
            )
            if promoted:
                write_model_card(
                    metadata=metadata,
                    feature_names=feature_names,
                    path=str(DOCS_DIR / "model" / "model_card.md"),
                )

        _log_mlflow_training_flat(fs_name, metadata, winner_name)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # SHAP (opcional)
        # ------------------------------------------------------------------
        shap_plots: dict = {}
        if compute_shap:
            from src.reports.training.shap_plots import compute_shap_plots  # pylint: disable=import-outside-toplevel
            print("\n[SHAP] Generando analisis SHAP...")
            shap_plots = compute_shap_plots(
                model=winner_model,
                x_train=x_train,
                x_val=x_val,
                y_val=y_val,
                y_proba=winner_val["y_proba"],
                feature_names=feature_names,
            )

        return {
            "fs_name": fs_name,
            "cv_results": cv_results,
            "best_name": best_name,
            "best_params": best_params,
            "tuned_val": tuned_val,
            "stacking_val": stacking_val,
            "moe_val": moe_val,
            "winner_name": winner_name,
            "winner_val": winner_val,
            "winner_model": winner_model,
            "feature_names": feature_names,
            "error_tables": error_tables,
            "best_threshold": best_threshold,
            "threshold_acc": threshold_acc,
            "top_names": top_names,
            "metadata": metadata,
            "promoted": promoted,
            "shap_plots": shap_plots,
        }
