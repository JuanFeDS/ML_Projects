"""
07_segmented_train.py - Entrenamiento segmentado por CryoSleep

Entrena un CatBoost independiente por segmento:
  - cryo:   CryoSleep=True  (~4,000 filas)
  - active: CryoSleep=False o NaN (~4,700 filas)

Cada modelo usa el pipeline de fs-017 (LastName fold-aware TE).
El experto cryo no recibe senal de gasto (todos 0) — el modelo aprende
exclusivamente de demografia, cabina y grupo.

Genera submission combinada enrutando cada pasajero de test por CryoSleep.

Ejecutar:
    python scripts/07_segmented_train.py
    python scripts/07_segmented_train.py --n-iter 50
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config.settings import (
    EXPERIMENTS_DIR,
    SUBMISSIONS_DIR,
    TEST_RAW,
    TRAIN_RAW,
    get_scaler_path,
    get_target_encoder_path,
)
from src.features.feature_sets import FEATURE_SETS
from src.models.evaluation import optimize_threshold
from src.models.inference import preprocess_test
from src.models.training import (
    MODELS,
    PARAM_SPACES,
    TuneConfig,
    add_model_prefix,
    make_fold_te_pipeline,
    prefix_param_space,
    tune_model,
)
from src.pipelines.data_pipeline import run_ingestion_to_features_pipeline
from src.reports.experiments.log import get_next_exp_id

FS_NAME = "fs-017_lastname_te"
FOLD_TE_COLS = ["LastName"]


def train_segment(df_seg, fs, cv, n_iter, seg_name, exp_id):
    """Feature engineering + tuning Optuna + OOF para un segmento.

    Args:
        df_seg: Subset de train.csv filtrado por segmento.
        fs: FeatureSetConfig de fs-017.
        cv: StratifiedKFold compartido.
        n_iter: Trials de Optuna.
        seg_name: Nombre del segmento ('cryo' o 'active').
        exp_id: ID del experimento (para nombrar artefactos).

    Returns:
        Dict con model, scaler, target_encoder, feature_names, metricas.
    """
    seg_fs_name = f"segment_{seg_name}"
    print(f"\n{'-'*50}")
    print(f"Segmento '{seg_name}': {len(df_seg):,} filas")
    print(f"{'-'*50}")

    results = run_ingestion_to_features_pipeline(df_seg, fs, seg_fs_name)
    X = results["X_scaled"]
    y = results["y"]
    print(f"  Features: {X.shape[1]} | Target rate: {y.mean():.3f}")

    pipeline = make_fold_te_pipeline(MODELS["CatBoost"], FOLD_TE_COLS)
    tuned, best_params, cv_score = tune_model(
        pipeline,
        X,
        y,
        cv,
        TuneConfig(
            prefix_param_space(PARAM_SPACES["CatBoost"]),
            n_iter=n_iter,
            param_transform=add_model_prefix,
        ),
    )
    print(f"  Mejor CV tuneado: {cv_score:.4f}")

    # OOF honesto (sin refit sobre los datos de validacion)
    oof_proba = cross_val_predict(
        tuned, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_acc = round(float(accuracy_score(y, (oof_proba >= 0.5).astype(int))), 4)
    oof_roc = round(float(roc_auc_score(y, oof_proba)), 4)
    threshold, thr_acc = optimize_threshold(y, oof_proba)
    print(
        f"  OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | thr={threshold:.4f} -> acc={thr_acc:.4f}"
    )

    # Refit final sobre el segmento completo
    tuned.fit(X, y)
    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_CatBoost_{seg_name}.pkl"
    joblib.dump(tuned, artifact)
    print(f"  Guardado: {artifact.name}")

    return {
        "model": tuned,
        "scaler": results["scaler"],
        "target_encoder": results["target_encoder"],
        "feature_names": list(X.columns),
        "oof_acc": oof_acc,
        "oof_roc": oof_roc,
        "threshold": threshold,
        "thr_acc": thr_acc,
        "best_params": best_params,
        "n_samples": len(X),
        "artifact": str(artifact),
        "seg_fs_name": seg_fs_name,
    }


def main() -> None:
    """Pipeline de entrenamiento segmentado y generacion de submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Trials de Optuna por segmento (default: 25).",
    )
    args = parser.parse_args()

    fs = FEATURE_SETS[FS_NAME]
    df_raw = pd.read_csv(TRAIN_RAW)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("07_segmented_train.py -- Entrenamiento segmentado CryoSleep")
    print(f"  Feature set : {FS_NAME}")
    print(f"  Exp ID      : {exp_id}")
    print(f"  n_iter      : {args.n_iter}")
    print("=" * 60)

    cryo_mask = df_raw["CryoSleep"].fillna(False).astype(bool)
    df_cryo = df_raw[cryo_mask].reset_index(drop=True)
    df_active = df_raw[~cryo_mask].reset_index(drop=True)
    print(
        f"\nSegmento cryo:   {len(df_cryo):,} filas ({100*len(df_cryo)/len(df_raw):.1f}%)"
    )
    print(
        f"Segmento active: {len(df_active):,} filas ({100*len(df_active)/len(df_raw):.1f}%)"
    )

    arts = {}
    for seg_name, df_seg in [("cryo", df_cryo), ("active", df_active)]:
        arts[seg_name] = train_segment(df_seg, fs, cv, args.n_iter, seg_name, exp_id)

    # OOF combinado ponderado por tamaño de segmento
    n_total = sum(a["n_samples"] for a in arts.values())
    combined_oof = sum(a["oof_acc"] * a["n_samples"] for a in arts.values()) / n_total

    print(f"\n{'='*60}")
    print(f"OOF combinado (ponderado): {combined_oof:.4f}")
    for seg, a in arts.items():
        print(
            f"  {seg}: OOF={a['oof_acc']:.4f} | ROC={a['oof_roc']:.4f} (n={a['n_samples']:,})"
        )
    print("=" * 60)

    # Guardar metadata del experimento segmentado
    meta = {
        "exp_id": exp_id,
        "fs_name": FS_NAME,
        "combined_oof_acc": round(combined_oof, 4),
        "segments": {
            seg: {
                "artifact": a["artifact"],
                "feature_names": a["feature_names"],
                "threshold": a["threshold"],
                "oof_acc": a["oof_acc"],
                "oof_roc": a["oof_roc"],
                "n_samples": a["n_samples"],
                "best_params": a["best_params"],
                "seg_fs_name": a["seg_fs_name"],
            }
            for seg, a in arts.items()
        },
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_segmented_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata guardado: {meta_path.name}")

    # --- Submission ---
    print("\nGenerando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    cryo_mask_test = df_test["CryoSleep"].fillna(False).astype(bool)

    predictions = np.zeros(len(df_test), dtype=bool)

    for seg_name, seg_mask in [("cryo", cryo_mask_test), ("active", ~cryo_mask_test)]:
        df_seg_test = df_test[seg_mask].copy()
        if df_seg_test.empty:
            continue
        a = arts[seg_name]
        seg_fs_name = a["seg_fs_name"]

        scaler = joblib.load(get_scaler_path(seg_fs_name))
        te_path = get_target_encoder_path(seg_fs_name)
        te = joblib.load(te_path) if te_path.exists() else None

        x_test_seg = preprocess_test(
            df_seg_test, fs, a["feature_names"], scaler, target_encoder=te
        )
        y_proba = a["model"].predict_proba(x_test_seg)[:, 1]
        predictions[np.where(seg_mask.values)[0]] = y_proba >= a["threshold"]

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids.values,
            "Transported": predictions,
        }
    )
    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    submission.to_csv(sub_path, index=False)

    n_true = int(submission["Transported"].sum())
    print(f"Submission guardado: {sub_path.name}")
    print(
        f"Distribucion: {n_true} True ({100*n_true/len(submission):.1f}%) | {len(submission)-n_true} False"
    )
    print("\n[OK] Entrenamiento segmentado completado.")


if __name__ == "__main__":
    main()
