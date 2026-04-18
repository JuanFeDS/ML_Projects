"""Extracción y parsing de datos de experimentos desde MLflow."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def _parse_fs_num(feature_set: str) -> int:
    """Extrae el numero de orden del nombre del feature set (fs-004 -> 4)."""
    match = re.search(r"fs-(\d+)", feature_set)
    return int(match.group(1)) if match else 999


def _parse_fs_from_run_name(run_name: str) -> Optional[str]:
    """Extrae el identificador de feature set del nombre del run."""
    match = re.search(r"(fs-\d+_\S+)", run_name)
    return match.group(1) if match else None


def load_experiment_data(tracking_uri: str, experiment_name: str = "spacechip_titanic") -> Dict[str, Any]:
    """Consulta MLflow y retorna datos estructurados para el reporte.

    Args:
        tracking_uri: URI del servidor MLflow (ej. 'sqlite:///mlflow.db').
        experiment_name: Nombre del experimento MLflow.

    Returns:
        Dict con claves: training_df, all_training_df, trials_df, predictions_df, summary.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado en {tracking_uri}")
    exp_id = exp.experiment_id

    all_runs = client.search_runs(exp_id, max_results=2000, order_by=["start_time ASC"])

    training_rows: List[Dict] = []
    trials_rows: List[Dict] = []
    prediction_rows: List[Dict] = []

    parent_map = {
        r.info.run_id: r.data.tags.get("mlflow.parentRunId", "")
        for r in all_runs
    }

    def _resolve_training_parent(run_id: str, depth: int = 0) -> str:
        """Sube por la jerarquia hasta encontrar el run raiz (sin parent)."""
        if depth > 5:
            return run_id
        pid = parent_map.get(run_id, "")
        if not pid:
            return run_id
        return _resolve_training_parent(pid, depth + 1)

    for run in all_runs:
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        parent_id = run.data.tags.get("mlflow.parentRunId", "")
        name = run.info.run_name or ""
        metrics = run.data.metrics

        # Optuna trial (child run con trial_accuracy)
        if parent_id and "trial_accuracy" in metrics:
            root_id = _resolve_training_parent(run.info.run_id)
            trials_rows.append({
                "parent_run_id": parent_id,
                "root_training_id": root_id,
                "trial_num": int(re.search(r"\d+", name).group()) if re.search(r"\d+", name) else -1,
                "trial_accuracy": metrics["trial_accuracy"],
                "run_id": run.info.run_id,
            })
            continue

        # Prediccion
        if tags.get("stage") == "predict":
            prediction_rows.append({
                "exp_id": tags.get("exp_id", "?"),
                "pct_transported": metrics.get("pct_transported", None),
                "n_predictions": int(metrics.get("n_predictions", 0)),
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
            })
            continue

        # Training run (padre)
        if "val_accuracy" not in metrics and "cv_accuracy" not in metrics:
            continue

        feature_set = (
            tags.get("feature_set")
            or _parse_fs_from_run_name(name)
            or "unknown"
        )

        start_ms = run.info.start_time or 0
        bp_params = {
            k.replace("bp_", ""): v
            for k, v in run.data.params.items()
            if k.startswith("bp_")
        }
        training_rows.append({
            "run_id": run.info.run_id,
            "run_name": name,
            "feature_set": feature_set,
            "fs_num": _parse_fs_num(feature_set),
            "val_accuracy": metrics.get("val_accuracy", None),
            "val_roc_auc": metrics.get("val_roc_auc", None),
            "cv_accuracy": metrics.get("cv_accuracy") or metrics.get("cv_accuracy_mean", None),
            "best_cv_accuracy": metrics.get("best_cv_accuracy", None),
            "winner_model": run.data.params.get("winner_model", ""),
            "exp_id": run.data.params.get("exp_id", ""),
            "best_params": bp_params,
            "git_commit": tags.get("git_commit", ""),
            "start_time": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            if start_ms else "",
        })

    training_df = pd.DataFrame(training_rows)
    trials_df = pd.DataFrame(trials_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    if training_df.empty:
        return {"training_df": training_df, "trials_df": trials_df,
                "predictions_df": predictions_df, "summary": {}}

    # Deduplicar: por feature_set, quedarse con el run de mejor val_accuracy
    best_idx = training_df.groupby("feature_set")["val_accuracy"].idxmax()
    best_df = training_df.loc[best_idx].copy().sort_values("fs_num").reset_index(drop=True)

    if not trials_df.empty:
        group_col = "root_training_id" if "root_training_id" in trials_df.columns else "parent_run_id"
        trial_counts = trials_df.groupby(group_col).agg(
            n_trials=("trial_accuracy", "count"),
            best_trial_acc=("trial_accuracy", "max"),
            mean_trial_acc=("trial_accuracy", "mean"),
        ).reset_index().rename(columns={group_col: "run_id"})
        best_df = best_df.merge(trial_counts, on="run_id", how="left")
    else:
        best_df["n_trials"] = 0
        best_df["best_trial_acc"] = None
        best_df["mean_trial_acc"] = None

    best_df["n_trials"] = best_df["n_trials"].fillna(0).astype(int)

    # Marcar anomalias: val_accuracy muy superior al resto sugiere fuga de datos
    acc_clean = best_df["val_accuracy"].dropna()
    median_acc = acc_clean.median()
    best_df["anomaly"] = best_df["val_accuracy"] > (median_acc + 0.08)

    summary = {
        "total_experiments": len(best_df),
        "total_runs": len(training_rows),
        "total_trials": len(trials_df),
        "best_val_accuracy": best_df.loc[~best_df["anomaly"], "val_accuracy"].max(),
        "best_feature_set": best_df.loc[
            ~best_df["anomaly"], "feature_set"
        ].iloc[best_df.loc[~best_df["anomaly"], "val_accuracy"].argmax()],
        "total_predictions": len(predictions_df),
        "n_anomalies": int(best_df["anomaly"].sum()),
    }

    return {
        "training_df": best_df,
        "all_training_df": training_df,
        "trials_df": trials_df,
        "predictions_df": predictions_df,
        "summary": summary,
    }
