"""
Reporte HTML de seguimiento de experimentos desde MLflow.

Consulta la base de datos de MLflow, extrae runs de entrenamiento, Optuna trials
y predicciones, y genera un HTML interactivo usando el HTMLReport builder.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.reports.builder import HTMLReport


# ---------------------------------------------------------------------------
# Extraccion de datos desde MLflow
# ---------------------------------------------------------------------------

def _parse_fs_num(feature_set: str) -> int:
    """Extrae el numero de orden del nombre del feature set (fs-004 -> 4)."""
    match = re.search(r"fs-(\d+)", feature_set)
    return int(match.group(1)) if match else 999


def _parse_fs_from_run_name(run_name: str) -> Optional[str]:
    """Extrae el identificador de feature set del nombre del run (03_Train_fs-004_target_encoding)."""
    match = re.search(r"(fs-\d+_\S+)", run_name)
    return match.group(1) if match else None


def load_experiment_data(tracking_uri: str, experiment_name: str = "spacechip_titanic") -> Dict[str, Any]:
    """Consulta MLflow y retorna datos estructurados para el reporte.

    Args:
        tracking_uri: URI del servidor MLflow (ej. 'sqlite:///mlflow.db').
        experiment_name: Nombre del experimento MLflow.

    Returns:
        Dict con claves: training_df, trials_df, predictions_df, summary.
    """
    import mlflow  # pylint: disable=import-outside-toplevel
    from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Buscar el experimento por nombre
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado en {tracking_uri}")
    exp_id = exp.experiment_id

    all_runs = client.search_runs(exp_id, max_results=2000, order_by=["start_time ASC"])

    # Separar por tipo
    training_rows: List[Dict] = []
    trials_rows: List[Dict] = []
    prediction_rows: List[Dict] = []

    run_index = {r.info.run_id: r for r in all_runs}

    # Mapa global de parentesco: run_id -> parent_run_id
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
        params = run.data.params

        # Optuna trial (child run con trial_accuracy)
        if parent_id and "trial_accuracy" in metrics:
            # Resolver hasta el training run raiz
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
        # Extraer hiperparametros con prefijo bp_ (best params de Optuna)
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

    # Anotar numero de trials por training run raiz (root_training_id)
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


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def _color_scale(values: pd.Series, low: str = "#fca5a5", high: str = "#86efac") -> List[str]:
    """Genera colores interpolados entre low y high segun los valores normalizados."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return [high] * len(values)
    norm = (values - vmin) / (vmax - vmin)
    # Interpolacion simple en hex
    colors = []
    for n in norm:
        r = int(int(low[1:3], 16) + n * (int(high[1:3], 16) - int(low[1:3], 16)))
        g = int(int(low[3:5], 16) + n * (int(high[3:5], 16) - int(low[3:5], 16)))
        b = int(int(low[5:7], 16) + n * (int(high[5:7], 16) - int(low[5:7], 16)))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def plot_leaderboard(df: pd.DataFrame) -> go.Figure:
    """Tabla Plotly del leaderboard de experimentos, ordenada por val_accuracy desc."""
    df_sorted = df.sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    # ── Rank con medalla para el podio ──────────────────────────────────────
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    df_sorted["rank_label"] = [
        medals.get(i, str(i + 1)) for i in range(len(df_sorted))
    ]

    # ── Nota al final ────────────────────────────────────────────────────────
    df_sorted["nota"] = df_sorted["anomaly"].apply(
        lambda a: "⚠ Data leakage" if a else "✓"
    )

    # ── Colores fila ─────────────────────────────────────────────────────────
    # Filas alternas base: blanco / gris muy suave
    row_base = ["#ffffff" if i % 2 == 0 else "#f8fafc" for i in range(len(df_sorted))]

    # Val Accuracy: degradado por valor (excluye anomalias del rango)
    normal_mask = ~df_sorted["anomaly"]
    normal_colors = _color_scale(
        df_sorted.loc[normal_mask, "val_accuracy"],
        low="#fee2e2", high="#bbf7d0",
    )
    val_colors: List[str] = []
    normal_iter = iter(normal_colors)
    for is_anomaly in df_sorted["anomaly"]:
        val_colors.append("#fed7aa" if is_anomaly else next(normal_iter))

    # Feature Set: mismo tono que val pero mas tenue
    normal_colors_light = _color_scale(
        df_sorted.loc[normal_mask, "val_accuracy"],
        low="#fff1f1", high="#f0fdf4",
    )
    fs_colors: List[str] = []
    light_iter = iter(normal_colors_light)
    for is_anomaly in df_sorted["anomaly"]:
        fs_colors.append("#fff7ed" if is_anomaly else next(light_iter))

    # Nota: verde suave / ambar segun tipo
    nota_colors = ["#fff7ed" if a else "#f0fdf4" for a in df_sorted["anomaly"]]
    nota_font_colors = ["#c2410c" if a else "#15803d" for a in df_sorted["anomaly"]]

    # ── Definicion de columnas (nota al final) ───────────────────────────────
    cols_display = {
        "rank_label":     "  #",
        "feature_set":    "Feature Set",
        "val_accuracy":   "Val Acc",
        "val_roc_auc":    "Val AUC",
        "cv_accuracy":    "CV Acc",
        "n_trials":       "Trials",
        "best_trial_acc": "Best Trial",
        "git_commit":     "Commit",
        "start_time":     "Fecha",
        "nota":           "Estado",
    }
    display_cols = [c for c in cols_display if c in df_sorted.columns]
    headers = [f"<b>{cols_display[c]}</b>" for c in display_cols]

    # ── Valores de celda ─────────────────────────────────────────────────────
    cell_values = []
    for col in display_cols:
        series = df_sorted[col]
        if col in ("val_accuracy", "val_roc_auc", "cv_accuracy", "best_trial_acc"):
            cell_values.append([f"<b>{v:.4f}</b>" if pd.notna(v) else "—" for v in series])
        elif col == "n_trials":
            cell_values.append([str(int(v)) if pd.notna(v) else "—" for v in series])
        else:
            cell_values.append(series.fillna("—").astype(str).tolist())

    # ── Fill colors por columna ───────────────────────────────────────────────
    fill_colors = []
    font_colors_per_col = []
    for col in display_cols:
        if col == "val_accuracy":
            fill_colors.append(val_colors)
            font_colors_per_col.append(["#1e293b"] * len(df_sorted))
        elif col == "feature_set":
            fill_colors.append(fs_colors)
            font_colors_per_col.append(["#1e293b"] * len(df_sorted))
        elif col == "nota":
            fill_colors.append(nota_colors)
            font_colors_per_col.append(nota_font_colors)
        elif col == "rank_label":
            fill_colors.append(["#f1f5f9"] * len(df_sorted))
            font_colors_per_col.append(["#475569"] * len(df_sorted))
        else:
            fill_colors.append(row_base)
            font_colors_per_col.append(["#334155"] * len(df_sorted))

    col_widths = {
        "rank_label": 42, "feature_set": 190, "val_accuracy": 85,
        "val_roc_auc": 80, "cv_accuracy": 80, "n_trials": 58,
        "best_trial_acc": 90, "git_commit": 75, "start_time": 105, "nota": 110,
    }
    widths = [col_widths.get(c, 80) for c in display_cols]

    fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header=dict(
            values=headers,
            fill_color="#0f172a",
            font=dict(color="#e2e8f0", size=11, family="Inter, sans-serif"),
            align=["center"] + ["left"] * (len(display_cols) - 1),
            height=38,
            line=dict(color="#1e293b", width=1),
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(
                color=font_colors_per_col,
                size=12,
                family="Inter, monospace",
            ),
            align=["center"] + ["left"] * (len(display_cols) - 2) + ["center"],
            height=34,
            line=dict(color="#e2e8f0", width=0.5),
        ),
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        height=max(320, 44 + 36 * len(df_sorted)),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_accuracy_progression(df: pd.DataFrame) -> go.Figure:
    """Bar chart de val_accuracy y cv_accuracy por feature set, ordenado por fs_num."""
    df_sorted = df.sort_values("fs_num").copy()
    short_names = df_sorted["feature_set"].str.replace(r"fs-\d+_", "", regex=True)

    bar_colors = df_sorted["anomaly"].map({True: "#fb923c", False: "#6366f1"}).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Val Accuracy",
        x=short_names,
        y=df_sorted["val_accuracy"],
        marker_color=bar_colors,
        text=df_sorted["val_accuracy"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else ""),
        textposition="outside",
    ))
    if "cv_accuracy" in df_sorted.columns:
        cv_colors = df_sorted["anomaly"].map({True: "#fdba74", False: "#a5b4fc"}).tolist()
        fig.add_trace(go.Bar(
            name="CV Accuracy",
            x=short_names,
            y=df_sorted["cv_accuracy"],
            marker_color=cv_colors,
            opacity=0.8,
        ))

    # Linea de referencia sobre los runs validos (sin anomalias)
    best_val_clean = df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].max()
    fig.add_hline(
        y=best_val_clean, line_dash="dot", line_color="#10b981",
        annotation_text=f"Best (valido): {best_val_clean:.4f}",
        annotation_position="top right",
    )

    ymax = df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].max() + 0.05
    fig.update_layout(
        barmode="group",
        title_text="Progresion de Accuracy por Feature Set (naranja = posible data leakage)",
        xaxis_title="Feature Set",
        yaxis_title="Accuracy",
        yaxis_range=[max(0, df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].min() - 0.02),
                     min(1.01, ymax)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=440,
    )
    return fig


def plot_cv_vs_val(df: pd.DataFrame) -> go.Figure:
    """Scatter CV accuracy vs Val accuracy para detectar sobreajuste."""
    df_clean = df.dropna(subset=["cv_accuracy", "val_accuracy"]).copy()
    short_names = df_clean["feature_set"].str.replace(r"^fs-\d+_", "", regex=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_clean["cv_accuracy"],
        y=df_clean["val_accuracy"],
        mode="markers+text",
        text=short_names,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=10,
            color=df_clean["val_accuracy"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Val Acc"),
        ),
        hovertemplate="<b>%{text}</b><br>CV: %{x:.4f}<br>Val: %{y:.4f}<extra></extra>",
    ))

    # Linea diagonal (cv == val, sin gap)
    diag_min = min(df_clean["cv_accuracy"].min(), df_clean["val_accuracy"].min()) - 0.005
    diag_max = max(df_clean["cv_accuracy"].max(), df_clean["val_accuracy"].max()) + 0.005
    fig.add_trace(go.Scatter(
        x=[diag_min, diag_max], y=[diag_min, diag_max],
        mode="lines",
        line=dict(dash="dot", color="#94a3b8", width=1),
        name="Val = CV (sin gap)",
        showlegend=True,
    ))

    fig.update_layout(
        title_text="CV Accuracy vs Val Accuracy (diagnostico de sobreajuste)",
        xaxis_title="CV Accuracy",
        yaxis_title="Val Accuracy",
        height=420,
    )
    return fig


def plot_optuna_trials(df: pd.DataFrame, training_df: pd.DataFrame) -> go.Figure:
    """Box plot de distribucion de trial_accuracy por experimento."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos de trials Optuna", showarrow=False, font=dict(size=14))
        return fig

    # Mapear root_training_id -> feature_set (usa el run raiz, no el intermedio)
    map_col = "root_training_id" if "root_training_id" in df.columns else "parent_run_id"
    run_to_fs = training_df.set_index("run_id")["feature_set"].to_dict()
    df = df.copy()
    df["feature_set"] = df[map_col].map(run_to_fs).fillna("unknown")
    df["fs_short"] = df["feature_set"].str.replace(r"^fs-\d+_", "", regex=True)

    # Solo experimentos con trials
    df_valid = df[df["feature_set"] != "unknown"].copy()
    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin trials mapeados a experimentos", showarrow=False)
        return fig

    order = df_valid.groupby("fs_short")["trial_accuracy"].median().sort_values().index.tolist()

    fig = go.Figure()
    for name in order:
        subset = df_valid[df_valid["fs_short"] == name]
        fig.add_trace(go.Box(
            y=subset["trial_accuracy"],
            name=name,
            boxpoints="outliers",
            marker=dict(size=4),
        ))

    fig.update_layout(
        title_text="Distribucion de Trial Accuracy en busqueda Optuna",
        yaxis_title="Trial Accuracy",
        xaxis_title="Feature Set",
        showlegend=False,
        height=420,
    )
    return fig


def plot_submission_rates(pred_df: pd.DataFrame) -> go.Figure:
    """Bar chart de pct_transported por exp_id (deduplicado por exp_id)."""
    if pred_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos de prediccion", showarrow=False)
        return fig

    # Deduplicar por exp_id: conservar la prediccion mas reciente
    pred_clean = (
        pred_df.dropna(subset=["pct_transported"])
        .sort_values("start_time")
        .drop_duplicates(subset=["exp_id"], keep="last")
        .sort_values("exp_id")
    )

    fig = go.Figure(go.Bar(
        x=pred_clean["exp_id"].astype(str),
        y=pred_clean["pct_transported"],
        text=pred_clean["pct_transported"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        marker_color="#6366f1",
    ))
    fig.add_hline(
        y=50, line_dash="dot", line_color="#ef4444",
        annotation_text="50% (balance del dataset)",
    )
    fig.update_layout(
        title_text="Tasa de prediccion 'Transported' por experimento (submission)",
        xaxis_title="Exp ID",
        yaxis_title="% Transported predicho",
        yaxis_range=[40, 70],
        height=360,
    )
    return fig


# ---------------------------------------------------------------------------
# Tarjetas de detalle por experimento
# ---------------------------------------------------------------------------

_CARD_CSS = """
<style>
.exp-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  margin: 0 0 28px 0;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.exp-card-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 24px 16px;
  border-bottom: 1px solid #f1f5f9;
  background: linear-gradient(to right, #f8fafc, #ffffff);
}
.exp-fs-badge {
  font-size: 0.6875rem;
  font-weight: 700;
  color: #fff;
  background: #6366f1;
  padding: 3px 10px;
  border-radius: 99px;
  letter-spacing: 0.04em;
  white-space: nowrap;
  flex-shrink: 0;
}
.exp-fs-badge.anomaly { background: #f59e0b; }
.exp-fs-name {
  font-size: 1rem;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: -0.02em;
  flex: 1;
}
.exp-acc-pill {
  font-size: 0.8125rem;
  font-weight: 700;
  padding: 4px 14px;
  border-radius: 99px;
  color: #fff;
  background: #10b981;
  flex-shrink: 0;
}
.exp-acc-pill.anomaly { background: #f59e0b; }
.exp-model-chip {
  font-size: 0.75rem;
  font-weight: 500;
  color: #6366f1;
  background: #eef2ff;
  padding: 3px 10px;
  border-radius: 6px;
  flex-shrink: 0;
}
.exp-card-body {
  display: grid;
  grid-template-columns: 1fr 220px;
  gap: 0;
}
.exp-narrative {
  padding: 20px 24px;
  border-right: 1px solid #f1f5f9;
}
.exp-field-label {
  font-size: 0.625rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #94a3b8;
  margin: 14px 0 4px;
}
.exp-field-label:first-child { margin-top: 0; }
.exp-field-text {
  font-size: 0.875rem;
  color: #475569;
  line-height: 1.65;
  margin: 0;
}
.exp-metrics-col {
  padding: 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.exp-metric-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.exp-metric-key {
  font-size: 0.625rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}
.exp-metric-val {
  font-size: 1.125rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
}
.exp-metric-delta {
  font-size: 0.6875rem;
  font-weight: 600;
}
.exp-metric-delta.pos { color: #10b981; }
.exp-metric-delta.neg { color: #ef4444; }
.exp-metric-delta.neu { color: #94a3b8; }
.exp-card-footer {
  border-top: 1px solid #f1f5f9;
  padding: 16px 24px;
  background: #f8fafc;
}
.exp-params-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 0;
}
.exp-param-chip {
  font-size: 0.75rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 3px 10px;
  color: #334155;
}
.exp-param-chip b { color: #6366f1; }
.exp-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.exp-tag {
  font-size: 0.625rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #64748b;
  background: #f1f5f9;
  border-radius: 4px;
  padding: 2px 8px;
}
.exp-divider {
  height: 1px;
  background: #f1f5f9;
  margin: 0 24px;
}
</style>
"""


def _delta_html(val: Optional[float], baseline: float) -> str:
    """Genera badge de delta vs baseline."""
    if val is None:
        return ""
    delta = val - baseline
    cls = "pos" if delta > 0.001 else ("neg" if delta < -0.001 else "neu")
    sign = "+" if delta >= 0 else ""
    return f'<span class="exp-metric-delta {cls}">{sign}{delta:.4f} vs base</span>'


def _params_html(params: dict) -> str:
    """Genera chips de hiperparametros."""
    if not params:
        return "<span style='color:#94a3b8;font-size:0.8rem'>Sin hiperparametros registrados</span>"
    chips = []
    fmt_map = {
        "iterations": ("iterations", lambda v: str(int(float(v)))),
        "depth": ("depth", lambda v: str(int(float(v)))),
        "learning_rate": ("lr", lambda v: f"{float(v):.4f}"),
        "l2_leaf_reg": ("l2", lambda v: f"{float(v):.3f}"),
        "bagging_temperature": ("bag_temp", lambda v: f"{float(v):.3f}"),
    }
    for k, v in params.items():
        label, fmt = fmt_map.get(k, (k, str))
        try:
            display = fmt(v)
        except Exception:  # pylint: disable=broad-except
            display = str(v)
        chips.append(f'<span class="exp-param-chip"><b>{label}</b> {display}</span>')
    return "".join(chips)


def plot_trials_convergence(trials_df: pd.DataFrame, root_id: str) -> Optional[go.Figure]:
    """Mini grafico de convergencia Optuna para un experimento especifico."""
    subset = trials_df[trials_df["root_training_id"] == root_id].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("trial_num").reset_index(drop=True)
    subset["running_best"] = subset["trial_accuracy"].cummax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["trial_num"],
        y=subset["trial_accuracy"],
        mode="markers",
        name="Trial",
        marker=dict(size=5, color="#a5b4fc", opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=subset["trial_num"],
        y=subset["running_best"],
        mode="lines",
        name="Mejor acumulado",
        line=dict(color="#6366f1", width=2),
    ))
    fig.update_layout(
        margin=dict(l=30, r=10, t=20, b=30),
        height=200,
        showlegend=False,
        xaxis_title="Trial",
        yaxis_title="Accuracy",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    return fig


def _render_experiment_card(
    html: HTMLReport,
    row: pd.Series,
    trials_df: pd.DataFrame,
    baseline_val: float,
    desc: dict,
) -> None:
    """Renderiza la tarjeta completa de un experimento."""
    fs = row["feature_set"]
    fs_num = re.search(r"fs-(\d+)", fs)
    fs_label = f"fs-{fs_num.group(1)}" if fs_num else fs
    fs_name = re.sub(r"^fs-\d+_", "", fs)
    is_anomaly = bool(row.get("anomaly", False))

    badge_cls = "anomaly" if is_anomaly else ""
    acc_cls = "anomaly" if is_anomaly else ""
    val_acc = row["val_accuracy"]
    val_auc = row.get("val_roc_auc")
    cv_acc = row.get("cv_accuracy")
    model = row.get("winner_model", "")
    params = row.get("best_params", {}) or {}

    # Header
    header = f"""
<div class="exp-card-header">
  <span class="exp-fs-badge {badge_cls}">{fs_label}</span>
  <span class="exp-fs-name">{fs_name.replace("_", " ").title()}</span>
  {'<span class="exp-model-chip">' + model + "</span>" if model else ""}
  <span class="exp-acc-pill {acc_cls}">Val {val_acc:.4f}</span>
</div>"""

    # Narrativa
    what = desc.get("what", "Sin descripcion disponible.")
    hypothesis = desc.get("hypothesis", "")
    result = desc.get("result", "")
    tags = desc.get("tags", [])

    tags_html = "".join(f'<span class="exp-tag">{t}</span>' for t in tags)

    narrative = f"""
<div class="exp-narrative">
  <div class="exp-field-label">Que se hizo</div>
  <p class="exp-field-text">{what}</p>
  <div class="exp-field-label">Hipotesis</div>
  <p class="exp-field-text">{hypothesis}</p>
  <div class="exp-field-label">Resultado</div>
  <p class="exp-field-text">{result}</p>
  <div class="exp-tags">{tags_html}</div>
</div>"""

    # Metricas laterales
    delta_val = _delta_html(val_acc, baseline_val)
    delta_auc = _delta_html(val_auc, 0.8985) if val_auc else ""
    delta_cv = _delta_html(cv_acc, 0.8128) if cv_acc else ""

    metrics_col = f"""
<div class="exp-metrics-col">
  <div class="exp-metric-row">
    <span class="exp-metric-key">Val Accuracy</span>
    <span class="exp-metric-val">{val_acc:.4f}</span>
    {delta_val}
  </div>
  <div class="exp-metric-row">
    <span class="exp-metric-key">Val AUC-ROC</span>
    <span class="exp-metric-val">{f'{val_auc:.4f}' if val_auc else '—'}</span>
    {delta_auc}
  </div>
  <div class="exp-metric-row">
    <span class="exp-metric-key">CV Accuracy</span>
    <span class="exp-metric-val">{f'{cv_acc:.4f}' if cv_acc else '—'}</span>
    {delta_cv}
  </div>
  <div class="exp-metric-row">
    <span class="exp-metric-key">Trials Optuna</span>
    <span class="exp-metric-val">{int(row.get('n_trials', 0))}</span>
  </div>
</div>"""

    # Footer: params
    footer = f"""
<div class="exp-card-footer">
  <div class="exp-field-label" style="margin-top:0">Mejores hiperparametros</div>
  <div class="exp-params-grid">{_params_html(params)}</div>
</div>"""

    html.add_html(f'<div class="exp-card">{header}<div class="exp-card-body">{narrative}{metrics_col}</div>{footer}</div>')

    # Grafico de convergencia Optuna (si hay trials)
    fig = plot_trials_convergence(trials_df, row["run_id"])
    if fig is not None:
        html.add_figure(fig, title=f"Convergencia Optuna — {fs_label} {fs_name}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def build_tracking_report(tracking_uri: str, output_path: str,
                           experiment_name: str = "spacechip_titanic") -> None:
    """Genera el reporte HTML de seguimiento de experimentos.

    Args:
        tracking_uri: URI de MLflow (ej. 'sqlite:///mlflow.db').
        output_path: Ruta del archivo HTML de salida.
        experiment_name: Nombre del experimento en MLflow.
    """
    print("[...] Consultando MLflow...")
    data = load_experiment_data(tracking_uri, experiment_name)
    df = data["training_df"]
    trials_df = data["trials_df"]
    pred_df = data["predictions_df"]
    summary = data["summary"]

    if df.empty:
        print("[WARN] No se encontraron runs de entrenamiento en MLflow.")
        return

    print(f"[OK] {summary['total_experiments']} experimentos, "
          f"{summary['total_trials']} trials Optuna, "
          f"{summary['total_predictions']} predicciones.")

    html = HTMLReport(title="Seguimiento de Experimentos — Spaceship Titanic")

    # Header intro
    html.add_intro(
        f"Reporte de <b>{summary['total_experiments']} feature sets</b> evaluados "
        f"({summary['total_runs']} runs totales, {summary['total_trials']} trials Optuna). "
        f"Mejor resultado: <b>{summary['best_val_accuracy']:.4f} val_accuracy</b> "
        f"con <code>{summary['best_feature_set']}</code>."
    )

    html.add_metrics_grid([
        (summary["total_experiments"], "Feature Sets"),
        (summary["total_runs"], "Runs totales"),
        (summary["total_trials"], "Trials Optuna"),
        (f"{summary['best_val_accuracy']:.4f}", "Mejor Val Acc"),
        (summary["best_feature_set"].split("_")[0], "Mejor FS"),
        (summary["total_predictions"], "Submissions"),
    ])

    # Seccion 1: Leaderboard
    html.add_section("Leaderboard de Experimentos")
    html.add_text(
        "Tabla ordenada por <b>val_accuracy descendente</b>. "
        "El color de la columna Feature Set va de rojo (peor) a verde (mejor). "
        "Se muestra el mejor run por feature set."
    )
    if summary.get("n_anomalies", 0) > 0:
        anomaly_fs = df.loc[df["anomaly"], "feature_set"].tolist()
        html.add_callout(
            f"<b>Anomalia detectada ({summary['n_anomalies']} feature set(s)):</b> "
            f"<code>{', '.join(anomaly_fs)}</code> muestra una val_accuracy muy superior "
            "al resto del grupo (&gt;8 puntos sobre la mediana). Esto indica probable "
            "<b>fuga de datos</b> — las reglas de dominio codificaban el target de forma directa. "
            "La submission correspondiente confirma el problema: tasa de transporte anomalamente baja. "
            "Se excluye de las metricas de referencia del reporte.",
            kind="warning",
        )
    html.add_figure(plot_leaderboard(df), title="Ranking de Feature Sets")

    # Seccion 2: Progresion
    html.add_section("Progresion de Accuracy")
    html.add_text(
        "Evolucion del val_accuracy y cv_accuracy a lo largo de los experimentos, "
        "ordenados por numero de feature set. La linea discontinua marca el mejor resultado alcanzado."
    )
    html.add_figure(plot_accuracy_progression(df), title="Val Accuracy y CV Accuracy por Feature Set")

    # Seccion 3: CV vs Val
    html.add_section("Diagnostico de Sobreajuste")
    html.add_text(
        "Puntos por encima de la diagonal (val > cv) pueden indicar split favorecido. "
        "Puntos muy por debajo indican sobreajuste. Un buen modelo se mantiene cerca de la diagonal."
    )
    html.add_figure(plot_cv_vs_val(df), title="CV Accuracy vs Val Accuracy")

    # Seccion 4: Optuna trials
    html.add_section("Busqueda de Hiperparametros (Optuna)")
    html.add_text(
        f"Distribucion de <b>trial_accuracy</b> en los {summary['total_trials']} trials de busqueda "
        "Bayesiana. Cada caja representa un experimento con tuning. "
        "Una distribucion estrecha y alta indica que Optuna convergio bien."
    )
    html.add_figure(plot_optuna_trials(trials_df, data["all_training_df"]),
                    title="Distribucion de trial_accuracy por experimento")

    # Seccion 5: Submissions
    html.add_section("Tasas de Prediccion (Submissions)")
    html.add_text(
        "Porcentaje de pasajeros predichos como <i>Transported=True</i> en cada submission. "
        "El dataset real tiene ~50% de balance — desviaciones grandes pueden indicar "
        "calibracion incorrecta del modelo."
    )
    html.add_figure(plot_submission_rates(pred_df), title="% Transported predicho por experimento")

    # Seccion 6: Detalle por experimento
    from src.reports.experiments.descriptions import EXPERIMENT_DESCRIPTIONS  # pylint: disable=import-outside-toplevel

    html.add_section("Detalle por Experimento")
    html.add_text(
        "Tarjeta narrativa de cada feature set: que se hizo, hipotesis, resultado, "
        "hiperparametros encontrados por Optuna y curva de convergencia."
    )
    html.add_html(_CARD_CSS)

    # Baseline val_accuracy para calcular deltas
    baseline_row = df[df["feature_set"] == "fs-001_baseline"]
    baseline_val = float(baseline_row["val_accuracy"].iloc[0]) if not baseline_row.empty else 0.8285

    for _, row in df.sort_values("fs_num").iterrows():
        fs = row["feature_set"]
        desc = EXPERIMENT_DESCRIPTIONS.get(fs, {
            "what": "Descripcion pendiente.",
            "hypothesis": "Pendiente.",
            "result": "Pendiente.",
            "tags": [],
        })
        _render_experiment_card(html, row, trials_df, baseline_val, desc)

    html.save(output_path)
    print(f"[OK] Reporte generado en {output_path}")
