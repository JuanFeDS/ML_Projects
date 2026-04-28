"""Visualizaciones Plotly para el reporte de seguimiento de experimentos."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go


def _color_scale(
    values: pd.Series, low: str = "#fca5a5", high: str = "#86efac"
) -> List[str]:
    """Genera colores interpolados entre low y high segun los valores normalizados."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return [high] * len(values)
    norm = (values - vmin) / (vmax - vmin)
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

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    df_sorted["rank_label"] = [medals.get(i, str(i + 1)) for i in range(len(df_sorted))]
    df_sorted["nota"] = df_sorted["anomaly"].apply(
        lambda a: "⚠ Data leakage" if a else "✓"
    )

    row_base = ["#ffffff" if i % 2 == 0 else "#f8fafc" for i in range(len(df_sorted))]

    normal_mask = ~df_sorted["anomaly"]
    normal_colors = _color_scale(
        df_sorted.loc[normal_mask, "val_accuracy"],
        low="#fee2e2",
        high="#bbf7d0",
    )
    val_colors: List[str] = []
    normal_iter = iter(normal_colors)
    for is_anomaly in df_sorted["anomaly"]:
        val_colors.append("#fed7aa" if is_anomaly else next(normal_iter))

    normal_colors_light = _color_scale(
        df_sorted.loc[normal_mask, "val_accuracy"],
        low="#fff1f1",
        high="#f0fdf4",
    )
    fs_colors: List[str] = []
    light_iter = iter(normal_colors_light)
    for is_anomaly in df_sorted["anomaly"]:
        fs_colors.append("#fff7ed" if is_anomaly else next(light_iter))

    nota_colors = ["#fff7ed" if a else "#f0fdf4" for a in df_sorted["anomaly"]]
    nota_font_colors = ["#c2410c" if a else "#15803d" for a in df_sorted["anomaly"]]

    cols_display = {
        "rank_label": "  #",
        "feature_set": "Feature Set",
        "val_accuracy": "Val Acc",
        "val_roc_auc": "Val AUC",
        "cv_accuracy": "CV Acc",
        "n_trials": "Trials",
        "best_trial_acc": "Best Trial",
        "git_commit": "Commit",
        "start_time": "Fecha",
        "nota": "Estado",
    }
    display_cols = [c for c in cols_display if c in df_sorted.columns]
    headers = [f"<b>{cols_display[c]}</b>" for c in display_cols]

    cell_values = []
    for col in display_cols:
        series = df_sorted[col]
        if col in ("val_accuracy", "val_roc_auc", "cv_accuracy", "best_trial_acc"):
            cell_values.append(
                [f"<b>{v:.4f}</b>" if pd.notna(v) else "—" for v in series]
            )
        elif col == "n_trials":
            cell_values.append([str(int(v)) if pd.notna(v) else "—" for v in series])
        else:
            cell_values.append(series.fillna("—").astype(str).tolist())

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
        "rank_label": 42,
        "feature_set": 190,
        "val_accuracy": 85,
        "val_roc_auc": 80,
        "cv_accuracy": 80,
        "n_trials": 58,
        "best_trial_acc": 90,
        "git_commit": 75,
        "start_time": 105,
        "nota": 110,
    }
    widths = [col_widths.get(c, 80) for c in display_cols]

    fig = go.Figure(
        data=[
            go.Table(
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
                        color=font_colors_per_col, size=12, family="Inter, monospace"
                    ),
                    align=["center"] + ["left"] * (len(display_cols) - 2) + ["center"],
                    height=34,
                    line=dict(color="#e2e8f0", width=0.5),
                ),
            )
        ]
    )
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
    fig.add_trace(
        go.Bar(
            name="Val Accuracy",
            x=short_names,
            y=df_sorted["val_accuracy"],
            marker_color=bar_colors,
            text=df_sorted["val_accuracy"].apply(
                lambda v: f"{v:.4f}" if pd.notna(v) else ""
            ),
            textposition="outside",
        )
    )
    if "cv_accuracy" in df_sorted.columns:
        cv_colors = (
            df_sorted["anomaly"].map({True: "#fdba74", False: "#a5b4fc"}).tolist()
        )
        fig.add_trace(
            go.Bar(
                name="CV Accuracy",
                x=short_names,
                y=df_sorted["cv_accuracy"],
                marker_color=cv_colors,
                opacity=0.8,
            )
        )

    best_val_clean = df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].max()
    fig.add_hline(
        y=best_val_clean,
        line_dash="dot",
        line_color="#10b981",
        annotation_text=f"Best (valido): {best_val_clean:.4f}",
        annotation_position="top right",
    )

    ymax = df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].max() + 0.05
    fig.update_layout(
        barmode="group",
        title_text="Progresion de Accuracy por Feature Set (naranja = posible data leakage)",
        xaxis_title="Feature Set",
        yaxis_title="Accuracy",
        yaxis_range=[
            max(0, df_sorted.loc[~df_sorted["anomaly"], "val_accuracy"].min() - 0.02),
            min(1.01, ymax),
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=440,
    )
    return fig


def plot_cv_vs_val(df: pd.DataFrame) -> go.Figure:
    """Scatter CV accuracy vs Val accuracy para detectar sobreajuste."""
    df_clean = df.dropna(subset=["cv_accuracy", "val_accuracy"]).copy()
    short_names = df_clean["feature_set"].str.replace(r"^fs-\d+_", "", regex=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
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
        )
    )

    diag_min = (
        min(df_clean["cv_accuracy"].min(), df_clean["val_accuracy"].min()) - 0.005
    )
    diag_max = (
        max(df_clean["cv_accuracy"].max(), df_clean["val_accuracy"].max()) + 0.005
    )
    fig.add_trace(
        go.Scatter(
            x=[diag_min, diag_max],
            y=[diag_min, diag_max],
            mode="lines",
            line=dict(dash="dot", color="#94a3b8", width=1),
            name="Val = CV (sin gap)",
            showlegend=True,
        )
    )

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
        fig.add_annotation(
            text="Sin datos de trials Optuna", showarrow=False, font=dict(size=14)
        )
        return fig

    map_col = (
        "root_training_id" if "root_training_id" in df.columns else "parent_run_id"
    )
    run_to_fs = training_df.set_index("run_id")["feature_set"].to_dict()
    df = df.copy()
    df["feature_set"] = df[map_col].map(run_to_fs).fillna("unknown")
    df["fs_short"] = df["feature_set"].str.replace(r"^fs-\d+_", "", regex=True)

    df_valid = df[df["feature_set"] != "unknown"].copy()
    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin trials mapeados a experimentos", showarrow=False)
        return fig

    order = (
        df_valid.groupby("fs_short")["trial_accuracy"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig = go.Figure()
    for name in order:
        subset = df_valid[df_valid["fs_short"] == name]
        fig.add_trace(
            go.Box(
                y=subset["trial_accuracy"],
                name=name,
                boxpoints="outliers",
                marker=dict(size=4),
            )
        )

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

    pred_clean = (
        pred_df.dropna(subset=["pct_transported"])
        .sort_values("start_time")
        .drop_duplicates(subset=["exp_id"], keep="last")
        .sort_values("exp_id")
    )

    fig = go.Figure(
        go.Bar(
            x=pred_clean["exp_id"].astype(str),
            y=pred_clean["pct_transported"],
            text=pred_clean["pct_transported"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            marker_color="#6366f1",
        )
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#ef4444",
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


def plot_trials_convergence(
    trials_df: pd.DataFrame, root_id: str
) -> Optional[go.Figure]:
    """Mini grafico de convergencia Optuna para un experimento especifico."""
    subset = trials_df[trials_df["root_training_id"] == root_id].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("trial_num").reset_index(drop=True)
    subset["running_best"] = subset["trial_accuracy"].cummax()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset["trial_num"],
            y=subset["trial_accuracy"],
            mode="markers",
            name="Trial",
            marker=dict(size=5, color="#a5b4fc", opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["trial_num"],
            y=subset["running_best"],
            mode="lines",
            name="Mejor acumulado",
            line=dict(color="#6366f1", width=2),
        )
    )
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
