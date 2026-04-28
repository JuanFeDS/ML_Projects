"""Renderización HTML de tarjetas de detalle por experimento."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from src.reports.builder import HTMLReport
from src.reports.experiments.charts import plot_trials_convergence

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
    fmt_map = {
        "iterations": ("iterations", lambda v: str(int(float(v)))),
        "depth": ("depth", lambda v: str(int(float(v)))),
        "learning_rate": ("lr", lambda v: f"{float(v):.4f}"),
        "l2_leaf_reg": ("l2", lambda v: f"{float(v):.3f}"),
        "bagging_temperature": ("bag_temp", lambda v: f"{float(v):.3f}"),
    }
    chips = []
    for k, v in params.items():
        label, fmt = fmt_map.get(k, (k, str))
        try:
            display = fmt(v)
        except Exception:  # pylint: disable=broad-except
            display = str(v)
        chips.append(f'<span class="exp-param-chip"><b>{label}</b> {display}</span>')
    return "".join(chips)


def render_experiment_card(
    html: HTMLReport,
    row: pd.Series,
    trials_df: pd.DataFrame,
    baseline_val: float,
    desc: dict,
) -> None:
    """Renderiza la tarjeta completa de un experimento en el HTMLReport.

    Args:
        html: Instancia de HTMLReport donde se inyecta el HTML.
        row: Fila del DataFrame de experimentos.
        trials_df: DataFrame de trials Optuna.
        baseline_val: Val accuracy del experimento baseline para calcular deltas.
        desc: Dict con claves 'what', 'hypothesis', 'result', 'tags'.
    """
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

    header = f"""
<div class="exp-card-header">
  <span class="exp-fs-badge {badge_cls}">{fs_label}</span>
  <span class="exp-fs-name">{fs_name.replace("_", " ").title()}</span>
  {'<span class="exp-model-chip">' + model + "</span>" if model else ""}
  <span class="exp-acc-pill {acc_cls}">Val {val_acc:.4f}</span>
</div>"""

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

    footer = f"""
<div class="exp-card-footer">
  <div class="exp-field-label" style="margin-top:0">Mejores hiperparametros</div>
  <div class="exp-params-grid">{_params_html(params)}</div>
</div>"""

    html.add_html(
        f'<div class="exp-card">{header}'
        f'<div class="exp-card-body">{narrative}{metrics_col}</div>'
        f"{footer}</div>"
    )

    fig = plot_trials_convergence(trials_df, row["run_id"])
    if fig is not None:
        html.add_figure(fig, title=f"Convergencia Optuna — {fs_label} {fs_name}")
