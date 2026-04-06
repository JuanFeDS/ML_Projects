"""
Clases base para generacion de reportes Markdown y HTML, y ReportFactory.

MarkdownReport: construye un .md con metricas, tablas y texto.
HTMLReport: construye un .html con figuras Plotly embebidas.
ReportFactory: dispatcher que emite parejas MD/HTML desde resultados de pipeline.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio

class MarkdownReport:
    """Construye un reporte Markdown de forma incremental."""

    def __init__(self, title: str):
        """
        Args:
            title: Titulo principal del reporte (H1).
        """
        self._lines: List[str] = [f"# {title}\n"]

    def add_section(self, title: str) -> "MarkdownReport":
        """Agrega una seccion H2."""
        self._lines.append(f"\n## {title}\n")
        return self

    def add_subsection(self, title: str) -> "MarkdownReport":
        """Agrega una subseccion H3."""
        self._lines.append(f"\n### {title}\n")
        return self

    def add_text(self, text: str) -> "MarkdownReport":
        """Agrega un parrafo de texto."""
        self._lines.append(f"{text}\n")
        return self

    def add_metric(self, key: str, value: Any) -> "MarkdownReport":
        """Agrega una linea de metrica con formato `- **key:** value`."""
        self._lines.append(f"- **{key}:** {value}\n")
        return self

    def add_table(self, df: Any, index: bool = False) -> "MarkdownReport":
        """Agrega un DataFrame como tabla Markdown.

        Args:
            df: DataFrame a renderizar.
            index: Si True, incluye el indice como columna.
        """
        self._lines.append(df.to_markdown(index=index) + "\n")
        return self

    def add_bullet_list(self, items: List[Any]) -> "MarkdownReport":
        """Agrega una lista de vinetas."""
        for item in items:
            self._lines.append(f"- {item}\n")
        return self

    def add_code(self, code: str, lang: str = "python") -> "MarkdownReport":
        """Agrega un bloque de codigo con resaltado de sintaxis."""
        self._lines.append(f"```{lang}\n{code}\n```\n")
        return self

    def save(self, path: str) -> None:
        """Guarda el reporte en disco.

        Args:
            path: Ruta del archivo .md a escribir.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self._lines), encoding="utf-8")
        print(f"Reporte MD guardado: {path}")


_HTML_CSS = """
:root {
  --primary: #5b21b6;
  --primary-light: #7c3aed;
  --success: #059669;
  --warning: #d97706;
  --danger: #dc2626;
  --info: #0284c7;
  --bg: #f8fafc;
  --surface: #ffffff;
  --border: #e2e8f0;
  --text: #1e293b;
  --muted: #64748b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  max-width: 1200px;
  margin: 0 auto;
  padding-bottom: 60px;
  line-height: 1.6;
  font-size: 15px;
}
.rpt-header {
  background: linear-gradient(135deg, #1e1b4b 0%, var(--primary) 60%, var(--primary-light) 100%);
  color: white;
  padding: 40px 48px 36px;
  margin-bottom: 40px;
}
.rpt-header h1 {
  font-size: 1.875rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  margin-bottom: 6px;
}
.rpt-header .rpt-subtitle {
  opacity: 0.7;
  font-size: 0.875rem;
}
.rpt-body { padding: 0 48px; }
h2 {
  font-size: 1.0625rem;
  font-weight: 700;
  color: var(--text);
  margin: 36px 0 14px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--border);
  letter-spacing: -0.01em;
}
h3 {
  color: var(--muted);
  margin: 22px 0 10px;
  font-weight: 600;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
p { color: var(--muted); margin-bottom: 12px; font-size: 0.9375rem; }
b { color: var(--text); }
.rpt-intro {
  background: var(--surface);
  border-radius: 10px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  margin-bottom: 28px;
  border-left: 4px solid var(--primary-light);
  color: var(--muted);
  font-size: 0.9375rem;
  line-height: 1.75;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 12px;
  margin: 14px 0 28px;
}
.metric-card {
  background: var(--surface);
  border-radius: 10px;
  padding: 16px 18px 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  border-top: 3px solid var(--primary-light);
}
.metric-card .mv {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
  display: block;
  line-height: 1.2;
  margin-bottom: 5px;
}
.metric-card .ml {
  font-size: 0.71rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.callout {
  border-radius: 8px;
  padding: 13px 18px;
  margin: 14px 0 20px;
  border-left: 4px solid;
  font-size: 0.9rem;
  line-height: 1.6;
}
.callout.info    { background: #eff6ff; border-color: var(--info);    color: #1e40af; }
.callout.success { background: #f0fdf4; border-color: var(--success); color: #166534; }
.callout.warning { background: #fffbeb; border-color: var(--warning); color: #92400e; }
.callout.danger  { background: #fef2f2; border-color: var(--danger);  color: #991b1b; }
table {
  border-collapse: collapse;
  width: 100%;
  background: var(--surface);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  margin: 10px 0 20px;
  font-size: 0.84rem;
}
th {
  background: #f1f5f9;
  color: var(--text);
  font-weight: 600;
  padding: 9px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
td { padding: 8px 14px; border-bottom: 1px solid #f1f5f9; color: var(--muted); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
.plotly-graph-div { margin: 14px 0 6px; }
"""


class HTMLReport:
    """Construye un reporte HTML con figuras Plotly embebidas."""

    def __init__(self, title: str):
        """
        Args:
            title: Titulo de la pagina HTML.
        """
        self.title = title
        self._blocks: List[str] = []
        self._first_figure = True

    def add_section(self, title: str, level: int = 2) -> "HTMLReport":
        """Agrega un encabezado H{level} (por defecto H2)."""
        self._blocks.append(f"<h{level}>{title}</h{level}>")
        return self

    def add_text(self, text: str) -> "HTMLReport":
        """Agrega un parrafo de texto."""
        self._blocks.append(f"<p>{text}</p>")
        return self

    def add_intro(self, text: str) -> "HTMLReport":
        """Agrega un parrafo de introduccion con estilo destacado (borde izquierdo)."""
        self._blocks.append(f'<div class="rpt-intro">{text}</div>')
        return self

    def add_metrics_grid(self, pairs: List[Tuple[Any, str]]) -> "HTMLReport":
        """Renderiza un grid de tarjetas de metricas.

        Args:
            pairs: Lista de tuplas (valor, etiqueta).
        """
        cards = "".join(
            f'<div class="metric-card">'
            f'<span class="mv">{v}</span>'
            f'<span class="ml">{label}</span>'
            f'</div>'
            for v, label in pairs
        )
        self._blocks.append(f'<div class="metrics-grid">{cards}</div>')
        return self

    def add_callout(self, text: str, kind: str = "info") -> "HTMLReport":
        """Agrega una caja de aviso coloreada.

        Args:
            text: Contenido HTML del aviso.
            kind: Tipo visual — 'info', 'success', 'warning' o 'danger'.
        """
        self._blocks.append(f'<div class="callout {kind}">{text}</div>')
        return self

    def add_figure(self, fig: go.Figure, title: str = "") -> "HTMLReport":
        """Embebe una figura Plotly.

        El CDN de Plotly.js se incluye solo en la primera figura.

        Args:
            fig: Figura Plotly a embeber.
            title: Titulo opcional (H3).
        """
        if title:
            self._blocks.append(f"<h3>{title}</h3>")
        include_js = "cdn" if self._first_figure else False
        self._blocks.append(
            pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
        )
        self._first_figure = False
        return self

    def save(self, path: str) -> None:
        """Guarda el reporte HTML en disco.

        Args:
            path: Ruta del archivo .html a escribir.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = "\n".join([
            "<!DOCTYPE html><html lang='es'>",
            f"<head><meta charset='utf-8'><title>{self.title}</title>",
            f"<style>{_HTML_CSS}</style></head>",
            "<body>",
            f'<div class="rpt-header">'
            f'<h1>{self.title}</h1>'
            f'<div class="rpt-subtitle">Generado: {now}</div>'
            f'</div>',
            '<div class="rpt-body">',
            *self._blocks,
            "</div></body></html>",
        ])
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Reporte HTML guardado: {path}")


def _top_feature_names_for_insights(
    model: object, feature_names: List[str], k: int = 5
) -> List[str]:
    """Nombres de las k features con mayor importancia si el estimador las expone."""
    try:
        import numpy as np

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
            order = np.argsort(imp)[::-1][:k]
            return [feature_names[i] for i in order]
    except Exception:  # pylint: disable=broad-except
        pass
    return []


def build_training_insights_context(results: Dict[str, Any]) -> Dict[str, Any]:
    """Construye el dict esperado por get_training_insights desde run_training_pipeline."""
    cv_df = results["cv_results"]
    best_name = results["best_name"]
    cv_list = [
        {"model": str(idx), "cv_accuracy": float(row["cv_accuracy_mean"])}
        for idx, row in cv_df.iterrows()
    ]
    winner_val = results["winner_val"]
    tuned_val = results["tuned_val"]
    stacking_val = results["stacking_val"]
    moe_val = results["moe_val"]
    threshold_gain = float(results["threshold_acc"] - winner_val["val_accuracy"])
    return {
        "fs_name": results["fs_name"],
        "cv_results": cv_list,
        "best_model": best_name,
        "best_cv_accuracy": float(cv_df.loc[best_name, "cv_accuracy_mean"]),
        "tuned_val_accuracy": float(tuned_val["val_accuracy"]),
        "tuned_val_roc_auc": float(tuned_val["val_roc_auc"]),
        "stacking_val_accuracy": float(stacking_val["val_accuracy"]),
        "moe_val_accuracy": float(moe_val["val_accuracy"]),
        "winner_name": results["winner_name"],
        "winner_val_accuracy": float(winner_val["val_accuracy"]),
        "winner_val_roc_auc": float(winner_val["val_roc_auc"]),
        "best_threshold": float(results["best_threshold"]),
        "threshold_gain": threshold_gain,
        "top_features": _top_feature_names_for_insights(
            results["winner_model"], results["feature_names"], k=5
        ),
    }


class ReportFactory:
    """Genera parejas MD/HTML de reportes operacionales a partir de resultados de pipeline."""

    @staticmethod
    def emit_training_reports(results: Dict[str, Any]) -> None:
        """Escribe reports/03_training.md y reports/03_training.html.

        Args:
            results: Salida de src.pipelines.training_pipeline.run_training_pipeline.
        """
        from src.reports.training.reports import build_training_html, build_training_md

        build_training_md(
            cv_results=results["cv_results"],
            best_name=results["best_name"],
            best_params=results["best_params"],
            tuned_val=results["tuned_val"],
            stacking_val=results["stacking_val"],
            moe_val=results["moe_val"],
            winner_name=results["winner_name"],
            winner_val=results["winner_val"],
            top_names=results["top_names"],
            fs_name=results["fs_name"],
            error_tables=results["error_tables"],
            best_threshold=results["best_threshold"],
            threshold_acc=results["threshold_acc"],
        )
        build_training_html(
            cv_results=results["cv_results"],
            tuned_val=results["tuned_val"],
            stacking_val=results["stacking_val"],
            moe_val=results["moe_val"],
            best_name=results["best_name"],
            winner_model=results["winner_model"],
            feature_names=results["feature_names"],
            error_tables=results["error_tables"],
            best_threshold=results["best_threshold"],
            threshold_acc=results["threshold_acc"],
        )
