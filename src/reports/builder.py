"""
Clases base para generacion de reportes Markdown y HTML, y ReportFactory.

MarkdownReport: construye un .md con metricas, tablas y texto.
HTMLReport: construye un .html con figuras Plotly embebidas y sidebar de navegacion.
ReportFactory: dispatcher que emite parejas MD/HTML desde resultados de pipeline.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio


def _slugify(text: str) -> str:
    """Convierte un titulo en un ID valido para ancla HTML."""
    slug = text.lower()
    for src, dst in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n")]:
        slug = slug.replace(src, dst)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --indigo-50:  #eef2ff;
  --indigo-100: #e0e7ff;
  --indigo-400: #818cf8;
  --indigo-500: #6366f1;
  --indigo-600: #4f46e5;
  --indigo-900: #312e81;
  --slate-50:  #f8fafc;
  --slate-100: #f1f5f9;
  --slate-200: #e2e8f0;
  --slate-300: #cbd5e1;
  --slate-400: #94a3b8;
  --slate-500: #64748b;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1e293b;
  --slate-900: #0f172a;
  --emerald-500: #10b981;
  --amber-500: #f59e0b;
  --rose-500:  #ef4444;
  --sky-500:   #0ea5e9;
  --sidebar-w: 256px;
}
html { scroll-behavior: smooth; height: 100%; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--slate-100);
  color: var(--slate-900);
  display: flex;
  min-height: 100%;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ──────────────────────────────────────────────── */
.sidebar {
  width: var(--sidebar-w);
  min-height: 100vh;
  background: var(--slate-900);
  position: fixed;
  top: 0; left: 0; bottom: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  z-index: 100;
  border-right: 1px solid rgba(255,255,255,0.04);
}
.sidebar::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

.sb-brand {
  padding: 24px 20px 20px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sb-logo {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--indigo-500), var(--indigo-400));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 14px;
  box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}
.sb-logo svg { width: 18px; height: 18px; }
.sb-title {
  font-size: 0.8125rem;
  font-weight: 700;
  color: #e2e8f0;
  letter-spacing: -0.01em;
  line-height: 1.3;
}
.sb-sub {
  font-size: 0.6875rem;
  color: var(--slate-500);
  margin-top: 2px;
  font-weight: 400;
}

.sb-nav { padding: 16px 0; flex: 1; }
.sb-nav-label {
  padding: 10px 20px 5px;
  font-size: 0.625rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--slate-700);
}
.sb-nav a {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 20px;
  color: var(--slate-400);
  font-size: 0.8125rem;
  font-weight: 400;
  text-decoration: none;
  border-left: 2px solid transparent;
  transition: color 0.15s, background 0.15s, border-color 0.15s;
  line-height: 1.4;
}
.sb-nav a:hover {
  color: #e2e8f0;
  background: rgba(255,255,255,0.04);
  border-left-color: var(--indigo-400);
}
.sb-nav a.active {
  color: var(--indigo-400);
  border-left-color: var(--indigo-500);
  background: rgba(99,102,241,0.08);
  font-weight: 500;
}
.sb-num {
  font-size: 0.5625rem;
  font-weight: 700;
  background: var(--slate-800);
  color: var(--slate-500);
  width: 18px; height: 18px;
  border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  letter-spacing: 0;
}
.sb-nav a.active .sb-num {
  background: rgba(99,102,241,0.2);
  color: var(--indigo-400);
}
.sb-footer {
  padding: 14px 20px;
  border-top: 1px solid rgba(255,255,255,0.05);
  font-size: 0.625rem;
  color: var(--slate-700);
  line-height: 1.6;
}
.sb-footer strong { color: var(--slate-500); font-weight: 500; }

/* ── Main layout ──────────────────────────────────────────── */
.main {
  margin-left: var(--sidebar-w);
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

/* ── Header ───────────────────────────────────────────────── */
.rpt-header {
  background: linear-gradient(140deg, #1e1b4b 0%, var(--indigo-900) 40%, var(--indigo-600) 100%);
  padding: 48px 56px 44px;
  position: relative;
  overflow: hidden;
}
.rpt-header::before {
  content: '';
  position: absolute;
  top: -120px; right: -60px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(129,140,248,0.18) 0%, transparent 65%);
  pointer-events: none;
}
.rpt-header::after {
  content: '';
  position: absolute;
  bottom: -80px; left: 30%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 65%);
  pointer-events: none;
}
.rpt-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255,255,255,0.1);
  backdrop-filter: blur(4px);
  color: #c7d2fe;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 5px 14px;
  border-radius: 99px;
  margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.1);
  position: relative;
}
.rpt-header h1 {
  font-size: 2rem;
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.03em;
  line-height: 1.15;
  margin-bottom: 10px;
  position: relative;
}
.rpt-header-desc {
  color: rgba(255,255,255,0.55);
  font-size: 0.9375rem;
  line-height: 1.6;
  max-width: 560px;
  position: relative;
}
.rpt-header-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-top: 28px;
  position: relative;
}
.rpt-meta-item {
  display: flex;
  align-items: center;
  gap: 7px;
  color: rgba(255,255,255,0.45);
  font-size: 0.8rem;
}
.rpt-meta-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--indigo-400);
  flex-shrink: 0;
}

/* ── Body ─────────────────────────────────────────────────── */
.rpt-body {
  padding: 40px 56px 80px;
  max-width: 1100px;
}

/* ── Intro block ──────────────────────────────────────────── */
.rpt-intro {
  background: linear-gradient(to right, var(--indigo-50), var(--slate-50));
  border: 1px solid var(--indigo-100);
  border-left: 4px solid var(--indigo-500);
  border-radius: 10px;
  padding: 20px 24px;
  color: var(--slate-600);
  font-size: 0.9375rem;
  line-height: 1.75;
  margin-bottom: 32px;
}
.rpt-intro b { color: var(--slate-800); }

/* ── Metrics grid ─────────────────────────────────────────── */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 14px;
  margin: 4px 0 36px;
}
.metric-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px 22px 18px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 1px 4px rgba(0,0,0,0.04);
  border: 1px solid var(--slate-200);
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--indigo-500), var(--indigo-400));
}
.metric-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.1);
  transform: translateY(-1px);
}
.metric-card .mv {
  display: block;
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--slate-900);
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin-bottom: 7px;
}
.metric-card .ml {
  font-size: 0.6875rem;
  font-weight: 500;
  color: var(--slate-400);
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

/* ── Section blocks ───────────────────────────────────────── */
.section-block {
  margin-bottom: 44px;
}
h2 {
  display: flex;
  align-items: center;
  gap: 11px;
  font-size: 1.0625rem;
  font-weight: 700;
  color: var(--slate-900);
  letter-spacing: -0.02em;
  margin-bottom: 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--slate-200);
}
.h2-num {
  font-size: 0.6875rem;
  font-weight: 700;
  color: #fff;
  background: var(--indigo-500);
  min-width: 22px;
  height: 22px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  letter-spacing: 0;
}
h3 {
  font-size: 0.6875rem;
  font-weight: 700;
  color: var(--slate-400);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin: 22px 0 10px;
}

/* ── Figures ──────────────────────────────────────────────── */
.fig-wrap {
  background: #fff;
  border-radius: 12px;
  padding: 20px 20px 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  border: 1px solid var(--slate-200);
  margin: 14px 0 24px;
}
.fig-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--slate-700);
  margin-bottom: 14px;
}
.fig-label::before {
  content: '';
  display: block;
  width: 3px;
  height: 14px;
  background: var(--indigo-500);
  border-radius: 2px;
  flex-shrink: 0;
}
.plotly-graph-div { margin: 0 !important; }

/* ── Callouts ─────────────────────────────────────────────── */
.callout {
  border-radius: 8px;
  padding: 14px 18px;
  margin: 14px 0 20px;
  border-left: 4px solid;
  font-size: 0.9rem;
  line-height: 1.65;
}
.callout.info    { background: #eff6ff; border-color: var(--sky-500);     color: #1e40af; }
.callout.success { background: #f0fdf4; border-color: var(--emerald-500); color: #166534; }
.callout.warning { background: #fffbeb; border-color: var(--amber-500);   color: #92400e; }
.callout.danger  { background: #fef2f2; border-color: var(--rose-500);    color: #991b1b; }

/* ── Text ─────────────────────────────────────────────────── */
p {
  color: var(--slate-500);
  font-size: 0.9375rem;
  line-height: 1.7;
  margin-bottom: 12px;
}
b { color: var(--slate-700); }

/* ── Tables ───────────────────────────────────────────────── */
table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  border: 1px solid var(--slate-200);
  margin: 12px 0 20px;
  font-size: 0.8125rem;
}
thead { background: var(--slate-50); }
th {
  padding: 10px 16px;
  text-align: left;
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--slate-500);
  border-bottom: 1px solid var(--slate-200);
  white-space: nowrap;
}
td {
  padding: 9px 16px;
  color: var(--slate-600);
  border-bottom: 1px solid var(--slate-100);
  font-size: 0.8125rem;
}
tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: var(--slate-50); }
"""

_HTML_JS = """
(function () {
  const sections = document.querySelectorAll('.section-block');
  const navLinks = document.querySelectorAll('.sb-nav a[data-id]');
  if (!sections.length || !navLinks.length) return;

  const obs = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          navLinks.forEach(l => l.classList.remove('active'));
          const active = document.querySelector(`.sb-nav a[data-id="${entry.target.id}"]`);
          if (active) active.classList.add('active');
        }
      });
    },
    { rootMargin: '-10% 0px -60% 0px' }
  );
  sections.forEach(s => obs.observe(s));
})();
"""

_LOGO_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 2L2 7l10 5 10-5-10-5z"/>
  <path d="M2 17l10 5 10-5"/>
  <path d="M2 12l10 5 10-5"/>
</svg>"""


class HTMLReport:
    """Construye un reporte HTML con sidebar de navegacion y figuras Plotly embebidas."""

    def __init__(self, title: str):
        """
        Args:
            title: Titulo de la pagina HTML.
        """
        self.title = title
        self._blocks: List[str] = []
        self._sections: List[Tuple[str, str, int]] = []  # (id, title, num)
        self._section_count: int = 0
        self._in_section: bool = False
        self._first_figure: bool = True

    def add_section(self, title: str, level: int = 2) -> "HTMLReport":
        """Agrega un encabezado de seccion.

        Args:
            title: Texto del encabezado.
            level: Nivel HTML (2 = H2 principal con TOC, 3 = subseccion).
        """
        if level == 2:
            if self._in_section:
                self._blocks.append("</div>")  # cierra seccion anterior
            self._section_count += 1
            slug = _slugify(title)
            self._sections.append((slug, title, self._section_count))
            self._blocks.append(f'<div class="section-block" id="{slug}">')
            num_badge = f'<span class="h2-num">{self._section_count}</span>'
            self._blocks.append(f"<h2>{num_badge}{title}</h2>")
            self._in_section = True
        else:
            self._blocks.append(f"<h{level}>{title}</h{level}>")
        return self

    def add_text(self, text: str) -> "HTMLReport":
        """Agrega un parrafo de texto."""
        self._blocks.append(f"<p>{text}</p>")
        return self

    def add_intro(self, text: str) -> "HTMLReport":
        """Agrega un bloque de introduccion destacado."""
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
        """Embebe una figura Plotly dentro de un contenedor con estilo.

        Args:
            fig: Figura Plotly a embeber.
            title: Titulo descriptivo de la figura.
        """
        include_js = "cdn" if self._first_figure else False
        plotly_html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
        self._first_figure = False

        label_html = (
            f'<div class="fig-label">{title}</div>'
            if title else ""
        )
        self._blocks.append(
            f'<div class="fig-wrap">{label_html}{plotly_html}</div>'
        )
        return self

    def add_image(self, b64_png: str, title: str = "") -> "HTMLReport":
        """Embebe una imagen PNG en base64 dentro del reporte.

        Util para incrustar graficos matplotlib (ej. SHAP plots) sin
        depender de archivos externos.

        Args:
            b64_png: Cadena base64 del PNG (sin prefijo data URI).
            title: Titulo descriptivo de la imagen.
        """
        label_html = f'<div class="fig-label">{title}</div>' if title else ""
        self._blocks.append(
            f'<div class="fig-wrap">{label_html}'
            f'<img src="data:image/png;base64,{b64_png}" '
            f'style="max-width:100%;height:auto;" /></div>'
        )
        return self

    def save(self, path: str) -> None:
        """Guarda el reporte HTML en disco.

        Args:
            path: Ruta del archivo .html a escribir.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Cierra el ultimo bloque de seccion abierto
        blocks = list(self._blocks)
        if self._in_section:
            blocks.append("</div>")

        # Genera items del TOC en el sidebar
        toc_items = "\n".join(
            f'<a href="#{sid}" data-id="{sid}">'
            f'<span class="sb-num">{num}</span>'
            f'{stitle}</a>'
            for sid, stitle, num in self._sections
        )

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>{_HTML_CSS}</style>
</head>
<body>

<!-- ── Sidebar ──────────────────────────────────────── -->
<aside class="sidebar">
  <div class="sb-brand">
    <div class="sb-logo">{_LOGO_SVG}</div>
    <div class="sb-title">Spaceship Titanic</div>
    <div class="sb-sub">ML Project · Kaggle</div>
  </div>
  <nav class="sb-nav">
    <div class="sb-nav-label">Contenido</div>
    {toc_items}
  </nav>
  <div class="sb-footer">
    <strong>Generado</strong><br>{now}
  </div>
</aside>

<!-- ── Main ─────────────────────────────────────────── -->
<div class="main">
  <div class="rpt-header">
    <div class="rpt-tag">Análisis Exploratorio de Datos</div>
    <h1>{self.title}</h1>
    <div class="rpt-header-meta">
      <div class="rpt-meta-item">
        <span class="rpt-meta-dot"></span>
        Dataset: train.csv
      </div>
      <div class="rpt-meta-item">
        <span class="rpt-meta-dot"></span>
        Target: Transported
      </div>
      <div class="rpt-meta-item">
        <span class="rpt-meta-dot"></span>
        {now}
      </div>
    </div>
  </div>

  <div class="rpt-body">
    {"".join(blocks)}
  </div>
</div>

<script>{_HTML_JS}</script>
</body>
</html>"""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Reporte HTML guardado: {path}")


def _top_feature_names_for_insights(
    model: object, feature_names: List[str], k: int = 5
) -> List[str]:
    """Nombres de las k features con mayor importancia si el estimador las expone."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

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
        from src.reports.training.reports import build_training_html, build_training_md  # pylint: disable=import-outside-toplevel

        exp_id = results["metadata"]["exp_id"]
        winner_name = results["winner_name"]
        build_training_md(
            cv_results=results["cv_results"],
            best_name=results["best_name"],
            best_params=results["best_params"],
            tuned_val=results["tuned_val"],
            stacking_val=results["stacking_val"],
            moe_val=results["moe_val"],
            winner_name=winner_name,
            winner_val=results["winner_val"],
            top_names=results["top_names"],
            fs_name=results["fs_name"],
            error_tables=results["error_tables"],
            best_threshold=results["best_threshold"],
            threshold_acc=results["threshold_acc"],
            exp_id=exp_id,
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
            exp_id=exp_id,
            winner_name=winner_name,
            shap_plots=results.get("shap_plots", {}),
        )
