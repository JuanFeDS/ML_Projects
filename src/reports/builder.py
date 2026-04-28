"""
Clases base para generacion de reportes Markdown y HTML.

MarkdownReport: construye un .md con metricas, tablas y texto.
HTMLReport: construye un .html con figuras Plotly embebidas y sidebar de navegacion.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import plotly.graph_objects as go
import plotly.io as pio

from src.reports.assets import _HTML_CSS, _HTML_JS, _LOGO_SVG


def _slugify(text: str) -> str:
    """Convierte un titulo en un ID valido para ancla HTML."""
    slug = text.lower()
    for src, dst in [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    ]:
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
                self._blocks.append("</div>")
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
            f"</div>"
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
        label_html = f'<div class="fig-label">{title}</div>' if title else ""
        self._blocks.append(f'<div class="fig-wrap">{label_html}{plotly_html}</div>')
        return self

    def add_html(self, raw_html: str) -> "HTMLReport":
        """Inserta HTML arbitrario directamente en el cuerpo del reporte.

        Args:
            raw_html: Fragmento HTML valido a insertar sin modificaciones.
        """
        self._blocks.append(raw_html)
        return self

    def add_image(self, b64_png: str, title: str = "") -> "HTMLReport":
        """Embebe una imagen PNG en base64 dentro del reporte.

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

        blocks = list(self._blocks)
        if self._in_section:
            blocks.append("</div>")

        toc_items = "\n".join(
            f'<a href="#{sid}" data-id="{sid}">'
            f'<span class="sb-num">{num}</span>'
            f"{stitle}</a>"
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
