"""
Utilidades para generacion de reportes Markdown y HTML.

MarkdownReport: construye un .md con metricas, tablas y texto.
HTMLReport: construye un .html con figuras Plotly embebidas.
Funciones standalone para actualizar docs/ desde el pipeline:
  write_data_quality_doc, write_model_card, write_experiment_card, append_experiment_log.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# ---------------------------------------------------------------------------
# Funciones standalone para actualizar docs/ desde el pipeline
# ---------------------------------------------------------------------------

def is_duplicate_experiment(metadata: dict, log_path: str) -> bool:
    """Devuelve True si ya existe un experimento identico en el log.

    Se considera duplicado cuando hay una entrada con el mismo model_name,
    val_accuracy, val_roc_auc y best_params. Evita registrar la misma
    corrida dos veces si se ejecuta el pipeline sin cambios.

    Args:
        metadata: Diccionario de metricas del experimento actual.
        log_path: Ruta del archivo experimentation_log.md.

    Returns:
        True si ya existe una entrada identica, False en caso contrario.
    """
    out = Path(log_path)
    if not out.exists():
        return False
    content = out.read_text(encoding="utf-8")

    model_name = metadata.get("model_name", "")
    val_acc = metadata.get("val_accuracy")
    val_roc = metadata.get("val_roc_auc")
    best_params = metadata.get("best_params")

    fs_name = metadata.get("feature_set_name")

    for section in content.split("\n## Exp-")[1:]:
        # El nombre del modelo aparece en la cabecera: "013 | fecha | ModelName | status"
        if f"| {model_name} |" not in section:
            continue
        if f"**val_accuracy:** {val_acc}" not in section:
            continue
        if f"**val_roc_auc:** {val_roc}" not in section:
            continue
        # El feature set aparece como "- **nombre:** `fs-xxx`" en el formato actual
        if fs_name and f"**nombre:** `{fs_name}`" not in section:
            continue
        # Si los hiperparametros difieren, es un experimento distinto
        if best_params:
            for k, v in best_params.items():
                if f"`{k}`: {v}" not in section:
                    break
            else:
                return True
            continue
        return True
    return False


def get_next_exp_id(log_path: str) -> str:
    """Devuelve el proximo ID de experimento sin escribir nada.

    Permite obtener el ID antes de entrenar, para nombrar el artefacto
    con el mismo ID que aparecera en el log y la card.

    Args:
        log_path: Ruta del archivo experimentation_log.md.

    Returns:
        ID como string zero-padded, e.g. '003'.
    """
    out = Path(log_path)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    is_template = not existing or "## Experimentos" in existing or "EXP-001" in existing
    if is_template:
        return "001"
    count = existing.count("\n## Exp-")
    return f"{count + 1:03d}"

def write_data_quality_doc(df: pd.DataFrame, target_col: str, path: str) -> None:
    """Sobreescribe docs/data/data_quality.md con metricas reales del dataset.

    Args:
        df: DataFrame raw cargado desde data/raw/train.csv.
        target_col: Nombre de la columna target.
        path: Ruta de destino del archivo .md.
    """
    md = MarkdownReport("Data Quality — Spaceship Titanic")

    md.add_section("Dimensiones")
    md.add_metric("Filas", f"{df.shape[0]:,}")
    md.add_metric("Columnas", df.shape[1])

    md.add_section("Tipos de datos")
    dtypes_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores unicos": [df[c].nunique() for c in df.columns],
    })
    md.add_table(dtypes_df, index=False)

    md.add_section("Valores nulos")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    nulls_df = (
        pd.DataFrame({
            "Columna": null_counts.index,
            "Nulos": null_counts.values,
            "% Nulos": null_pct.values,
        })
        .query("Nulos > 0")
        .reset_index(drop=True)
    )
    if nulls_df.empty:
        md.add_text("Sin valores nulos.")
    else:
        md.add_table(nulls_df, index=False)

    n_dupes = int(df.duplicated().sum())
    md.add_section("Duplicados")
    md.add_metric("Filas duplicadas", f"{n_dupes} ({n_dupes / len(df) * 100:.2f}%)")

    if target_col in df.columns:
        md.add_section(f"Balance del target ({target_col})")
        counts = df[target_col].value_counts()
        pcts = df[target_col].value_counts(normalize=True) * 100
        balance_df = pd.DataFrame({
            "Clase": counts.index.astype(str),
            "Conteo": counts.values,
            "% del total": pcts.values.round(2),
        })
        md.add_table(balance_df, index=False)

    md.add_section("Ultima actualizacion")
    md.add_metric("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M"))

    md.save(path)


def write_model_card(metadata: dict, feature_names: List[str], path: str) -> None:
    """Sobreescribe docs/model/model_card.md con los datos reales del modelo ganador.

    Args:
        metadata: Diccionario con metricas e hiperparametros del modelo.
        feature_names: Lista de nombres de features usadas en el modelo.
        path: Ruta de destino del archivo .md.
    """
    md = MarkdownReport("Model Card — Spaceship Titanic")

    md.add_section("Modelo")
    md.add_metric("Nombre", metadata.get("model_name", "—"))
    md.add_metric("Tipo", "Clasificacion binaria")
    md.add_metric("Target", "Transported (True/False)")
    md.add_metric("Numero de features", metadata.get("n_features", "—"))
    n_samples = metadata.get("n_train_samples")
    md.add_metric(
        "Muestras de entrenamiento",
        f"{n_samples:,}" if isinstance(n_samples, int) else "—",
    )

    md.add_section("Metricas de rendimiento")
    metrics_df = pd.DataFrame([
        {"Metrica": "Accuracy (validacion)", "Valor": metadata.get("val_accuracy", "—")},
        {"Metrica": "ROC-AUC (validacion)", "Valor": metadata.get("val_roc_auc", "—")},
        {"Metrica": "Accuracy (CV 5-fold)", "Valor": metadata.get("cv_accuracy", "—")},
    ])
    md.add_table(metrics_df, index=False)

    best_params = metadata.get("best_params")
    if best_params:
        md.add_section("Hiperparametros")
        md.add_code(
            "\n".join(f"{k}: {v}" for k, v in best_params.items()), lang=""
        )

    md.add_section("Features del modelo")
    md.add_bullet_list(feature_names)

    md.add_section("Validacion y artefactos")
    md.add_metric("Estrategia", "StratifiedKFold (5 folds) + hold-out 20%")
    md.add_metric("Fecha de entrenamiento", datetime.now().strftime("%Y-%m-%d"))
    md.add_metric("Archivo del modelo", "models/production/best_model.pkl")
    md.add_metric("Scaler", "models/production/scaler.pkl")

    md.save(path)


def write_experiment_card(
    metadata: dict,
    feature_names: List[str],
    exp_id: str,
    cards_dir: str,
    promoted: bool,
    current_best_acc: Optional[float] = None,
) -> None:
    """Escribe una model card individual para un experimento en docs/model/cards/.

    Se genera para TODOS los experimentos, no solo los exitosos.
    El archivo se guarda como exp-{exp_id}_{model_name}.md.

    Args:
        metadata: Diccionario con metricas e hiperparametros del modelo.
        feature_names: Lista de nombres de features usadas en el modelo.
        exp_id: ID del experimento (e.g. '003'). Obtener con get_next_exp_id().
        cards_dir: Directorio donde guardar las cards (e.g. 'docs/model/cards').
        promoted: True si el modelo supero al actual en produccion.
        current_best_acc: val_accuracy del modelo en produccion antes de este run.
    """
    model_name = metadata.get("model_name", "unknown")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"exp-{exp_id}_{safe_name}.md"

    md = MarkdownReport(f"Model Card — Exp-{exp_id} | {model_name}")

    md.add_section("Identificacion")
    md.add_metric("Experimento", f"Exp-{exp_id}")
    md.add_metric("Fecha", datetime.now().strftime("%Y-%m-%d %H:%M"))
    md.add_metric("Modelo", model_name)
    md.add_metric("Tipo", "Clasificacion binaria")
    md.add_metric("Target", "Transported (True/False)")

    md.add_section("Estado")
    status = "🏆 Promovido a produccion" if promoted else "❌ No supero al modelo actual"
    md.add_metric("Resultado", status)
    new_acc = metadata.get("val_accuracy")
    if current_best_acc is not None and isinstance(new_acc, float):
        diff = new_acc - current_best_acc
        sign = "+" if diff >= 0 else ""
        md.add_metric("val_accuracy este run", new_acc)
        md.add_metric("val_accuracy referencia", current_best_acc)
        md.add_metric("Diferencia", f"{sign}{diff:.4f}")
    elif current_best_acc is None:
        md.add_metric("Nota", "Primer experimento — sin referencia previa")
    model_path = (
        f"models/production/best_model.pkl"
        if promoted
        else f"models/experiments/exp-{exp_id}_{safe_name}.pkl"
    )
    md.add_metric("Artefacto", model_path)

    fs_name = metadata.get("feature_set_name")
    fs_description = metadata.get("feature_set_description")
    if fs_name:
        md.add_section("Feature Set")
        md.add_metric("Nombre", fs_name)
        if fs_description:
            md.add_metric("Descripcion", fs_description)

    md.add_section("Metricas de rendimiento")
    metrics_df = pd.DataFrame([
        {"Metrica": "Accuracy (validacion)", "Valor": metadata.get("val_accuracy", "—")},
        {"Metrica": "ROC-AUC (validacion)", "Valor": metadata.get("val_roc_auc", "—")},
        {"Metrica": "Accuracy (CV 5-fold)", "Valor": metadata.get("cv_accuracy", "—")},
    ])
    md.add_table(metrics_df, index=False)

    best_params = metadata.get("best_params")
    if best_params:
        md.add_section("Hiperparametros")
        md.add_code(
            "\n".join(f"{k}: {v}" for k, v in best_params.items()), lang=""
        )

    n_samples = metadata.get("n_train_samples")
    md.add_section("Dataset")
    md.add_metric("Features", metadata.get("n_features", "—"))
    md.add_metric(
        "Muestras de entrenamiento",
        f"{n_samples:,}" if isinstance(n_samples, int) else "—",
    )
    md.add_metric("Estrategia de validacion", "StratifiedKFold (5 folds) + hold-out 20%")

    md.add_section("Features del modelo")
    md.add_bullet_list(feature_names)

    out_dir = Path(cards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md.save(str(out_dir / filename))


def append_experiment_log(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    metadata: dict,
    path: str,
    exp_id: str,
    promoted: bool,
    current_best_acc: Optional[float] = None,
    cv_results: Optional[Any] = None,
    features_added: Optional[List[str]] = None,
    features_removed: Optional[List[str]] = None,
) -> None:
    """Agrega una nueva entrada detallada al log de experimentos.

    Registra TODOS los experimentos, exitosos o no, con un indicador de estado.
    Incluye tabla de CV, cambios de features vs parent y detalles del modelo.
    Si el archivo no existe o contiene plantilla, lo inicializa.

    Args:
        metadata: Diccionario con metricas e hiperparametros del modelo.
        path: Ruta del archivo experimentation_log.md.
        exp_id: ID del experimento, e.g. '003'. Debe obtenerse con get_next_exp_id().
        promoted: True si el modelo supero al actual en produccion.
        current_best_acc: val_accuracy del modelo en produccion antes de este run.
        cv_results: DataFrame con resultados de CV de todos los modelos.
        features_added: Lista de features anadidas vs el feature set parent.
        features_removed: Lista de features eliminadas vs el feature set parent.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    is_template = not existing or "## Experimentos" in existing or "EXP-001" in existing
    if is_template:
        existing = "# Experimentation Log — Spaceship Titanic\n\n"

    status = "🏆 Promovido a produccion" if promoted else "❌ No supero al modelo actual"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    model_name = metadata.get("model_name", "—")
    new_acc = metadata.get("val_accuracy", "—")
    n_samples = metadata.get("n_train_samples")
    n_samples_str = f"{n_samples:,}" if isinstance(n_samples, int) else "—"

    fs_name = metadata.get("feature_set_name", "—")
    fs_description = metadata.get("feature_set_description", "—")
    fs_parent = metadata.get("feature_set_parent", None)

    # --- Cabecera ---
    lines = [f"\n## Exp-{exp_id} | {date_str} | {model_name} | {status}\n"]

    # --- Metricas principales ---
    lines.append("\n### Metricas\n\n")
    acc_line = f"- **val_accuracy:** {new_acc}"
    if current_best_acc is not None and isinstance(new_acc, float):
        diff = new_acc - current_best_acc
        sign = "+" if diff >= 0 else ""
        acc_line += f"  _(ref: {current_best_acc}, {sign}{diff:.4f})_"
    lines.append(acc_line + "\n")
    lines += [
        f"- **val_roc_auc:** {metadata.get('val_roc_auc', '—')}\n",
        f"- **cv_accuracy (ganador):** {metadata.get('cv_accuracy', '—')}\n",
        f"- **n_features:** {metadata.get('n_features', '—')}\n",
        f"- **n_train_samples:** {n_samples_str}\n",
        f"- **artefacto:** `models/experiments/exp-{exp_id}_{model_name.replace(' ', '_')}.pkl`\n",
    ]

    # --- Feature Set ---
    lines.append("\n### Feature Set\n\n")
    lines += [
        f"- **nombre:** `{fs_name}`\n",
        f"- **parent:** `{fs_parent}`\n" if fs_parent else "- **parent:** ninguno (primer set)\n",
        f"- **descripcion:** {fs_description}\n",
    ]
    if features_added:
        lines.append(f"- **features anadidas vs parent ({len(features_added)}):** "
                     f"{', '.join(f'`{f}`' for f in features_added)}\n")
    if features_removed:
        lines.append(f"- **features eliminadas vs parent ({len(features_removed)}):** "
                     f"{', '.join(f'`{f}`' for f in features_removed)}\n")
    if not features_added and not features_removed and fs_parent:
        lines.append("- **cambios vs parent:** solo se modifico el tipo de encoding\n")

    # --- Modelo e hiperparametros ---
    lines.append("\n### Modelo\n\n")
    lines.append(f"- **algoritmo:** {model_name}\n")
    best_params = metadata.get("best_params")
    if best_params:
        lines.append("- **hiperparametros optimos:**\n")
        for k, v in best_params.items():
            lines.append(f"  - `{k}`: {v}\n")

    # --- Tabla CV de todos los modelos ---
    if cv_results is not None:
        lines.append("\n### Cross-Validation — todos los modelos\n\n")
        cv_display = cv_results.reset_index().rename(columns={"index": "Modelo"})
        lines.append(cv_display.to_markdown(index=False) + "\n")

    lines.append("\n---\n")

    out.write_text(existing + "".join(lines), encoding="utf-8")
    print(f"Experiment log actualizado: {path}")


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

    def add_table(self, df: pd.DataFrame, index: bool = False) -> "MarkdownReport":
        """
        Agrega un DataFrame como tabla Markdown.

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
        """
        Guarda el reporte en disco.

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
            pairs: Lista de tuplas (valor, etiqueta). El valor se muestra grande
                   y la etiqueta en uppercase pequeño debajo.
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
        """
        Embebe una figura Plotly. El CDN de Plotly.js se incluye solo
        en la primera figura para no duplicarlo.

        Args:
            fig: Figura Plotly a embeber.
            title: Titulo opcional para mostrar sobre la figura (H3).
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
        """
        Guarda el reporte HTML en disco.

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
