"""
Funciones para gestionar el log de experimentos en docs/model/experimentation_log.md.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        if f"| {model_name} |" not in section:
            continue
        if f"**val_accuracy:** {val_acc}" not in section:
            continue
        if f"**val_roc_auc:** {val_roc}" not in section:
            continue
        if fs_name and f"**nombre:** `{fs_name}`" not in section:
            continue
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

    lines = [f"\n## Exp-{exp_id} | {date_str} | {model_name} | {status}\n"]

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

    lines.append("\n### Modelo\n\n")
    lines.append(f"- **algoritmo:** {model_name}\n")
    best_params = metadata.get("best_params")
    if best_params:
        lines.append("- **hiperparametros optimos:**\n")
        for k, v in best_params.items():
            lines.append(f"  - `{k}`: {v}\n")

    if cv_results is not None:
        lines.append("\n### Cross-Validation — todos los modelos\n\n")
        cv_display = cv_results.reset_index().rename(columns={"index": "Modelo"})
        lines.append(cv_display.to_markdown(index=False) + "\n")

    lines.append("\n---\n")

    out.write_text(existing + "".join(lines), encoding="utf-8")
    print(f"Experiment log actualizado: {path}")
