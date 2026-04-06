"""
Escritores de tarjetas de modelo y documentos de calidad de datos en docs/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.reports.builder import MarkdownReport


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
        "models/production/best_model.pkl"
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
