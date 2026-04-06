"""Orquestación de etapas del pipeline (subprocesos + MLflow parent run)."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Tuple

import mlflow

from src.models.tracking import setup_mlflow
from src.config.settings import MLFLOW_PIPELINE_RUN_PREFIX

os.environ["PYTHONUTF8"] = "1"

# Etapas en orden: (nombre_corto, ruta_script)
PIPELINE_STAGES: List[Tuple[str, str]] = [
    ("01_eda", "scripts/01_eda.py"),
    ("02_features", "scripts/02_features.py"),
    ("03_train", "scripts/03_train.py"),
    ("04_predict", "scripts/04_predict.py"),
]

_FS_STAGE_NAMES = frozenset({"02_features", "03_train"})

def select_pipeline_scripts(
    *,
    skip_eda: bool = False,
    from_train: bool = False,
    predict_only: bool = False,
) -> List[Tuple[str, str]]:
    """Filtra etapas según flags.
    
    Args:
        skip_eda: Omite la etapa de EDA.
        from_train: Inicia desde la etapa de entrenamiento.
        predict_only: Ejecuta solo la etapa de predicción.
    
    Returns:
        Lista de tuplas (nombre_corto, ruta_script) filtradas.
    """
    stages = PIPELINE_STAGES
    if predict_only:
        return [stages[3]]
    if from_train:
        return [stages[2], stages[3]]
    if skip_eda:
        return [stages[1], stages[2], stages[3]]
    return list(stages)


def run_subprocess_stages(
    stages: List[Tuple[str, str]],
    feature_set: Optional[str],
    parent_run_id: Optional[str],
) -> None:
    """
    Ejecuta cada script en orden. Si parent_run_id está definido, 
    los runs MLflow hijos lo heredan.

    Args:
        stages: Lista de tuplas (nombre_corto, ruta_script) a ejecutar.
        feature_set: Nombre del conjunto de features a usar.
        parent_run_id: ID del run MLflow padre.
    """
    fs_args = ["--feature-set", feature_set] if feature_set else []
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    else:
        os.environ.pop("MLFLOW_PARENT_RUN_ID", None)

    for name, path in stages:
        extra = fs_args if name in _FS_STAGE_NAMES else []
        cmd = [sys.executable, path] + extra
        print(f'\n{"=" * 60}')
        print(f"  ▶  {name}")
        print(f'{"=" * 60}')
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n❌ [ERROR] {name} fallo con codigo {result.returncode}")
            sys.exit(1)

def run_pipeline_with_parent_run(
    stages: List[Tuple[str, str]],
    feature_set: Optional[str],
) -> None:
    """Un run MLflow padre envuelve todas las etapas listadas.
    
    Args:
        stages: Lista de tuplas (nombre_corto, ruta_script) a ejecutar.
        feature_set: Nombre del conjunto de features a usar.
    """

    fs_label = feature_set or "fs-001_baseline"
    setup_mlflow()
    run_name = f"{MLFLOW_PIPELINE_RUN_PREFIX}_{fs_label}"
    with mlflow.start_run(
        run_name=run_name,
        tags={"type": "pipeline", "fs": fs_label},
    ) as parent_run:
        run_subprocess_stages(stages, feature_set, parent_run.info.run_id)

    print(f'\n{"=" * 60}')
    print("  ✅ Pipeline completado exitosamente")
    print(f'{"=" * 60}\n')
