"""
Utilidades para el seguimiento de experimentos con MLflow.

Este modulo permite centralizar la configuracion de MLflow y provee 
funciones helper para loguear parametros, metricas y artefactos.
"""
import os
import mlflow
import mlflow.sklearn
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from contextlib import contextmanager

from src.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def setup_mlflow():
    """Configura el servidor de tracking de MLflow y el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

@contextmanager
def mlrun(run_name: str, nested: bool = False, tags: Optional[Dict[str, Any]] = None):
    """Context manager para crear un run de MLflow con soporte jerarquico.

    1. Si hay un run activo en el hilo actual, crea un run anidado (nested=True).
    2. Si existe la variable MLFLOW_PARENT_RUN_ID, se anida bajo ese ID.
    3. De lo contrario, crea un run de nivel superior.

    Args:
        run_name: Nombre visible del run.
        nested: Forzar comportamiento anidado (util para sub-etapas como tuning).
        tags: Diccionario de tags opcionales.
    """
    setup_mlflow()
    
    # 1. Comprobar si ya hay un run activo (vía código o herencia de run_pipeline.py)
    active_run = mlflow.active_run()
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    
    # Decidir si este run debe ser anidado
    should_be_nested = nested or active_run is not None or parent_run_id is not None
    
    if parent_run_id and not active_run:
        # Caso: Ejecución de sub-script heredando del orquestador global
        with mlflow.start_run(run_id=parent_run_id):
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                if tags:
                    mlflow.set_tags(tags)
                yield run
    else:
        # Caso: Ejecución aislada o ya dentro de un bloque 'with mlrun'
        with mlflow.start_run(run_name=run_name, nested=should_be_nested) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run

def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None):
    """Loguea un diccionario de metricas."""
    mlflow.log_metrics(metrics, step=step)

def log_params_dict(params: Dict[str, Any]):
    """Loguea un diccionario de parametros."""
    mlflow.log_params(params)

def log_plotly_figure(fig, filename: str):
    """Guarda una figura de Plotly como artefacto HTML en MLflow."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        fig.write_html(str(tmp_path))
        mlflow.log_artifact(str(tmp_path))

def log_model_metadata(metadata: Dict[str, Any], filename: str = "model_metadata.json"):
    """Guarda metadatos del modelo como artefacto JSON."""
    import json
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        mlflow.log_artifact(str(tmp_path))
