"""
02_features.py - Ingeniería de Características (Receta Transparente)

Este script orquesta el pipeline de transformación, encoding y escalado
del dataset para un feature set seleccionado, delegando la lógica a src/.
"""
import argparse

import mlflow

from src.config.settings import get_train_features, get_train_scaled
from src.features.constants import TARGET
from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS
from src.models.tracking import mlrun
from src.pipelines.data_pipeline import load_raw_train, run_ingestion_to_features_pipeline
from src.reports.features.reports import build_feature_report


def main():
    """Ejecuta el flujo secuencial de feature engineering."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=list(FEATURE_SETS.keys()))
    fs_name = parser.parse_args().feature_set
    fs = FEATURE_SETS[fs_name]

    print("=" * 60)
    print(f"02_features.py -- Feature Engineering: {fs_name}")
    print("=" * 60)

    # 2. Carga de datos
    df_raw = load_raw_train()
    print(f"[OK] Datos crudos cargados: {df_raw.shape[0]:,} registros")

    # Iniciar seguimiento de MLflow (sera child run si hay un parent activo)
    with mlrun(f"02_Features_{fs_name}", tags={"stage": "features", "fs": fs_name}) as run:

        # 3. Ejecutar Pipeline de Caracteristicas (Modularizado)
        print("\n[...] Ejecutando transformaciones y encoding...")
        results = run_ingestion_to_features_pipeline(df_raw, fs, fs_name)

        X_raw = results["X_raw"]
        X_scaled = results["X_scaled"]
        y = results["y"]
        meta = results["metadata"]

        print(f"  - [X] Pipeline base de {fs_name} ejecutado")
        print(f"  - [X] Encoding (Label, Target, One-Hot) finalizado")
        print(f"  - [X] Escalado (StandardScaler) finalizado")
        print(f"  - [X] {meta['n_features']} features finales generadas")

        # 4. Guardar Datasets Resultantes
        print("\n[...] Guardando datasets procesados...")

        # Guardar Raw (X + y)
        train_features = X_raw.copy()
        train_features[TARGET] = y
        train_features.to_csv(get_train_features(fs_name), index=False)

        # Guardar Escalado (X_scaled + y)
        train_scaled = X_scaled.copy()
        train_scaled[TARGET] = y
        train_scaled.to_csv(get_train_scaled(fs_name), index=False)

        print(f"  - [X] Features crudas: {get_train_features(fs_name)}")
        print(f"  - [X] Features escaladas: {get_train_scaled(fs_name)}")

        # 5. Generacion de Reportes Automaticos
        print("\n[...] Generando reportes (.md, .html)...")
        build_feature_report(df_raw, results, fs_name, fs.description)

        # Tracking en MLflow
        mlflow.log_params({
            "fs_name": fs_name,
            "n_features": meta["n_features"],
            "n_samples": meta["n_samples"]
        })

    print("\n[OK] Feature Engineering finalizado exitosamente.")

if __name__ == "__main__":
    main()
