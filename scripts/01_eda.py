"""
01_eda.py - Analisis Exploratorio de Datos (Receta Transparente)

Este script orquesta el analisis estadistico y la generacion de reportes
para el dataset Spaceship Titanic, utilizando la logica modular en src/.
"""
import pandas as pd

import mlflow

from src.config.settings import TRAIN_RAW
from src.features.constants import TARGET
from src.models.tracking import mlrun
from src.pipelines.eda import (
    run_basic_analysis,
    run_target_analysis,
    run_statistical_analysis,
    run_derived_analysis,
    run_cabin_analysis,
    run_spending_analysis,
    run_domain_rules_validation,
    run_bivariate_analysis,
)
from src.reports.eda.reports import build_eda_report

def main():
    """Ejecuta el flujo secuencial de EDA."""
    print("=" * 60)
    print("[EDA] 01_eda.py -- Orquestacion del Exploratorio")
    print("=" * 60)

    # 1. Carga de datos
    df = pd.read_csv(TRAIN_RAW)
    print(f"[OK] Datos cargados: {df.shape[0]:,} registros")

    # Iniciar seguimiento de MLflow
    with mlrun(run_name="01_EDA") as run:

        # 2. Ejecutar analisis por bloques (Receta Transparente)
        print("\n[...] Ejecutando analisis estadistico...")

        basic_res = run_basic_analysis(df)
        print("  - [X] Dimensiones, tipos y nulos")

        target_res = run_target_analysis(df)
        print(f"  - [X] Balance del target ({TARGET})")

        stats_res = run_statistical_analysis(df)
        print("  - [X] Tests estadisticos de variables raw")

        derived_res = run_derived_analysis(df)
        print("  - [X] Impacto de features personalizadas")

        cabin_res = run_cabin_analysis(df)
        print("  - [X] Analisis de Cabin (Deck, Side, HomePlanet x Deck)")

        spending_res = run_spending_analysis(df)
        print("  - [X] Analisis de gasto por servicio y zero-inflation")

        domain_res = run_domain_rules_validation(df)
        print("  - [X] Validacion de reglas de dominio e imputacion por inferencia")

        bivariate_res = run_bivariate_analysis(df)
        print("  - [X] Analisis bivariado (CryoSleep x HomePlanet, Deck, Age)")

        # 3. Consolidar resultados
        results = {
            "basic": basic_res,
            "target": target_res,
            "stats": stats_res,
            "derived": derived_res,
            "cabin": cabin_res,
            "spending": spending_res,
            "domain_rules": domain_res,
            "bivariate": bivariate_res,
        }

        # 4. Generacion de Reportes Automaticos
        print("\n[...] Generando reportes (.md, .html)...")
        build_eda_report(df, results)

        # Registrar metadatos basicos en MLflow
        mlflow.log_params({
            "rows": basic_res["shape"][0],
            "cols": basic_res["shape"][1],
            "target_balance": f"{target_res['pcts'].get(True, 0):.1f}%"
        })

    print("\n[OK] EDA finalizado exitosamente.")

if __name__ == "__main__":
    main()
