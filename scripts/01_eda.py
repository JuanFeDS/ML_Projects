"""
01_eda.py - Análisis Exploratorio de Datos (Receta Transparente)

Este script orquesta el análisis estadístico y la generación de reportes
para el dataset Spaceship Titanic, utilizando la lógica modular en src/.
"""
import sys
import pandas as pd

# Añadir la raíz del proyecto al path
sys.path.insert(0, ".")

from src.config.settings import TRAIN_RAW
from src.features.constants import TARGET
from src.models.tracking import mlrun
from src.pipelines.eda_pipeline import (
    run_basic_analysis,
    run_target_analysis,
    run_statistical_analysis,
    run_derived_analysis
)
from src.reports.eda_reports import build_eda_report

def main():
    """Ejecuta el flujo secuencial de EDA."""
    print("=" * 60)
    print("📊 01_eda.py — Orquestación del Exploratorio")
    print("=" * 60)

    # 1. Carga de datos
    df = pd.read_csv(TRAIN_RAW)
    print(f"✅ Datos cargados: {df.shape[0]:,} registros")

    # Iniciar seguimiento de MLflow (será child run si hay un parent activo)
    with mlrun(run_name="01_EDA") as run:
        
        # 2. Ejecutar análisis por bloques (Receta Transparente)
        print("\n🔍 Ejecutando análisis estadístico...")
        
        # Análisis Básico
        basic_res = run_basic_analysis(df)
        print("  - [X] Dimensiones, tipos y nulos")
        
        # Análisis del Target
        target_res = run_target_analysis(df)
        print(f"  - [X] Balance del target ({TARGET})")
        
        # Tests Estadísticos (Chi2, Mann-Whitney)
        stats_res = run_statistical_analysis(df)
        print("  - [X] Tests estadísticos de variables raw")
        
        # Features Derivadas
        derived_res = run_derived_analysis(df)
        print("  - [X] Impacto de features personalizadas")

        # 3. Consolidar resultados
        results = {
            "basic": basic_res,
            "target": target_res,
            "stats": stats_res,
            "derived": derived_res
        }

        # 4. Generación de Reportes Automáticos
        print("\n📄 Generando reportes (.md, .html)...")
        build_eda_report(df, results)
        
        # Registrar metadatos básicos en MLflow
        import mlflow
        mlflow.log_params({
            "rows": basic_res["shape"][0],
            "cols": basic_res["shape"][1],
            "target_balance": f"{target_res['pcts'].get(True, 0):.1f}%"
        })

    print("\n✅ EDA finalizado exitosamente.")

if __name__ == "__main__":
    main()
