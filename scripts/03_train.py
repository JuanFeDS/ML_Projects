"""
03_train.py — Entrenamiento (receta transparente)

Ejecutar desde la raiz del proyecto:
    python scripts/03_train.py
    python scripts/03_train.py --feature-set fs-002_cryo_interactions
"""
import argparse

from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS
from src.pipelines.training_pipeline import run_training_pipeline
from src.reports.builder import ReportFactory


def main() -> None:
    """Orquesta el pipeline de entrenamiento y los reportes operacionales."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=list(FEATURE_SETS.keys()))
    fs_name = parser.parse_args().feature_set

    print("=" * 60)
    print("🤖 03_train.py — Entrenamiento")
    print(f"  Feature set: {fs_name}")
    print("=" * 60)

    results = run_training_pipeline(fs_name)
    ReportFactory.emit_training_reports(results)

    print("\n✅ Pipeline de entrenamiento completado.")


if __name__ == "__main__":
    main()
