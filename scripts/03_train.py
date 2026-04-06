"""
03_train.py — Entrenamiento (receta transparente)

Ejecutar desde la raiz del proyecto:
    python scripts/03_train.py
    python scripts/03_train.py --feature-set fs-002_cryo_interactions
"""
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.pipelines.training_pipeline import run_training_pipeline
from src.reports.builder import ReportFactory
from src.utils.cli import parse_feature_set_args


def main() -> None:
    """Orquesta el pipeline de entrenamiento y los reportes operacionales."""
    args = parse_feature_set_args("Entrenamiento — Spaceship Titanic")
    fs_name = args.feature_set

    print("=" * 60)
    print("🤖 03_train.py — Entrenamiento")
    print(f"  Feature set: {fs_name}")
    print("=" * 60)

    results = run_training_pipeline(fs_name)
    ReportFactory.emit_training_reports(results, include_ai=False)

    print("\n✅ Pipeline de entrenamiento completado.")


if __name__ == "__main__":
    main()
