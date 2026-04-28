"""
06_experiment_report.py - Reporte HTML de seguimiento de experimentos.

Consulta el tracking store de MLflow y genera un reporte HTML interactivo
con leaderboard, progresion de accuracy, diagnostico de sobreajuste,
distribucion de trials Optuna y tasas de prediccion por submission.
"""

from pathlib import Path

from src.config.settings import REPORTS_DIR
from src.reports.experiments.tracking_report import build_tracking_report

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "spacechip_titanic"
OUTPUT_PATH = str(REPORTS_DIR / "experiments" / "experiment_tracking.html")


def main() -> None:
    print("=" * 60)
    print("[EXP] 06_experiment_report.py -- Reporte de experimentos")
    print("=" * 60)
    build_tracking_report(TRACKING_URI, OUTPUT_PATH, EXPERIMENT_NAME)
    print("[OK] Listo.")


if __name__ == "__main__":
    main()
