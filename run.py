"""Entrada unificada del proyecto Spaceship Titanic."""
import argparse

from src.models.tracking import setup_mlflow
from src.config.settings import MLFLOW_TRACKING_URI

from src.pipelines.orchestration import (
    PIPELINE_STAGES,
    run_pipeline_with_parent_run,
    run_subprocess_stages,
    select_pipeline_scripts,
)

def main() -> None:
    """
    CLI principal: una etapa o pipeline completo con run MLflow padre.
    """
    # Configuración del parser
    parser = argparse.ArgumentParser(
        description="""
            Spaceship Titanic — pipeline ML (entrada unificada)
        """
    )
    # Argumentos principales
    parser.add_argument(
        "--stage",
        choices=(
                "all", 
                "pipeline", 
                "eda", 
                "features", 
                "train", 
                "predict"
            ),
        default=None,
        help="Etapa individual a ejecutar.",
    )
    # Feature set
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Feature set para features y train (default: fs-001_baseline).",
    )
    # Flags de control
    parser.add_argument(
        "--skip-eda", 
        action="store_true",
        help="Omite 01_eda.py"
    )

    # Desde train
    parser.add_argument(
        "--from-train", 
        action="store_true",
        help="Ejecuta desde 03_train.py"
    )

    # Predict only
    parser.add_argument(
        "--predict-only", 
        action="store_true",
        help="Solo 04_predict.py"
    )

    # Init
    parser.add_argument(
        "--init", 
        action="store_true",
        help="Inicializa MLflow y sale"
    )
    args = parser.parse_args()

    if args.init:
        setup_mlflow()
        print("MLflow inicializado.")
        print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
        print(f"  UI           : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        return

    # Etapa individual explícita
    if args.stage and args.stage not in ("all", "pipeline"):
        stage_index = {"eda": 0, "features": 1, "train": 2, "predict": 3}[args.stage]
        run_subprocess_stages([PIPELINE_STAGES[stage_index]], args.feature_set, None)
        return

    # Pipeline con flags de filtrado o completo
    stages = select_pipeline_scripts(
        skip_eda=args.skip_eda,
        from_train=args.from_train,
        predict_only=args.predict_only,
    )
    run_pipeline_with_parent_run(stages, args.feature_set)

if __name__ == "__main__":
    main()
