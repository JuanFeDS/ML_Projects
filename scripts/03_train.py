"""
03_train.py -- Entrenamiento (receta transparente)

Ejecutar desde la raiz del proyecto:
    python scripts/03_train.py
    python scripts/03_train.py --feature-set fs-001_baseline --model LogisticRegression --no-tune --no-stack --no-moe
    python scripts/03_train.py --feature-set fs-002_cryo_interactions
"""
import argparse

from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS
from src.models.catalogue import MODELS
from src.pipelines.training_pipeline import run_training_pipeline
from src.reports.builder import ReportFactory


def main() -> None:
    """Orquesta el pipeline de entrenamiento y los reportes operacionales."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help="Feature set a usar (default: fs-001_baseline)",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODELS.keys()),
        help="Modelo especifico a entrenar. Si no se indica, compara todos.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Omitir tuning de hiperparametros (usa params por defecto).",
    )
    parser.add_argument(
        "--no-stack",
        action="store_true",
        help="Omitir construccion del Stacking ensemble.",
    )
    parser.add_argument(
        "--no-moe",
        action="store_true",
        help="Omitir construccion del Mixture of Experts.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Calcular y agregar analisis SHAP al reporte HTML.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Numero de trials de Optuna para el tuning (default: 25).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("03_train.py -- Entrenamiento")
    print(f"  Feature set : {args.feature_set}")
    print(f"  Modelo      : {args.model or 'todos (comparacion)'}")
    print(f"  Tuning      : {'no' if args.no_tune else 'si'}")
    print(f"  Stacking    : {'no' if args.no_stack else 'si'}")
    print(f"  MoE         : {'no' if args.no_moe else 'si'}")
    print(f"  SHAP        : {'si' if args.shap else 'no'}")
    print("=" * 60)

    results = run_training_pipeline(
        fs_name=args.feature_set,
        model_name=args.model,
        tune=not args.no_tune,
        build_stack=not args.no_stack,
        build_moe_flag=not args.no_moe,
        compute_shap=args.shap,
        n_iter=args.n_iter,
    )
    ReportFactory.emit_training_reports(results)

    print("\n[OK] Pipeline de entrenamiento completado.")


if __name__ == "__main__":
    main()
