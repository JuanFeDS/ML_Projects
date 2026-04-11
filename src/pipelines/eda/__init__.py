"""
Pipeline de EDA para Spaceship Titanic.

Cada submódulo agrupa funciones compute_* y run_* de un dominio de análisis.
Importar desde aquí mantiene compatibilidad con scripts existentes.
"""
from src.pipelines.eda.basic import (
    run_basic_analysis,
    run_derived_analysis,
    run_statistical_analysis,
    run_target_analysis,
)
from src.pipelines.eda.bivariate import run_bivariate_analysis
from src.pipelines.eda.cabin import run_cabin_analysis
from src.pipelines.eda.domain_rules import run_domain_rules_validation
from src.pipelines.eda.spending import run_spending_analysis

__all__ = [
    "run_basic_analysis",
    "run_target_analysis",
    "run_statistical_analysis",
    "run_derived_analysis",
    "run_cabin_analysis",
    "run_spending_analysis",
    "run_domain_rules_validation",
    "run_bivariate_analysis",
]
