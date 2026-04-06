"""
Utilidades de línea de comandos compartidas entre scripts del pipeline.
"""
import argparse

from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS


def add_feature_set_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Añade el argumento --feature-set a un ArgumentParser existente.

    Args:
        parser: ArgumentParser al que se añade el argumento.

    Returns:
        El mismo parser con el argumento añadido.
    """
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help=(
            "Feature set a usar. Ver src/features/feature_sets.py. "
            f"Default: {DEFAULT_FEATURE_SET}"
        ),
    )
    return parser


def parse_feature_set_args(description: str) -> argparse.Namespace:
    """Crea un parser con --feature-set y retorna el Namespace parseado.

    Args:
        description: Descripción del script que aparece en --help.

    Returns:
        Namespace con atributo feature_set.
    """
    parser = argparse.ArgumentParser(description=description)
    add_feature_set_arg(parser)
    return parser.parse_args()
