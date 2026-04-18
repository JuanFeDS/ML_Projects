"""Utilidades de control de versiones (git) para trazabilidad de experimentos."""
from __future__ import annotations

import subprocess


def get_git_commit() -> str:
    """Retorna el hash corto del commit HEAD, o 'unknown' si falla."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pylint: disable=broad-except
        return "unknown"


def create_git_tag(exp_id: str, fs_name: str, val_accuracy: float) -> None:
    """Crea un git tag para el experimento promovido.

    Args:
        exp_id: Identificador del experimento (ej. '027').
        fs_name: Nombre del feature set usado.
        val_accuracy: Accuracy de validacion del modelo ganador.
    """
    tag = f"exp-{exp_id}_{fs_name}_{val_accuracy:.4f}"
    try:
        subprocess.run(["git", "tag", tag], check=True, stderr=subprocess.DEVNULL)
        print(f"  [TAG] Git tag creado: {tag}")
    except subprocess.CalledProcessError:
        print(f"  [WARN] No se pudo crear git tag '{tag}' (ya existe?)")
