"""
Helpers de encoding y conversión de tipos para Spaceship Titanic.

Incluye tanto las funciones públicas reutilizables como los helpers
privados que eliminan duplicación interna entre funciones de feature engineering.
"""
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers privados (usados internamente por base.py y derived.py)
# ---------------------------------------------------------------------------

def _cryo_to_int(series: pd.Series) -> pd.Series:
    """Convierte CryoSleep a entero: 1=True, 0=False/Unknown/NaN.

    Punto único de conversión — evita duplicar el map en cada función
    que necesite operar numéricamente sobre CryoSleep.

    Args:
        series: Columna CryoSleep con valores True/False/"True"/"False"/NaN.

    Returns:
        Serie de enteros 0/1.
    """
    return (
        series.map({True: 1, "True": 1, False: 0, "False": 0})
        .fillna(0)
        .astype(int)
    )


# ---------------------------------------------------------------------------
# Funciones públicas de encoding
# ---------------------------------------------------------------------------

def encode_cryosleep(val) -> int:
    """Codifica CryoSleep a entero.

    Args:
        val: Valor original (True, False, "True", "False" o cualquier otro).

    Returns:
        1 para True, 0 para False, -1 para desconocido.
    """
    if val in (True, "True"):
        return 1
    if val in (False, "False"):
        return 0
    return -1


def encode_side(val) -> int:
    """Codifica Side a entero.

    Args:
        val: Valor original ("P", "S" o cualquier otro).

    Returns:
        0 para P (port), 1 para S (starboard), -1 para desconocido.
    """
    if val == "P":
        return 0
    if val == "S":
        return 1
    return -1
