"""Analisis de errores del modelo por segmento del dataset."""
from typing import Dict

import pandas as pd


def analyze_errors(
    x_val: pd.DataFrame,
    y_val: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Analiza la tasa de error por segmento en el conjunto de validacion.

    Reconstruye las variables originales desde columnas OHE/encoded y
    calcula cuantos errores comete el modelo en cada categoria.

    Args:
        x_val: Features del conjunto de validacion (escaladas, con OHE).
        y_val: Target real.
        y_pred: Predicciones del modelo (array o Series).

    Returns:
        Diccionario {nombre_segmento: DataFrame con [segmento, n, errors, error_rate]}.
    """
    df_err = x_val.copy()
    df_err["_y_true"] = y_val.values
    df_err["_y_pred"] = pd.array(y_pred)
    df_err["_error"] = (df_err["_y_true"] != df_err["_y_pred"]).astype(int)

    results = {}

    if "CryoSleep_Encoded" in df_err.columns:
        cryo_map = {1: "Cryo", 0: "Active", -1: "Unknown"}
        df_err["_CryoSleep"] = df_err["CryoSleep_Encoded"].map(cryo_map)
        g = df_err.groupby("_CryoSleep")["_error"].agg(["count", "sum"]).reset_index()
        g.columns = ["CryoSleep", "n", "errors"]
        g["error_rate"] = (g["errors"] / g["n"]).round(4)
        results["CryoSleep"] = g.sort_values("error_rate", ascending=False)

    for prefix in ["HomePlanet", "Destination", "AgeCategory", "Deck"]:
        cols = [c for c in df_err.columns if c.startswith(f"{prefix}_")]
        if not cols:
            continue
        segment = (
            df_err[cols]
            .idxmax(axis=1)
            .str.replace(f"{prefix}_", "", regex=False)
        )
        df_err[f"_{prefix}"] = segment
        g = (
            df_err.groupby(f"_{prefix}")["_error"]
            .agg(["count", "sum"])
            .reset_index()
        )
        g.columns = [prefix, "n", "errors"]
        g["error_rate"] = (g["errors"] / g["n"]).round(4)
        results[prefix] = g.sort_values("error_rate", ascending=False)

    return results
