"""
Análisis de la estructura de grupos de viaje (TravelGroup).

El PassengerId tiene formato GGGG_MM — los primeros 4 dígitos identifican
el grupo y MM el número de miembro. 4805 de 6217 grupos viajan solos.

Hallazgos clave que motivan features de grupo (fs-013):
  - Solo (n=4805) transported=45.2% vs agrupado (n=3888) transported=56.7%
  - HomePlanet es siempre homogéneo dentro del grupo (0 grupos mixtos)
  - GroupAllCryo → 80.5% transported vs 42.4% sin nadie en CryoSleep
  - 94% de grupos comparten el mismo apellido → indicador de familia
"""

from typing import Any, Dict

import pandas as pd

from src.features.constants import TARGET
from src.pipelines.eda.statistical import compute_chi2_stats


def _add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega TravelGroup, GroupSize y LastName sin mutar el input."""
    out = df.copy()
    out["TravelGroup"] = out["PassengerId"].str.split("_").str[0]
    out["GroupSize"] = out.groupby("TravelGroup")["TravelGroup"].transform("count")
    out["IsSolo"] = (out["GroupSize"] == 1).astype(int)
    if "Name" in out.columns:
        out["LastName"] = out["Name"].str.split().str[-1]
    return out


def compute_group_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de GroupSize y tasa de Transported por tamaño de grupo.

    Returns:
        DataFrame: GroupSize × [n_grupos, n_pasajeros, transported_rate].
    """
    temp = _add_group_cols(df)
    group_stats = (
        temp.groupby("GroupSize")
        .agg(
            n_pasajeros=(TARGET, "count"),
            transported_rate=(TARGET, "mean"),
        )
        .round({"transported_rate": 4})
        .reset_index()
    )
    group_stats["n_grupos"] = group_stats["n_pasajeros"] // group_stats["GroupSize"]
    return group_stats[["GroupSize", "n_grupos", "n_pasajeros", "transported_rate"]]


def compute_solo_vs_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de Transported para pasajeros solos vs agrupados.

    Returns:
        DataFrame: Segmento × [n, transported_rate].
    """
    temp = _add_group_cols(df)
    stats = (
        temp.groupby("IsSolo")
        .agg(n=(TARGET, "count"), transported_rate=(TARGET, "mean"))
        .reset_index()
    )
    stats["Segmento"] = stats["IsSolo"].map({1: "Solo", 0: "Agrupado"})
    stats["transported_rate"] = stats["transported_rate"].round(4)
    return stats[["Segmento", "n", "transported_rate"]]


def compute_group_cryo_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de Transported según el estado de CryoSleep del grupo completo.

    Segmentos:
      - AllCryo: todos los miembros del grupo en CryoSleep.
      - AnyCryo: al menos uno (pero no todos) en CryoSleep.
      - NoCryo: ninguno en CryoSleep.
      - Solo: grupos de 1 (el estado individual equivale al grupal).

    Returns:
        DataFrame: EstadoGrupo × [n, transported_rate].
    """
    temp = _add_group_cols(df)
    cryo_int = (
        temp["CryoSleep"]
        .map({True: 1, "True": 1, False: 0, "False": 0})
        .fillna(0)
        .astype(int)
    )
    temp["_cryo_int"] = cryo_int

    group_min = temp.groupby("TravelGroup")["_cryo_int"].transform("min")
    group_max = temp.groupby("TravelGroup")["_cryo_int"].transform("max")

    conditions = [
        temp["IsSolo"] == 1,
        (group_min == 1),
        (group_max == 1) & (group_min == 0),
    ]
    labels = ["Solo", "AllCryo", "AnyCryo"]
    temp["CryoSegment"] = "NoCryo"
    for cond, label in zip(reversed(conditions), reversed(labels)):
        temp.loc[cond, "CryoSegment"] = label

    result = (
        temp.groupby("CryoSegment")
        .agg(n=(TARGET, "count"), transported_rate=(TARGET, "mean"))
        .round({"transported_rate": 4})
        .reset_index()
        .rename(columns={"CryoSegment": "Estado Grupo"})
    )
    # Ordenar por tasa descendente
    return result.sort_values("transported_rate", ascending=False).reset_index(
        drop=True
    )


def compute_group_homeplanet_homogeneity(df: pd.DataFrame) -> dict:
    """Verifica si HomePlanet es siempre homogéneo dentro de un grupo.

    Hallazgo: 0 grupos con HomePlanet mixto. Esto valida que HomePlanet
    puede imputarse desde otros miembros del grupo sin ambigüedad.

    Returns:
        dict con n_grupos_total, n_grupos_mixtos, pct_homogeneous.
    """
    temp = _add_group_cols(df)
    grouped = temp[temp["GroupSize"] > 1]
    hp_per_group = (
        grouped.dropna(subset=["HomePlanet"])
        .groupby("TravelGroup")["HomePlanet"]
        .nunique()
    )
    mixed = int((hp_per_group > 1).sum())
    total = int(hp_per_group.shape[0])
    return {
        "n_grupos_total": int(temp["TravelGroup"].nunique()),
        "n_grupos_mixtos_homeplanet": mixed,
        "pct_homogeneous": (
            round((total - mixed) / total * 100, 2) if total > 0 else 100.0
        ),
    }


def compute_family_surname_stats(df: pd.DataFrame) -> dict:
    """Cuantifica cuántos grupos comparten apellido (indicador de familia).

    El apellido es el último token de Name. Se ignoran registros sin Name.

    Returns:
        dict con n_grupos, grupos_mismo_apellido, pct_familia,
        apellidos_compartidos_entre_grupos.
    """
    if "Name" not in df.columns:
        return {"error": "columna Name no disponible"}

    temp = _add_group_cols(df).dropna(subset=["Name"])
    multi = temp[temp["GroupSize"] > 1]

    apellidos_por_grupo = multi.groupby("TravelGroup")["LastName"].nunique()
    mismos = int((apellidos_por_grupo == 1).sum())
    total = int(apellidos_por_grupo.shape[0])

    # Apellidos que aparecen en más de un grupo distinto
    surname_groups = temp.groupby("LastName")["TravelGroup"].nunique()
    compartidos = int((surname_groups > 1).sum())

    return {
        "n_grupos_multi": total,
        "grupos_mismo_apellido": mismos,
        "pct_familia": round(mismos / total * 100, 2) if total > 0 else 0.0,
        "apellidos_compartidos_entre_grupos": compartidos,
    }


def run_group_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis completo de la estructura de grupos de viaje.

    Returns:
        dict con:
        - size_dist: distribución GroupSize y tasa transported.
        - solo_vs_grouped: comparación solo vs agrupado.
        - cryo_dynamics: tasa transported por estado CryoSleep del grupo.
        - homeplanet_homogeneity: validación de homogeneidad de planeta.
        - family_stats: indicador de familia por apellido.
        - groupsize_chi2: test chi2 GroupSize vs Transported.
    """
    return {
        "size_dist": compute_group_size_distribution(df),
        "solo_vs_grouped": compute_solo_vs_grouped(df),
        "cryo_dynamics": compute_group_cryo_dynamics(df),
        "homeplanet_homogeneity": compute_group_homeplanet_homogeneity(df),
        "family_stats": compute_family_surname_stats(df),
        "groupsize_chi2": compute_chi2_stats(_add_group_cols(df), "GroupSize", TARGET),
    }
