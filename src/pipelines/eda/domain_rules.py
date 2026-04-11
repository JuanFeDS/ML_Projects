"""
Validación empírica de las reglas físicas del dataset Spaceship Titanic.

Estas reglas sustentan apply_domain_rules() en src/features/engineering/base.py.
Este módulo cuantifica cuántos registros las violan y cuántos NaN pueden
resolverse por inferencia — justificando cada regla con datos reales.

Reglas validadas:
  R1. CryoSleep=True  → todos los gastos deben ser 0.
  R2. Gasto > 0       → CryoSleep debe ser False.
  R3. Age <= 12       → todos los gastos deben ser 0.
  R4. Deck A/B/C      → HomePlanet = Europa (casi exclusivo).
  R5. Deck G          → HomePlanet = Earth (casi exclusivo).

Oportunidades de imputación cuantificadas:
  I1. HomePlanet NaN resolubles por TravelGroup.
  I2. Deck/Side NaN resolubles por TravelGroup.
  I3. Age NaN resolubles por mediana del TravelGroup.
  I4. CryoSleep NaN inferibles desde gasto (R2 inversa).
"""
from typing import Any, Dict

import pandas as pd

from src.features.constants import SPENDING_COLS


def _has_spending(df: pd.DataFrame) -> pd.Series:
    return (df[SPENDING_COLS].fillna(0) > 0).any(axis=1)


def compute_rule_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Cuantifica violaciones a las reglas físicas del dataset.

    Returns:
        DataFrame con columnas [Regla, Descripción, Violaciones, % del total].
    """
    cryo_true = df["CryoSleep"].isin([True, "True"])
    cryo_unknown = ~df["CryoSleep"].isin([True, "True", False, "False"])
    has_spend = _has_spending(df)
    age_child = df["Age"].notna() & (df["Age"] <= 12)

    deck = df["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) else None
    )
    homeplanet = df["HomePlanet"]

    n = len(df)
    rows = [
        {
            "Regla": "R1",
            "Descripción": "CryoSleep=True con gasto > 0",
            "Violaciones": int((cryo_true & has_spend).sum()),
            "% del total": round((cryo_true & has_spend).mean() * 100, 3),
        },
        {
            "Regla": "R2",
            "Descripción": "Gasto > 0 con CryoSleep=Unknown (inferible = False)",
            "Violaciones": int((has_spend & cryo_unknown).sum()),
            "% del total": round((has_spend & cryo_unknown).mean() * 100, 3),
        },
        {
            "Regla": "R3",
            "Descripción": "Age <= 12 con gasto > 0",
            "Violaciones": int((age_child & has_spend).sum()),
            "% del total": round((age_child & has_spend).mean() * 100, 3),
        },
        {
            "Regla": "R4",
            "Descripción": "Deck A/B/C con HomePlanet ≠ Europa",
            "Violaciones": int(
                (deck.isin(["A", "B", "C"]) & homeplanet.notna() & (homeplanet != "Europa")).sum()
            ),
            "% del total": round(
                (deck.isin(["A", "B", "C"]) & homeplanet.notna() & (homeplanet != "Europa")).mean() * 100, 3
            ),
        },
        {
            "Regla": "R5",
            "Descripción": "Deck G con HomePlanet ≠ Earth",
            "Violaciones": int(((deck == "G") & homeplanet.notna() & (homeplanet != "Earth")).sum()),
            "% del total": round(
                ((deck == "G") & homeplanet.notna() & (homeplanet != "Earth")).mean() * 100, 3
            ),
        },
    ]
    return pd.DataFrame(rows)


def compute_imputation_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Cuantifica NaN resolubles por inferencia de grupo o reglas de dominio.

    Returns:
        DataFrame con columnas [Variable, NaN total, Resolubles, % resolubles, Método].
    """
    df = df.copy()
    df["TravelGroup"] = df["PassengerId"].str.split("_").str[0]
    df["Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else None)

    # HomePlanet: por grupo
    hp_null = df["HomePlanet"].isna()
    known_hp = df.dropna(subset=["HomePlanet"]).groupby("TravelGroup")["HomePlanet"].first()
    resolvable_hp_group = hp_null & df["TravelGroup"].isin(known_hp.index)

    # HomePlanet: por deck (A/B/C → Europa, G → Earth)
    deck_rule = {"A": "Europa", "B": "Europa", "C": "Europa", "G": "Earth"}
    resolvable_hp_deck = hp_null & ~resolvable_hp_group & df["Deck"].isin(deck_rule.keys())

    # Deck/Side: por grupo
    deck_null = df["Cabin"].isna()
    known_deck = (
        df[df["Cabin"].notna()]
        .assign(Deck_raw=lambda d: d["Cabin"].str.split("/").str[0])
        .groupby("TravelGroup")["Deck_raw"]
        .first()
    )
    resolvable_deck = deck_null & df["TravelGroup"].isin(known_deck.index)

    # Age: por mediana del grupo
    age_null = df["Age"].isna()
    group_has_age = df.dropna(subset=["Age"]).groupby("TravelGroup")["Age"].count() > 0
    resolvable_age = age_null & df["TravelGroup"].isin(group_has_age[group_has_age].index)

    # CryoSleep: por gasto > 0 (R2 inversa)
    cryo_null = ~df["CryoSleep"].isin([True, "True", False, "False"])
    has_spend = _has_spending(df)
    resolvable_cryo = cryo_null & has_spend

    rows = [
        {
            "Variable": "HomePlanet",
            "NaN total": int(hp_null.sum()),
            "Resolubles": int(resolvable_hp_group.sum() + resolvable_hp_deck.sum()),
            "% resolubles": round(
                (resolvable_hp_group.sum() + resolvable_hp_deck.sum()) / max(hp_null.sum(), 1) * 100, 1
            ),
            "Método": "TravelGroup + regla Deck→Planeta",
        },
        {
            "Variable": "Cabin/Deck/Side",
            "NaN total": int(deck_null.sum()),
            "Resolubles": int(resolvable_deck.sum()),
            "% resolubles": round(resolvable_deck.sum() / max(deck_null.sum(), 1) * 100, 1),
            "Método": "TravelGroup",
        },
        {
            "Variable": "Age",
            "NaN total": int(age_null.sum()),
            "Resolubles": int(resolvable_age.sum()),
            "% resolubles": round(resolvable_age.sum() / max(age_null.sum(), 1) * 100, 1),
            "Método": "Mediana del TravelGroup",
        },
        {
            "Variable": "CryoSleep",
            "NaN total": int(cryo_null.sum()),
            "Resolubles": int(resolvable_cryo.sum()),
            "% resolubles": round(resolvable_cryo.sum() / max(cryo_null.sum(), 1) * 100, 1),
            "Método": "Gasto > 0 → CryoSleep=False",
        },
    ]
    return pd.DataFrame(rows)


def run_domain_rules_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """Validación completa de reglas físicas y oportunidades de imputación.

    Returns:
        dict con violations (DataFrame) e imputation_opportunities (DataFrame).
    """
    return {
        "violations": compute_rule_violations(df),
        "imputation_opportunities": compute_imputation_opportunities(df),
    }
