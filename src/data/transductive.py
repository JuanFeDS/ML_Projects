"""Imputacion transductiva train + test con propagacion de contexto de grupo."""
from __future__ import annotations

import pandas as pd

from src.features.engineering.base import (
    apply_domain_rules,
    extract_cabin_features,
    extract_group_features,
    impute_age_by_group,
)

_SPLIT_COL = "_split"
IMPUTE_COLS = [
    "HomePlanet", "CryoSleep", "Age",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
]


def run_transductive_imputation(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa NaN usando informacion conjunta de train y test.

    Combina ambos conjuntos para propagar valores conocidos dentro de grupos
    familiares (HomePlanet, CryoSleep, Deck, Side, Age). Ningún paso filtra
    ni usa el target — solo se propagan valores ya conocidos.

    Args:
        df_train: DataFrame crudo de train.csv.
        df_test: DataFrame crudo de test.csv.

    Returns:
        Tupla (train_imputado, test_imputado) con la misma estructura que los
        DataFrames de entrada.
    """
    train_work = df_train.copy()
    test_work = df_test.copy()

    train_work[_SPLIT_COL] = "train"
    test_work[_SPLIT_COL] = "test"
    test_work["Transported"] = pd.NA

    combined = pd.concat([train_work, test_work], ignore_index=True)

    combined = extract_cabin_features(combined)
    combined = extract_group_features(combined)
    combined = apply_domain_rules(combined)
    combined = impute_age_by_group(combined)

    train_part = combined[combined[_SPLIT_COL] == "train"]
    test_part = combined[combined[_SPLIT_COL] == "test"]

    train_out = df_train.copy()
    test_out = df_test.copy()

    for col in IMPUTE_COLS:
        if col in train_out.columns:
            null_mask = train_out[col].isna()
            train_out.loc[null_mask, col] = train_part[col].values[null_mask]

        if col in test_out.columns:
            null_mask = test_out[col].isna()
            test_out.loc[null_mask, col] = test_part[col].values[null_mask]

    return train_out, test_out
