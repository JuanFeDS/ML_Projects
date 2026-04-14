"""
00_transductive_impute.py - Imputación transductiva train + test.

Combina train.csv y test.csv para resolver NaN por reglas de dominio y
contexto de grupo usando información de AMBOS conjuntos. Ningún paso
filtra el target — solo se propagan valores ya conocidos (HomePlanet,
CryoSleep, Deck, Side, Age) dentro de grupos que tienen miembros en
train y en test.

Output:
    data/processed/train_transductive.csv
    data/processed/test_transductive.csv

Ambos tienen la misma estructura que los CSVs originales.
Úsalos en lugar de data/raw/ con --train-path y --test-path.
"""
import pandas as pd

from src.config.settings import DATA_PROCESSED_DIR, TEST_RAW, TRAIN_RAW
from src.features.engineering.base import (
    apply_domain_rules,
    extract_cabin_features,
    extract_group_features,
    impute_age_by_group,
)

_SPLIT_COL = "_split"
_RAW_COLS_TRAIN = [
    "PassengerId", "HomePlanet", "CryoSleep", "Cabin",
    "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Name", "Transported",
]
_RAW_COLS_TEST = [c for c in _RAW_COLS_TRAIN if c != "Transported"]


def _imputed_back(combined: pd.DataFrame, original: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Reconstruye el DataFrame original con los valores imputados del combinado.

    Solo actualiza columnas que tenían NaN en el original — los valores
    conocidos no se tocan.
    """
    out = original.copy()
    for col in cols:
        if col not in combined.columns or col not in out.columns:
            continue
        null_mask = out[col].isna()
        if null_mask.sum() == 0:
            continue
        filled = combined.loc[combined[_SPLIT_COL] == out[_SPLIT_COL].iloc[0], col]
        filled.index = out.index
        out.loc[null_mask, col] = filled[null_mask]
    return out


def main() -> None:
    print("=" * 60)
    print("00_transductive_impute.py -- Imputación transductiva")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cargar train y test
    # ------------------------------------------------------------------
    df_train = pd.read_csv(TRAIN_RAW)
    df_test  = pd.read_csv(TEST_RAW)
    print(f"[OK] Train: {len(df_train):,} | Test: {len(df_test):,}")

    df_train[_SPLIT_COL] = "train"
    df_test[_SPLIT_COL]  = "test"

    # Agregar columna Transported vacía al test para poder concatenar
    df_test["Transported"] = pd.NA

    combined = pd.concat([df_train, df_test], ignore_index=True)
    print(f"[OK] Combinado: {len(combined):,} registros")

    # ------------------------------------------------------------------
    # 2. Imputación sobre el conjunto combinado
    # ------------------------------------------------------------------
    print("\n[...] Aplicando imputación transductiva...")

    combined = extract_cabin_features(combined)
    combined = extract_group_features(combined)
    combined = apply_domain_rules(combined)
    combined = impute_age_by_group(combined)

    # Cuantificar mejora
    train_part = combined[combined[_SPLIT_COL] == "train"]
    test_part  = combined[combined[_SPLIT_COL] == "test"]

    impute_cols = ["HomePlanet", "CryoSleep", "Age",
                   "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    print(f"\n  {'Columna':<14} {'Train NaN orig':>14} {'Train NaN post':>14} {'Test NaN orig':>13} {'Test NaN post':>13}")
    for col in impute_cols:
        t_orig  = df_train[col].isna().sum()
        t_post  = train_part[col].isna().sum()
        te_orig = df_test[col].isna().sum() if col in df_test else 0
        te_post = test_part[col].isna().sum()
        if t_orig + te_orig > 0:
            print(f"  {col:<14} {t_orig:>14} {t_post:>14} {te_orig:>13} {te_post:>13}")

    # ------------------------------------------------------------------
    # 3. Reconstruir CSVs con estructura original + valores imputados
    # ------------------------------------------------------------------
    # Solo actualizamos las columnas que estaban en el raw original
    train_out = df_train.drop(columns=[_SPLIT_COL]).copy()
    test_out  = df_test.drop(columns=[_SPLIT_COL, "Transported"]).copy()

    for col in impute_cols:
        if col in train_out.columns:
            null_mask = train_out[col].isna()
            train_out.loc[null_mask, col] = train_part.loc[
                combined[_SPLIT_COL] == "train", col
            ].values[null_mask]

        if col in test_out.columns:
            null_mask = test_out[col].isna()
            test_out.loc[null_mask, col] = test_part.loc[
                combined[_SPLIT_COL] == "test", col
            ].values[null_mask]

    # ------------------------------------------------------------------
    # 4. Guardar
    # ------------------------------------------------------------------
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_train = DATA_PROCESSED_DIR / "train_transductive.csv"
    out_test  = DATA_PROCESSED_DIR / "test_transductive.csv"

    train_out.to_csv(out_train, index=False)
    test_out.to_csv(out_test, index=False)

    print(f"\n[OK] Train transductivo: {out_train}")
    print(f"[OK] Test transductivo:  {out_test}")
    print("\nNaN residuales en train_transductive:")
    print(train_out[impute_cols].isna().sum().to_string())


if __name__ == "__main__":
    main()
