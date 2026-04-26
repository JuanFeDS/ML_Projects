"""
00_transductive_impute.py - Imputacion transductiva train + test.

Combina train.csv y test.csv para resolver NaN por reglas de dominio y
contexto de grupo usando informacion de AMBOS conjuntos. Ningún paso
filtra el target — solo se propagan valores ya conocidos (HomePlanet,
CryoSleep, Deck, Side, Age) dentro de grupos que tienen miembros en
train y en test.

Output:
    data/processed/train_transductive.csv
    data/processed/test_transductive.csv

Ambos tienen la misma estructura que los CSVs originales.
Usalos en lugar de data/raw/ con --train-path y --test-path.
"""
import pandas as pd

from src.config.settings import DATA_PROCESSED_DIR, TEST_RAW, TRAIN_RAW
from src.data.transductive import IMPUTE_COLS, run_transductive_imputation


def main() -> None:
    """Aplica imputacion transductiva y guarda los datasets resultantes."""
    print("=" * 60)
    print("00_transductive_impute.py -- Imputacion transductiva")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_RAW)
    df_test = pd.read_csv(TEST_RAW)
    print(f"[OK] Train: {len(df_train):,} | Test: {len(df_test):,}")

    print("\n[...] Aplicando imputacion transductiva...")
    train_out, test_out = run_transductive_imputation(df_train, df_test)

    print(f"\n  {'Columna':<14} {'Train NaN orig':>14} {'Train NaN post':>14} "
          f"{'Test NaN orig':>13} {'Test NaN post':>13}")
    for col in IMPUTE_COLS:
        t_orig = df_train[col].isna().sum() if col in df_train else 0
        t_post = train_out[col].isna().sum() if col in train_out else 0
        te_orig = df_test[col].isna().sum() if col in df_test else 0
        te_post = test_out[col].isna().sum() if col in test_out else 0
        if t_orig + te_orig > 0:
            print(f"  {col:<14} {t_orig:>14} {t_post:>14} {te_orig:>13} {te_post:>13}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_train = DATA_PROCESSED_DIR / "train_transductive.csv"
    out_test = DATA_PROCESSED_DIR / "test_transductive.csv"
    train_out.to_csv(out_train, index=False)
    test_out.to_csv(out_test, index=False)

    print(f"\n[OK] Train transductivo: {out_train}")
    print(f"[OK] Test transductivo:  {out_test}")
    print("\nNaN residuales en train_transductive:")
    print(train_out[IMPUTE_COLS].isna().sum().to_string())


if __name__ == "__main__":
    main()
