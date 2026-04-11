"""
Transformaciones base de feature engineering para Spaceship Titanic.

Cubre extracción de columnas estructurales, imputación, reglas de dominio
y creación de features fundamentales. Todas las funciones son puras:
nunca mutan el input (siempre df.copy()).
"""
import pandas as pd
import numpy as np

from src.features.engineering.encoders import _cryo_to_int


_SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
_CATEGORICAL_FILL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
_DECK_TO_HOMEPLANET = {"A": "Europa", "B": "Europa", "C": "Europa", "G": "Earth"}


def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae Deck, CabinNumber y Side desde la columna Cabin.

    NB02: Deck (chi²=392.3) y Side (chi²=91.1) son discriminadores
    estadisticamente significativos (p < 0.001).

    Args:
        df: DataFrame con columna 'Cabin' en formato 'Deck/Num/Side'.

    Returns:
        DataFrame con columnas 'Deck', 'CabinNumber', 'Side' añadidas.
    """
    df_copy = df.copy()
    df_copy["Deck"] = df_copy["Cabin"].apply(
        lambda x: x.split("/")[0] if pd.notna(x) else "Unknown"
    )
    df_copy["CabinNumber"] = df_copy["Cabin"].apply(
        lambda x: int(x.split("/")[1]) if pd.notna(x) else 0
    )
    df_copy["Side"] = df_copy["Cabin"].apply(
        lambda x: x.split("/")[2] if pd.notna(x) else "Unknown"
    )
    return df_copy


def extract_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae TravelGroup y GroupSize desde PassengerId.

    NB02: GroupSize (chi²=145.3, p < 0.001) muestra patrón no lineal
    con el target — viajeros en grupos de 3-6 tienen mayor tasa de transporte.

    Args:
        df: DataFrame con columna 'PassengerId' en formato 'GGGG_NN'.

    Returns:
        DataFrame con columnas 'TravelGroup' y 'GroupSize' añadidas.
    """
    df_copy = df.copy()
    df_copy["TravelGroup"] = df_copy["PassengerId"].str.split("_").str[0]
    df_copy["GroupSize"] = (
        df_copy.groupby("TravelGroup")["TravelGroup"].transform("count")
    )
    return df_copy


def create_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea TotalSpending, HasSpending, SpendingCategories y TotalSpending_Log.

    NB02: TotalSpending_Log es el feature más correlacionado con el target
    (r=-0.469 vs r=-0.200 del original) gracias a la transformación log.

    Args:
        df: DataFrame con columnas de gasto individuales.

    Returns:
        DataFrame con features de gasto agregadas añadidas.
    """
    df_copy = df.copy()
    df_copy["TotalSpending"] = df_copy[_SPENDING_COLS].fillna(0).sum(axis=1)
    df_copy["HasSpending"] = (df_copy["TotalSpending"] > 0).astype(int)
    df_copy["SpendingCategories"] = (
        (df_copy[_SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    )
    df_copy["TotalSpending_Log"] = np.log1p(df_copy["TotalSpending"])
    return df_copy


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza Age en rangos etarios (Child, Teen, YoungAdult, Adult, Senior).

    NB02: Age es significativa (Mann-Whitney p < 0.001). Se categoriza para
    capturar la no-linealidad de su relación con el target.

    Args:
        df: DataFrame con columna 'Age'.

    Returns:
        DataFrame con columna 'AgeCategory' añadida.
    """
    def _categorize(age) -> str:
        if pd.isna(age):
            return "Unknown"
        if age < 13:
            return "Child"
        if age < 18:
            return "Teen"
        if age < 30:
            return "YoungAdult"
        if age < 60:
            return "Adult"
        return "Senior"

    df_copy = df.copy()
    df_copy["AgeCategory"] = df_copy["Age"].apply(_categorize)
    return df_copy


def handle_missing_values_spaceship(
    df: pd.DataFrame,
    impute_age: bool = False,
) -> pd.DataFrame:
    """Aplica la estrategia de nulos definida en NB02.

    Estrategia:
    - Categoricas (HomePlanet, CryoSleep, Destination, VIP) → 'Unknown'
    - Variables de gasto → 0 (ausencia = sin gasto registrado)
    - Age (train): eliminar filas para evitar sesgo en AgeCategory
    - Age (test):  imputar con mediana para preservar todos los registros

    Args:
        df: DataFrame con valores faltantes.
        impute_age: Si True, imputa Age con la mediana en lugar de eliminar filas.

    Returns:
        DataFrame sin nulos en las columnas tratadas.
    """
    df_copy = df.copy()
    for col in _CATEGORICAL_FILL:
        df_copy[col] = df_copy[col].fillna("Unknown")
    for col in _SPENDING_COLS:
        df_copy[col] = df_copy[col].fillna(0)
    if impute_age:
        df_copy["Age"] = df_copy["Age"].fillna(df_copy["Age"].median())
    else:
        df_copy = df_copy.dropna(subset=["Age"])
    return df_copy


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas físicas del dataset para corregir e imputar valores nulos.

    Requiere: extract_cabin_features y extract_group_features aplicados antes.

    Reglas (en orden):
    1. HomePlanet por grupo — propaga el valor conocido a compañeros sin dato.
    2. Deck → HomePlanet — A/B/C exclusivos de Europa; G de Earth.
    3. Deck y Side por grupo — propaga valores conocidos dentro del grupo.
    4. CryoSleep=True → spending = 0 (congelados no consumen).
    5. Spending > 0 → CryoSleep = False (si hay gasto, no estaba en cryo).
    6. Age <= 12 → spending = 0 (menores no acceden a amenidades de pago).

    Args:
        df: DataFrame con Deck, Side, TravelGroup, HomePlanet, CryoSleep, Age
            y columnas de gasto.

    Returns:
        DataFrame con valores imputados por reglas de dominio.
    """
    df_copy = df.copy()

    # Regla 1: HomePlanet por grupo
    known_hp = (
        df_copy.dropna(subset=["HomePlanet"])
        .groupby("TravelGroup")["HomePlanet"]
        .first()
    )
    hp_null = df_copy["HomePlanet"].isna()
    df_copy.loc[hp_null, "HomePlanet"] = (
        df_copy.loc[hp_null, "TravelGroup"].map(known_hp)
    )

    # Regla 2: Deck → HomePlanet
    for deck, planet in _DECK_TO_HOMEPLANET.items():
        mask = (df_copy["Deck"] == deck) & df_copy["HomePlanet"].isna()
        df_copy.loc[mask, "HomePlanet"] = planet

    # Regla 3: Deck y Side por grupo
    for col in ["Deck", "Side"]:
        if col not in df_copy.columns:
            continue
        known_val = (
            df_copy[df_copy[col] != "Unknown"]
            .groupby("TravelGroup")[col]
            .first()
        )
        unknown_mask = df_copy[col] == "Unknown"
        filled = df_copy.loc[unknown_mask, "TravelGroup"].map(known_val)
        df_copy.loc[unknown_mask, col] = filled.fillna("Unknown")

    # Regla 4: CryoSleep=True → spending = 0
    cryo_true = df_copy["CryoSleep"].isin([True, "True"])
    for col in _SPENDING_COLS:
        df_copy.loc[cryo_true & df_copy[col].isna(), col] = 0.0

    # Regla 5: Spending > 0 → CryoSleep = False
    spending_positive = (df_copy[_SPENDING_COLS].fillna(0) > 0).any(axis=1)
    cryo_null = df_copy["CryoSleep"].isna()
    df_copy.loc[spending_positive & cryo_null, "CryoSleep"] = False

    # Regla 6: Age <= 12 → spending = 0
    age_child = df_copy["Age"].notna() & (df_copy["Age"] <= 12)
    for col in _SPENDING_COLS:
        df_copy.loc[age_child & df_copy[col].isna(), col] = 0.0

    return df_copy


def impute_spending_group_aware(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa columnas de gasto con conciencia de grupo y CryoSleep.

    Estrategia:
    - CryoSleep=True: gasto → 0 (regla del mundo: congelados no gastan).
    - CryoSleep=False/Unknown: NaN → mediana del TravelGroup → mediana global → 0.

    Prerequisito: TravelGroup y CryoSleep disponibles (llamar después de
    extract_group_features y fill categoricals).

    Args:
        df: DataFrame con TravelGroup, CryoSleep y columnas de gasto.

    Returns:
        DataFrame con columnas de gasto imputadas.
    """
    df_copy = df.copy()
    cryo_mask = df_copy["CryoSleep"].isin([True, "True"])

    for col in _SPENDING_COLS:
        null_mask = df_copy[col].isna()
        if null_mask.sum() == 0:
            continue
        global_med = df_copy[col].median()
        group_med = (
            df_copy.groupby("TravelGroup")[col]
            .transform("median")
            .fillna(global_med)
            .fillna(0)
        )
        df_copy.loc[null_mask & cryo_mask, col] = 0.0
        df_copy.loc[null_mask & ~cryo_mask, col] = group_med[null_mask & ~cryo_mask]

    return df_copy


def impute_age_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa Age NaN usando la mediana del TravelGroup antes del fallback global.

    82 de 179 NaN de Age pertenecen a grupos con al menos otro miembro con Age
    conocida. La mediana del grupo es más precisa que la global porque los grupos
    suelen ser familias (edades correladas).

    Llamar después de extract_group_features y antes de create_age_features.

    Args:
        df: DataFrame con columnas TravelGroup y Age.

    Returns:
        DataFrame con Age parcialmente imputada.
    """
    df_copy = df.copy()
    group_median = df_copy.groupby("TravelGroup")["Age"].transform("median")
    df_copy["Age"] = df_copy["Age"].fillna(group_median)
    return df_copy
