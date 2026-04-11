"""Tests unitarios para src/features/engineering/.

Cubre las transformaciones base, encoders y features derivadas.
Todos los tests usan DataFrames minimos construidos en memoria —
sin dependencia de archivos de datos externos.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    encode_cryosleep,
    encode_side,
    extract_cabin_features,
    extract_group_features,
    create_spending_features,
    create_age_features,
    handle_missing_values_spaceship,
    apply_domain_rules,
    create_solo_interaction_features,
    create_group_context_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """DataFrame minimo con las columnas crudas del dataset Spaceship Titanic."""
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01"],
        "Cabin": ["B/0034/P", "B/0035/S", "G/0100/P"],
        "HomePlanet": ["Europa", "Europa", "Earth"],
        "CryoSleep": [True, False, False],
        "Destination": ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
        "Age": [23.0, 45.0, 17.0],
        "VIP": [False, False, False],
        "RoomService": [0.0, 150.0, 0.0],
        "FoodCourt": [0.0, 200.0, 50.0],
        "ShoppingMall": [0.0, 0.0, 0.0],
        "Spa": [0.0, 300.0, 0.0],
        "VRDeck": [0.0, 100.0, 0.0],
        "Name": ["Alice A", "Bob A", "Carol B"],
        "Transported": [True, False, True],
    })


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """DataFrame con nulos tipicos para probar imputacion."""
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01"],
        "Cabin": [None, "B/0035/S", "G/0100/P"],
        "HomePlanet": [None, "Europa", "Earth"],
        "CryoSleep": [None, False, True],
        "Destination": ["TRAPPIST-1e", None, "PSO J318.5-22"],
        "Age": [None, 45.0, 17.0],
        "VIP": [False, None, False],
        "RoomService": [None, 150.0, None],
        "FoodCourt": [None, 200.0, None],
        "ShoppingMall": [None, 0.0, None],
        "Spa": [None, 300.0, None],
        "VRDeck": [None, 100.0, None],
        "Name": ["Alice A", "Bob A", "Carol B"],
        "Transported": [True, False, True],
    })


# ---------------------------------------------------------------------------
# extract_cabin_features
# ---------------------------------------------------------------------------

class TestExtractCabinFeatures:
    def test_splits_correctly(self, minimal_df):
        out = extract_cabin_features(minimal_df)
        assert out.loc[0, "Deck"] == "B"
        assert out.loc[0, "CabinNumber"] == 34
        assert out.loc[0, "Side"] == "P"

    def test_handles_null_cabin(self, df_with_nulls):
        out = extract_cabin_features(df_with_nulls)
        assert out.loc[0, "Deck"] == "Unknown"
        assert out.loc[0, "CabinNumber"] == 0
        assert out.loc[0, "Side"] == "Unknown"

    def test_does_not_mutate_input(self, minimal_df):
        original_cols = set(minimal_df.columns)
        extract_cabin_features(minimal_df)
        assert set(minimal_df.columns) == original_cols

    def test_adds_expected_columns(self, minimal_df):
        out = extract_cabin_features(minimal_df)
        assert {"Deck", "CabinNumber", "Side"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# extract_group_features
# ---------------------------------------------------------------------------

class TestExtractGroupFeatures:
    def test_group_size_correct(self, minimal_df):
        out = extract_group_features(minimal_df)
        # 0001_01 y 0001_02 comparten grupo -> size 2
        assert out.loc[0, "GroupSize"] == 2
        assert out.loc[1, "GroupSize"] == 2
        # 0002_01 viaja solo -> size 1
        assert out.loc[2, "GroupSize"] == 1

    def test_travel_group_extracted(self, minimal_df):
        out = extract_group_features(minimal_df)
        assert out.loc[0, "TravelGroup"] == "0001"
        assert out.loc[2, "TravelGroup"] == "0002"

    def test_does_not_mutate_input(self, minimal_df):
        original_shape = minimal_df.shape
        extract_group_features(minimal_df)
        assert minimal_df.shape == original_shape


# ---------------------------------------------------------------------------
# create_spending_features
# ---------------------------------------------------------------------------

class TestCreateSpendingFeatures:
    def test_total_spending_log_positive(self, minimal_df):
        out = create_spending_features(minimal_df)
        # Pasajero 1 tiene gasto -> log > 0
        assert out.loc[1, "TotalSpending_Log"] > 0

    def test_no_spending_gives_zero_log(self, minimal_df):
        out = create_spending_features(minimal_df)
        # Pasajero 0 en CryoSleep, gasto 0 -> log1p(0) = 0
        assert out.loc[0, "TotalSpending_Log"] == pytest.approx(0.0)

    def test_has_spending_flag(self, minimal_df):
        out = create_spending_features(minimal_df)
        assert out.loc[0, "HasSpending"] == 0
        assert out.loc[1, "HasSpending"] == 1

    def test_spending_categories_count(self, minimal_df):
        out = create_spending_features(minimal_df)
        # Pasajero 1: RoomService, FoodCourt, Spa, VRDeck -> 4 servicios
        assert out.loc[1, "SpendingCategories"] == 4


# ---------------------------------------------------------------------------
# create_age_features
# ---------------------------------------------------------------------------

class TestCreateAgeFeatures:
    def test_child_category(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, "Age"] = 10.0
        out = create_age_features(df)
        assert out.loc[0, "AgeCategory"] == "Child"

    def test_adult_category(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, "Age"] = 35.0
        out = create_age_features(df)
        assert out.loc[0, "AgeCategory"] == "Adult"

    def test_senior_category(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, "Age"] = 65.0
        out = create_age_features(df)
        assert out.loc[0, "AgeCategory"] == "Senior"

    def test_null_age_gets_category(self, df_with_nulls):
        df = df_with_nulls.copy()
        df["Age"] = df["Age"].fillna(df["Age"].median())
        out = create_age_features(df)
        assert out["AgeCategory"].notna().all()


# ---------------------------------------------------------------------------
# encode_cryosleep / encode_side
# ---------------------------------------------------------------------------

class TestEncoders:
    def test_encode_cryosleep_true(self, minimal_df):
        # encode_cryosleep es una funcion escalar; se aplica con .map()
        df = extract_cabin_features(minimal_df)
        df["CryoSleep_Encoded"] = df["CryoSleep"].map(encode_cryosleep)
        assert df.loc[0, "CryoSleep_Encoded"] == 1
        assert df.loc[1, "CryoSleep_Encoded"] == 0

    def test_encode_side(self, minimal_df):
        # encode_side es una funcion escalar; se aplica con .map()
        df = extract_cabin_features(minimal_df)
        df["Side_Encoded"] = df["Side"].map(encode_side)
        assert df.loc[0, "Side_Encoded"] == 0   # P -> 0
        assert df.loc[1, "Side_Encoded"] == 1   # S -> 1

    def test_encode_cryosleep_handles_null(self, df_with_nulls):
        df = df_with_nulls.copy()
        df["CryoSleep"] = df["CryoSleep"].astype(object)
        df = extract_cabin_features(df)
        # NaN no esta en el dict de encode_cryosleep -> devuelve NaN -> rellenamos con -1
        df["CryoSleep_Encoded"] = df["CryoSleep"].map(encode_cryosleep).fillna(-1).astype(int)
        # -1 = desconocido (comportamiento esperado para None/NaN)
        assert df.loc[0, "CryoSleep_Encoded"] == -1


# ---------------------------------------------------------------------------
# handle_missing_values_spaceship
# ---------------------------------------------------------------------------

class TestHandleMissingValues:
    def test_no_nulls_after_imputation(self, df_with_nulls):
        df = extract_cabin_features(df_with_nulls)
        df = extract_group_features(df)
        df = create_spending_features(df)
        # impute_age=True para preservar todas las filas (sin dropna)
        df = handle_missing_values_spaceship(df, impute_age=True)
        spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        assert df[spending_cols].isna().sum().sum() == 0

    def test_categorical_nulls_filled(self, df_with_nulls):
        df = extract_cabin_features(df_with_nulls)
        df = extract_group_features(df)
        df = create_spending_features(df)
        df = handle_missing_values_spaceship(df, impute_age=True)
        assert df["HomePlanet"].isna().sum() == 0
        assert df["Destination"].isna().sum() == 0


# ---------------------------------------------------------------------------
# apply_domain_rules
# ---------------------------------------------------------------------------

def _domain_rules_base_df(**overrides) -> pd.DataFrame:
    """DataFrame minimo con todas las columnas requeridas por apply_domain_rules."""
    base = {
        "CryoSleep": [None],
        "RoomService": [0.0],
        "FoodCourt": [0.0],
        "ShoppingMall": [0.0],
        "Spa": [0.0],
        "VRDeck": [0.0],
        "HomePlanet": [None],
        "TravelGroup": ["0001"],
        "Deck": ["B"],
        "Side": ["P"],
        "Age": [30.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestApplyDomainRules:
    def test_cryo_imputes_null_spending_to_zero(self):
        # Regla 4: CryoSleep=True + spending NaN -> spending = 0
        # La funcion imputa NaN; no sobreescribe valores existentes no-NaN
        df = _domain_rules_base_df(
            CryoSleep=[True],
            RoomService=[None],
            Spa=[None],
        )
        out = apply_domain_rules(df)
        assert out.loc[0, "RoomService"] == 0.0
        assert out.loc[0, "Spa"] == 0.0

    def test_spending_sets_cryo_false(self):
        # Regla 5: spending > 0 con CryoSleep=NaN -> CryoSleep = False
        df = _domain_rules_base_df(
            CryoSleep=[None],
            RoomService=[100.0],
        )
        out = apply_domain_rules(df)
        assert out.loc[0, "CryoSleep"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# create_solo_interaction_features
# ---------------------------------------------------------------------------

class TestSoloInteractions:
    def test_is_alone_flag(self, minimal_df):
        # create_solo_interaction_features requiere TotalSpending (de create_spending_features)
        df = extract_group_features(minimal_df)
        df = create_spending_features(df)
        out = create_solo_interaction_features(df)
        assert out.loc[2, "IsAlone"] == 1   # grupo de 1
        assert out.loc[0, "IsAlone"] == 0   # grupo de 2

    def test_is_child_flag(self, minimal_df):
        # IsChild: Age < 13 (umbral del dominio, no 18)
        df = minimal_df.copy()
        df.loc[2, "Age"] = 10.0  # forzar un nino real
        df = extract_group_features(df)
        df = create_spending_features(df)
        out = create_solo_interaction_features(df)
        assert out.loc[2, "IsChild"] == 1   # Age=10 -> child (< 13)
        assert out.loc[1, "IsChild"] == 0   # Age=45 -> no child


# ---------------------------------------------------------------------------
# create_group_context_features
# ---------------------------------------------------------------------------

class TestGroupContextFeatures:
    def test_group_all_cryo(self):
        df = pd.DataFrame({
            "PassengerId": ["0001_01", "0001_02"],
            "TravelGroup": ["0001", "0001"],
            "CryoSleep": [True, True],
            "RoomService": [0.0, 0.0],
            "FoodCourt": [0.0, 0.0],
            "ShoppingMall": [0.0, 0.0],
            "Spa": [0.0, 0.0],
            "VRDeck": [0.0, 0.0],
            "TotalSpending_Log": [0.0, 0.0],
        })
        out = create_group_context_features(df)
        assert out["GroupAllCryo"].iloc[0] == 1
        assert out["GroupAllCryo"].iloc[1] == 1

    def test_group_any_cryo_mixed(self):
        df = pd.DataFrame({
            "PassengerId": ["0001_01", "0001_02"],
            "TravelGroup": ["0001", "0001"],
            "CryoSleep": [True, False],
            "RoomService": [0.0, 100.0],
            "FoodCourt": [0.0, 0.0],
            "ShoppingMall": [0.0, 0.0],
            "Spa": [0.0, 0.0],
            "VRDeck": [0.0, 0.0],
            "TotalSpending_Log": [0.0, 4.6],
        })
        out = create_group_context_features(df)
        assert out["GroupAllCryo"].iloc[0] == 0   # no todos en cryo
        assert out["GroupAnyCryo"].iloc[0] == 1   # al menos uno en cryo
