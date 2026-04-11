"""Tests unitarios para src/features/feature_sets/.

Cubre el registro FEATURE_SETS, FeatureSetConfig y los pipelines principales.
Todos los tests usan DataFrames minimos construidos en memoria.
"""
import pandas as pd
import pytest

from src.features.feature_sets.config import FeatureSetConfig
from src.features.feature_sets.registry import FEATURE_SETS, DEFAULT_FEATURE_SET


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Dataset crudo minimo que simula train.csv."""
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01", "0003_02"],
        "Cabin": ["B/0034/P", "B/0035/S", "G/0100/P", "C/0010/S", "C/0011/P"],
        "HomePlanet": ["Europa", "Europa", "Earth", "Europa", "Europa"],
        "CryoSleep": [True, False, False, True, False],
        "Destination": ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e", "55 Cancri e"],
        "Age": [23.0, 45.0, 17.0, 8.0, 35.0],
        "VIP": [False, False, False, False, True],
        "RoomService": [0.0, 150.0, 0.0, 0.0, 200.0],
        "FoodCourt": [0.0, 200.0, 50.0, 0.0, 100.0],
        "ShoppingMall": [0.0, 0.0, 0.0, 0.0, 50.0],
        "Spa": [0.0, 300.0, 0.0, 0.0, 75.0],
        "VRDeck": [0.0, 100.0, 0.0, 0.0, 125.0],
        "Name": ["Alice A", "Bob A", "Carol B", "Dave C", "Eve C"],
        "Transported": [True, False, True, True, False],
    })


# ---------------------------------------------------------------------------
# Registro y FeatureSetConfig
# ---------------------------------------------------------------------------

class TestFeatureSetRegistry:
    def test_feature_sets_not_empty(self):
        assert len(FEATURE_SETS) > 0

    def test_all_entries_are_feature_set_config(self):
        for name, fs in FEATURE_SETS.items():
            assert isinstance(fs, FeatureSetConfig), f"{name} no es FeatureSetConfig"

    def test_default_feature_set_in_registry(self):
        assert DEFAULT_FEATURE_SET in FEATURE_SETS

    def test_active_feature_sets_exist(self):
        active_ids = [
            "fs-001_baseline",
            "fs-003_solo_interactions",
            "fs-004_target_encoding",
            "fs-005_structural_context",
            "fs-010_cryo_spending",
            "fs-011_child_route",
            "fs-012_child_route_te",
            "fs-013_group_context",
        ]
        for fs_id in active_ids:
            assert fs_id in FEATURE_SETS, f"{fs_id} no esta en el registro"

    def test_deprecated_feature_sets_are_flagged(self):
        deprecated_ids = [
            "fs-002_cryo_interactions",
            "fs-006_group_imputation",
            "fs-007_domain_rules",
            "fs-008_domain_rules_only",
            "fs-009_percentile_cabin",
        ]
        for fs_id in deprecated_ids:
            assert FEATURE_SETS[fs_id].deprecated, f"{fs_id} deberia ser deprecated"

    def test_active_feature_sets_not_deprecated(self):
        active_ids = [
            "fs-001_baseline",
            "fs-013_group_context",
        ]
        for fs_id in active_ids:
            assert not FEATURE_SETS[fs_id].deprecated, f"{fs_id} no deberia ser deprecated"

    def test_each_entry_has_description(self):
        for name, fs in FEATURE_SETS.items():
            assert fs.description, f"{name} no tiene description"

    def test_each_entry_has_callable_pipelines(self):
        for name, fs in FEATURE_SETS.items():
            assert callable(fs.pipeline), f"{name}.pipeline no es callable"
            assert callable(fs.test_pipeline), f"{name}.test_pipeline no es callable"

    def test_numeric_and_categorical_disjoint(self):
        """Ninguna feature debe estar en numéricas y categóricas al mismo tiempo."""
        for name, fs in FEATURE_SETS.items():
            overlap = set(fs.numeric_features) & set(fs.categorical_cols)
            assert not overlap, (
                f"{name}: features en ambas listas: {overlap}"
            )


# ---------------------------------------------------------------------------
# Pipeline fs-001
# ---------------------------------------------------------------------------

class TestPipelineFs001:
    def test_output_shape_has_more_cols_than_input(self, raw_df):
        fs = FEATURE_SETS["fs-001_baseline"]
        out = fs.pipeline(raw_df)
        assert out.shape[1] > raw_df.shape[1]

    def test_expected_derived_columns_present(self, raw_df):
        fs = FEATURE_SETS["fs-001_baseline"]
        out = fs.pipeline(raw_df)
        expected = {"Deck", "CabinNumber", "Side", "TravelGroup", "GroupSize",
                    "TotalSpending", "TotalSpending_Log", "HasSpending",
                    "SpendingCategories", "AgeCategory"}
        assert expected.issubset(out.columns), (
            f"Columnas faltantes: {expected - set(out.columns)}"
        )

    def test_no_nulls_in_spending_after_pipeline(self, raw_df):
        fs = FEATURE_SETS["fs-001_baseline"]
        out = fs.pipeline(raw_df)
        spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        assert out[spending_cols].isna().sum().sum() == 0

    def test_no_null_age_dropped_for_train(self, raw_df):
        """Sin impute_age, filas con Age NaN se eliminan."""
        df = raw_df.copy()
        df.loc[0, "Age"] = None
        fs = FEATURE_SETS["fs-001_baseline"]
        out = fs.pipeline(df)
        assert out["Age"].isna().sum() == 0
        assert len(out) < len(df)

    def test_test_pipeline_preserves_all_rows(self, raw_df):
        """test_pipeline imputa Age con mediana, no elimina filas."""
        df = raw_df.copy()
        df.loc[0, "Age"] = None
        fs = FEATURE_SETS["fs-001_baseline"]
        out = fs.test_pipeline(df)
        assert len(out) == len(df)
        assert out["Age"].isna().sum() == 0

    def test_input_not_mutated(self, raw_df):
        original_cols = set(raw_df.columns)
        fs = FEATURE_SETS["fs-001_baseline"]
        fs.pipeline(raw_df)
        assert set(raw_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Pipeline fs-013 (grupo + context features)
# ---------------------------------------------------------------------------

class TestPipelineFs013:
    def test_group_context_cols_added(self, raw_df):
        fs = FEATURE_SETS["fs-013_group_context"]
        out = fs.pipeline(raw_df)
        group_cols = {"GroupAllCryo", "GroupAnyCryo", "SpendShare", "GroupSpendOthers_Log"}
        assert group_cols.issubset(out.columns), (
            f"Columnas de grupo faltantes: {group_cols - set(out.columns)}"
        )

    def test_group_all_cryo_binary(self, raw_df):
        """GroupAllCryo solo debe contener 0 y 1."""
        fs = FEATURE_SETS["fs-013_group_context"]
        out = fs.pipeline(raw_df)
        assert set(out["GroupAllCryo"].unique()).issubset({0, 1})

    def test_spend_share_between_0_and_1(self, raw_df):
        """SpendShare es una proporcion; debe estar en [0, 1]."""
        fs = FEATURE_SETS["fs-013_group_context"]
        out = fs.pipeline(raw_df)
        assert (out["SpendShare"] >= 0).all()
        assert (out["SpendShare"] <= 1).all()
