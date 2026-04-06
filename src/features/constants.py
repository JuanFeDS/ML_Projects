"""
Constantes del dominio Spaceship Titanic.

NOTA: NUMERIC_FEATURES, CATEGORICAL_COLS y FEATURES_TO_DROP han sido
movidos a src/features/feature_sets.py, donde viven asociados a cada
FeatureSetConfig. Este archivo conserva solo constantes del dominio
que no cambian entre experimentos.
"""

TARGET: str = "Transported"

# Features originales del dataset crudo (antes de ingenieria)
RAW_NUMERIC: list[str] = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

RAW_CATEGORICAL: list[str] = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
]

SPENDING_COLS: list[str] = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]
