# Feature Engineering — Spaceship Titanic


## Feature Set

- **Nombre:** fs-009_percentile_cabin

- **Descripcion:** fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor distributional shift entre train y test. La percentil normaliza la posicion relativa dentro del deck, eliminando el shift de rango absoluto.

- **Hereda de:** fs-008_domain_rules_only


## Contexto

Pipeline de feature engineering para el feature set `fs-009_percentile_cabin`. Ejecuta encoding y escalado estandar sobre las features resultantes.


## Paso 1 — Pipeline del feature set

- **Shape entrada:** 8,693 x 14

- **Shape despues:** 8,514 x 25

- **Filas eliminadas (Age nulo):** 179


## Paso 2 — Label Encoding (CryoSleep, Side)

- **CryoSleep_Encoded:** True->1, False->0, Unknown->-1

- **Side_Encoded:** P->0, S->1, Unknown->-1


## Paso 3b — One-Hot Encoding

- **Columnas OHE generadas:** 22

- HomePlanet_Earth

- HomePlanet_Europa

- HomePlanet_Mars

- HomePlanet_Unknown

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- Deck_A

- Deck_B

- Deck_C

- Deck_D

- Deck_E

- Deck_F

- Deck_G

- Deck_T

- Deck_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult


## Paso 4 — Drop de columnas

- **Columnas eliminadas:** 11

- PassengerId

- Name

- Cabin

- TravelGroup

- CryoSleep

- VIP

- Side

- TotalSpending

- CabinNumber

- CryoSleep

- Side

- **Shape tras drop:** 8,514 x 36


## Paso 5 — Separacion X / y

- **Filas:** 8,514

- **Features:** 35

**Lista completa de features resultantes:**

- Age

- RoomService

- FoodCourt

- ShoppingMall

- Spa

- VRDeck

- GroupSize

- HasSpending

- SpendingCategories

- TotalSpending_Log

- CabinNumber_DeckPercentile

- CryoSleep_Encoded

- Side_Encoded

- HomePlanet_Earth

- HomePlanet_Europa

- HomePlanet_Mars

- HomePlanet_Unknown

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- Deck_A

- Deck_B

- Deck_C

- Deck_D

- Deck_E

- Deck_F

- Deck_G

- Deck_T

- Deck_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult


## Paso 6 — StandardScaler

- **Tipo:** StandardScaler

- **Fit sobre:** Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, GroupSize, CabinNumber_DeckPercentile, TotalSpending_Log, SpendingCategories

- **Medias (primeras 5):** 28.828, 221.148, 449.298, 171.044, 303.637

- **Desv. std (primeras 5):** 14.488, 663.900, 1598.314, 602.342, 1120.667


## Resumen del Pipeline

- **Feature set:** fs-009_percentile_cabin

- **Shape entrada:** 8,693 x 14

- **Shape final (X):** 8,514 x 35

- **Filas eliminadas:** 179 (Age nulo, 2.06%)

- **Nulos residuales en X:** 0

- **Features resultantes:** 35


## Archivos Generados

- `C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects\Projects\Classification\prueba_plantilla_de_datos\data\features\train_features_fs-009_percentile_cabin.csv` — X + Transported, sin escalar

- `C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects\Projects\Classification\prueba_plantilla_de_datos\data\features\train_scaled_fs-009_percentile_cabin.csv` — X escalado + Transported

- `C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects\Projects\Classification\prueba_plantilla_de_datos\models\scaler_fs-009_percentile_cabin.pkl` — StandardScaler serializado (joblib)
