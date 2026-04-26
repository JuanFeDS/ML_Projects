# Model Card — Exp-006 | LogisticRegression


## Identificacion

- **Experimento:** Exp-006

- **Fecha:** 2026-04-09 23:01

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.7874

- **val_accuracy referencia:** 0.7904

- **Diferencia:** -0.0030

- **Artefacto:** models/experiments/exp-006_LogisticRegression.pkl


## Feature Set

- **Nombre:** fs-006_group_imputation

- **Descripcion:** fs-001 con imputacion group-aware para columnas de gasto. Pasajeros no-cryo con spending NaN reciben la mediana del TravelGroup (en lugar de 0), haciendo que TotalSpending_Log capture mejor su perfil real. El orden del pipeline cambia: fill categoricals → impute → create_spending_features.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7874 |
| ROC-AUC (validacion)  |  0.8685 |
| Accuracy (CV 5-fold)  |  0.7893 |


## Dataset

- **Features:** 35

- **Muestras de entrenamiento:** 6,811

- **Estrategia de validacion:** StratifiedKFold (5 folds) + hold-out 20%


## Features del modelo

- Age

- RoomService

- FoodCourt

- ShoppingMall

- Spa

- VRDeck

- CabinNumber

- GroupSize

- HasSpending

- SpendingCategories

- TotalSpending_Log

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
