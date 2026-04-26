# Model Card — Exp-005 | LogisticRegression


## Identificacion

- **Experimento:** Exp-005

- **Fecha:** 2026-04-09 23:00

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.7886

- **val_accuracy referencia:** 0.7904

- **Diferencia:** -0.0018

- **Artefacto:** models/experiments/exp-005_LogisticRegression.pkl


## Feature Set

- **Nombre:** fs-005_structural_context

- **Descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7886 |
| ROC-AUC (validacion)  |  0.8701 |
| Accuracy (CV 5-fold)  |  0.7991 |


## Dataset

- **Features:** 42

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

- SpendingEntropy

- GroupSpendingZScore

- CabinNeighborhoodDensity

- FamilySizeFromName

- GroupCryoAlignment

- GroupAgeDispersion

- CryoSleep_Encoded

- Side_Encoded

- SpendingCategoryProfile_TE

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
