# Model Card — Exp-011 | CatBoost (tuneado)


## Identificacion

- **Experimento:** Exp-011

- **Fecha:** 2026-03-26 17:51

- **Modelo:** CatBoost (tuneado)

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8168

- **val_accuracy referencia:** 0.8227

- **Diferencia:** -0.0059

- **Artefacto:** models/experiments/exp-011_CatBoost_tuneado.pkl


## Feature Set

- **Nombre:** fs-005_structural_context

- **Descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8168 |
| ROC-AUC (validacion)  |  0.9039 |
| Accuracy (CV 5-fold)  |  0.8137 |


## Hiperparametros

```
learning_rate: 0.05
l2_leaf_reg: 5
iterations: 400
depth: 6
bagging_temperature: 0.5
```


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
