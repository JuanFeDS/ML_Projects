# Model Card — Exp-017 | CatBoost (tuneado)


## Identificacion

- **Experimento:** Exp-017

- **Fecha:** 2026-03-26 22:09

- **Modelo:** CatBoost (tuneado)

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8186

- **val_accuracy referencia:** 0.8227

- **Diferencia:** -0.0041

- **Artefacto:** models/experiments/exp-017_CatBoost_tuneado.pkl


## Feature Set

- **Nombre:** fs-009_percentile_cabin

- **Descripcion:** fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor distributional shift entre train y test. La percentil normaliza la posicion relativa dentro del deck, eliminando el shift de rango absoluto.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8186 |
| ROC-AUC (validacion)  |  0.9036 |
| Accuracy (CV 5-fold)  |  0.8141 |


## Hiperparametros

```
iterations: 600
depth: 9
learning_rate: 0.020589728197687916
l2_leaf_reg: 4.4546743769349115
bagging_temperature: 0.18340450985343382
```


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
