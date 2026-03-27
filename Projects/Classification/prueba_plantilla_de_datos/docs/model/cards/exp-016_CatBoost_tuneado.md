# Model Card — Exp-016 | CatBoost (tuneado)


## Identificacion

- **Experimento:** Exp-016

- **Fecha:** 2026-03-26 21:35

- **Modelo:** CatBoost (tuneado)

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8197

- **val_accuracy referencia:** 0.9495

- **Diferencia:** -0.1298

- **Artefacto:** models/experiments/exp-016_CatBoost_tuneado.pkl


## Feature Set

- **Nombre:** fs-008_domain_rules_only

- **Descripcion:** Imputacion por 6 reglas fisicas del dataset SIN target encoding de grupo. Elimina el leakage de TravelGroup_TE (fs-007). Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. Mismo pipeline que fs-007 pero con el mismo espacio de features que fs-001.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8197 |
| ROC-AUC (validacion)  |  0.9051 |
| Accuracy (CV 5-fold)  |  0.814  |


## Hiperparametros

```
iterations: 600
depth: 7
learning_rate: 0.020589575682537137
l2_leaf_reg: 5.381997322907315
bagging_temperature: 0.24927702441464045
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
