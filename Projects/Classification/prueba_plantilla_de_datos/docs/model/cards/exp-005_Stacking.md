# Model Card — Exp-005 | Stacking


## Identificacion

- **Experimento:** Exp-005

- **Fecha:** 2026-03-26 13:12

- **Modelo:** Stacking

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8115

- **val_accuracy referencia:** 0.8115

- **Diferencia:** +0.0000

- **Artefacto:** models/experiments/exp-005_Stacking.pkl


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8115 |
| ROC-AUC (validacion)  |  0.8991 |
| Accuracy (CV 5-fold)  |  0.8105 |


## Hiperparametros

```
min_samples_leaf: 10
max_leaf_nodes: 15
max_iter: 200
max_depth: 5
learning_rate: 0.05
l2_regularization: 1.0
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
