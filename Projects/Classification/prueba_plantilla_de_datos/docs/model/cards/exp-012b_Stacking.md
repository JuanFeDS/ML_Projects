# Model Card — Exp-012 | Stacking


## Identificacion

- **Experimento:** Exp-012

- **Fecha:** 2026-03-26 20:24

- **Modelo:** Stacking

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8227

- **val_accuracy referencia:** 0.8227

- **Diferencia:** +0.0000

- **Artefacto:** models/experiments/exp-012_Stacking.pkl


## Feature Set

- **Nombre:** fs-001_baseline

- **Descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8227 |
| ROC-AUC (validacion)  |  0.9052 |
| Accuracy (CV 5-fold)  |  0.8128 |


## Hiperparametros

```
iterations: 600
depth: 5
learning_rate: 0.05524411444827968
l2_leaf_reg: 8.778303618018892
bagging_temperature: 0.6339269511583692
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
