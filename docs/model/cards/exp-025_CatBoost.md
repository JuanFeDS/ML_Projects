# Model Card — Exp-025 | CatBoost


## Identificacion

- **Experimento:** Exp-025

- **Fecha:** 2026-04-13 21:31

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8182

- **val_accuracy referencia:** 0.825

- **Diferencia:** -0.0068

- **Artefacto:** models/experiments/exp-025_CatBoost.pkl


## Feature Set

- **Nombre:** fs-004_target_encoding

- **Descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding. Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan información ordinal que OHE no captura.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8182 |
| ROC-AUC (validacion)  |  0.9076 |
| Accuracy (CV 5-fold)  |  0.8154 |


## Hiperparametros

```
iterations: 600
depth: 5
learning_rate: 0.13570390610987357
l2_leaf_reg: 19.825773038488364
bagging_temperature: 0.9960169893612135
```


## Dataset

- **Features:** 24

- **Muestras de entrenamiento:** 8,514

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

- Deck_TE

- HomePlanet_TE

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult
