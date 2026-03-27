# Model Card — Exp-009 | CatBoost (tuneado)


## Identificacion

- **Experimento:** Exp-009

- **Fecha:** 2026-03-26 15:36

- **Modelo:** CatBoost (tuneado)

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8174

- **val_accuracy referencia:** 0.8227

- **Diferencia:** -0.0053

- **Artefacto:** models/experiments/exp-009_CatBoost_tuneado.pkl


## Feature Set

- **Nombre:** fs-004_target_encoding

- **Descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8174 |
| ROC-AUC (validacion)  |  0.9079 |
| Accuracy (CV 5-fold)  |  0.8109 |


## Hiperparametros

```
learning_rate: 0.2
l2_leaf_reg: 10
iterations: 100
depth: 6
bagging_temperature: 1.0
```


## Dataset

- **Features:** 24

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
