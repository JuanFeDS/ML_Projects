# Model Card — Exp-024 | CatBoost


## Identificacion

- **Experimento:** Exp-024

- **Fecha:** 2026-04-12 21:10

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8233

- **val_accuracy referencia:** 0.825

- **Diferencia:** -0.0017

- **Artefacto:** models/experiments/exp-024_CatBoost.pkl


## Feature Set

- **Nombre:** fs-016_transductive

- **Descripcion:** fs-015 con imputación transductiva: train.csv + test.csv se combinan antes de aplicar apply_domain_rules e impute_age_by_group, de modo que grupos con miembros en ambos splits comparten información. Requiere ejecutar scripts/00_transductive_impute.py primero y pasar --train-path data/processed/train_transductive.csv a 02_features.py. Misma dimensionalidad que fs-004/fs-015.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8233 |
| ROC-AUC (validacion)  |  0.9044 |
| Accuracy (CV 5-fold)  |  0.8067 |


## Hiperparametros

```
iterations: 400
depth: 8
learning_rate: 0.04893857897388962
l2_leaf_reg: 18.10519218594773
bagging_temperature: 0.3208501786198171
```


## Dataset

- **Features:** 24

- **Muestras de entrenamiento:** 6,876

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
