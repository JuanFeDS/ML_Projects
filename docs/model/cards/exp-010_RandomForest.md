# Model Card — Exp-010 | RandomForest


## Identificacion

- **Experimento:** Exp-010

- **Fecha:** 2026-04-09 23:07

- **Modelo:** RandomForest

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.805

- **val_accuracy referencia:** 0.9454

- **Diferencia:** -0.1404

- **Artefacto:** models/experiments/exp-010_RandomForest.pkl


## Feature Set

- **Nombre:** fs-004_target_encoding

- **Descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.805  |
| ROC-AUC (validacion)  |  0.8871 |
| Accuracy (CV 5-fold)  |  0.8031 |


## Hiperparametros

```
n_estimators: 400
max_depth: None
min_samples_split: 8
max_features: log2
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
