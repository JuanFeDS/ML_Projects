# Model Card — Exp-031 | CatBoost


## Identificacion

- **Experimento:** Exp-031

- **Fecha:** 2026-04-16 21:48

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8352

- **val_accuracy referencia:** 0.8176

- **Diferencia:** +0.0176

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-019_pseudo_labeled

- **Descripcion:** fs-017 con pseudo-labeling: entrenado sobre train.csv + 985 filas de test.csv donde exp-027 predice con confianza >= 0.95 (711 True, 274 False, confianza media 97.6%). Pipeline idéntico a fs-017; diferencia es el tamaño del dataset de entrenamiento (8,693 → 9,678 filas, +11.3%).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8352 |
| ROC-AUC (validacion)  |  0.9246 |
| Accuracy (CV 5-fold)  |  0.8347 |


## Hiperparametros

```
iterations: 400
depth: 4
learning_rate: 0.11971677241936392
l2_leaf_reg: 13.688056542097758
bagging_temperature: 0.9799558910910812
```


## Dataset

- **Features:** 25

- **Muestras de entrenamiento:** 9,472

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

- LastName

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
