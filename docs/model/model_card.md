# Model Card — Spaceship Titanic


## Modelo

- **Nombre:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)

- **Numero de features:** 25

- **Muestras de entrenamiento:** 9,472


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


## Validacion y artefactos

- **Estrategia:** StratifiedKFold (5 folds) + hold-out 20%

- **Fecha de entrenamiento:** 2026-04-16

- **Archivo del modelo:** models/production/best_model.pkl

- **Scaler:** models/production/scaler.pkl
