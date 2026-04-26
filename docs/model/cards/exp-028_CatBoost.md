# Model Card — Exp-028 | CatBoost


## Identificacion

- **Experimento:** Exp-028

- **Fecha:** 2026-04-13 22:41

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8181

- **val_accuracy referencia:** 0.8176

- **Diferencia:** +0.0005

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-018_group_consistency

- **Descripcion:** fs-017 + 3 features de coherencia interna de grupo (sin usar target). GroupAllSameDest: 1 si todos en el grupo comparten Destination. GroupAllSameHomePlanet: 1 si todos comparten HomePlanet. GroupConsistencyScore: suma ordinal 0-2 de cohesión grupal. Complementa LastName fold-aware TE con señal estructural del grupo.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8181 |
| ROC-AUC (validacion)  |  0.9073 |
| Accuracy (CV 5-fold)  |  0.8158 |


## Hiperparametros

```
iterations: 600
depth: 5
learning_rate: 0.08242047873638254
l2_leaf_reg: 16.675777579204556
bagging_temperature: 0.25858137370944156
```


## Dataset

- **Features:** 28

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

- LastName

- GroupAllSameDest

- GroupAllSameHomePlanet

- GroupConsistencyScore

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
