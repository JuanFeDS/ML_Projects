# Model Card — Exp-017 | CatBoost


## Identificacion

- **Experimento:** Exp-017

- **Fecha:** 2026-04-10 22:17

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.825

- **val_accuracy referencia:** 0.8233

- **Diferencia:** +0.0017

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-012_child_route_te

- **Descripcion:** fs-011 con Route como Target Encoding en lugar de OHE. Motivacion: PSO J318.5-22 tiene 31% error y su tasa de transporte difiere mucho segun HomePlanet de origen. Route_TE codifica la tasa media de transporte por ruta (una columna numerica vs 9 OHE), capturando la señal ordinal que OHE no puede expresar. Features: IsChild, GroupHasChild, GroupChildRate (contexto familiar) + Deck_TE, HomePlanet_TE, Route_TE (3 target encodings).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.825  |
| ROC-AUC (validacion)  |  0.9061 |
| Accuracy (CV 5-fold)  |  0.814  |


## Hiperparametros

```
iterations: 400
depth: 7
learning_rate: 0.05567776420011066
l2_leaf_reg: 12.048063218106869
bagging_temperature: 0.7532380708632854
```


## Dataset

- **Features:** 28

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

- IsChild

- GroupHasChild

- GroupChildRate

- CryoSleep_Encoded

- Side_Encoded

- Deck_TE

- HomePlanet_TE

- Route_TE

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult
