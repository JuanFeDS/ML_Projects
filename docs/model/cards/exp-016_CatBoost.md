# Model Card — Exp-016 | CatBoost


## Identificacion

- **Experimento:** Exp-016

- **Fecha:** 2026-04-10 22:09

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.818

- **val_accuracy referencia:** 0.8233

- **Diferencia:** -0.0053

- **Artefacto:** models/experiments/exp-016_CatBoost.pkl


## Feature Set

- **Nombre:** fs-011_child_route

- **Descripcion:** fs-004 + 4 features dirigidas a los segmentos con mayor error en exp-013: IsChild (binario Age<13), GroupHasChild (grupo tiene algun nino), GroupChildRate (proporcion de ninos en el grupo), Route (HomePlanet+Destination, 9 combinaciones categoricas). Motivacion: ninos (28% error) y PSO J318.5-22 (30% error) son los segmentos mas dificiles; el contexto familiar y la ruta completa aportan informacion que el modelo no capturaba de forma individual.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.818  |
| ROC-AUC (validacion)  |  0.9021 |
| Accuracy (CV 5-fold)  |  0.814  |


## Hiperparametros

```
iterations: 600
depth: 7
learning_rate: 0.0748149645868013
l2_leaf_reg: 9.48753158546387
bagging_temperature: 0.0035936085696097256
```


## Dataset

- **Features:** 43

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

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult

- Route_Earth_55 Cancri e

- Route_Earth_PSO J318.5-22

- Route_Earth_TRAPPIST-1e

- Route_Earth_Unknown

- Route_Europa_55 Cancri e

- Route_Europa_PSO J318.5-22

- Route_Europa_TRAPPIST-1e

- Route_Europa_Unknown

- Route_Mars_55 Cancri e

- Route_Mars_PSO J318.5-22

- Route_Mars_TRAPPIST-1e

- Route_Mars_Unknown

- Route_Unknown_55 Cancri e

- Route_Unknown_PSO J318.5-22

- Route_Unknown_TRAPPIST-1e

- Route_Unknown_Unknown
