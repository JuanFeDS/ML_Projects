# Model Card — Exp-022 | CatBoost


## Identificacion

- **Experimento:** Exp-022

- **Fecha:** 2026-04-12 18:45

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8145

- **val_accuracy referencia:** 0.825

- **Diferencia:** -0.0105

- **Artefacto:** models/experiments/exp-022_CatBoost.pkl


## Feature Set

- **Nombre:** fs-014_spend_clusters

- **Descripcion:** fs-013 + 6 features derivadas del EDA 2026-04-12: EntertainmentSpend_Log (FoodCourt+VRDeck+Spa, r inter=0.42-0.46), ComfortSpend_Log (RoomService+ShoppingMall, r=0.36), EntVsComfort_Ratio, IsExtremeSpender (any service > p99 training), AgeVsPlanetMedian (Earth=23/Europa=33/Mars=28), GroupCryoSegment ordinal 0-3 (NoCryo 33.9% / Solo 45.2% / AnyCryo 60.4% / AllCryo 92.2%).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8145 |
| ROC-AUC (validacion)  |  0.8996 |
| Accuracy (CV 5-fold)  |  0.8112 |


## Hiperparametros

```
iterations: 600
depth: 5
learning_rate: 0.030265805407797112
l2_leaf_reg: 18.741644709165087
bagging_temperature: 0.5680093335492117
```


## Dataset

- **Features:** 34

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

- GroupAllCryo

- GroupAnyCryo

- SpendShare

- GroupSpendOthers_Log

- EntertainmentSpend_Log

- ComfortSpend_Log

- EntVsComfort_Ratio

- IsExtremeSpender

- AgeVsPlanetMedian

- GroupCryoSegment

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
