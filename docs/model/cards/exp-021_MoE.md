# Model Card — Exp-021 | MoE


## Identificacion

- **Experimento:** Exp-021

- **Fecha:** 2026-04-11 08:38

- **Modelo:** MoE

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8203

- **val_accuracy referencia:** 0.825

- **Diferencia:** -0.0047

- **Artefacto:** models/experiments/exp-021_MoE.pkl


## Feature Set

- **Nombre:** fs-013_group_context

- **Descripcion:** fs-004 + imputacion Age por grupo (mediana del TravelGroup antes de global) + 4 features de contexto colectivo inspiradas en soluciones top Kaggle: GroupAllCryo (todos en CryoSleep: 80.5% vs 42.4% transported), GroupAnyCryo (alguno en CryoSleep: grupos mixtos 60.4%), SpendShare (gasto individual / gasto total grupo, corr=-0.15 entre no-cryo), GroupSpendOthers_Log (gasto del resto del grupo, corr=+0.09).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8203 |
| ROC-AUC (validacion)  |  0.9061 |
| Accuracy (CV 5-fold)  |  0.8112 |


## Hiperparametros

```
iterations: 600
depth: 6
learning_rate: 0.03208899494279826
l2_leaf_reg: 19.802111938430365
bagging_temperature: 0.33293772919177633
```


## Dataset

- **Features:** 28

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
