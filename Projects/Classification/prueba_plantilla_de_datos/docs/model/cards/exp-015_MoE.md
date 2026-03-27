# Model Card — Exp-015 | MoE


## Identificacion

- **Experimento:** Exp-015

- **Fecha:** 2026-03-26 21:24

- **Modelo:** MoE

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.9495

- **val_accuracy referencia:** 0.8227

- **Diferencia:** +0.1268

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-007_domain_rules

- **Descripcion:** Imputacion por 6 reglas fisicas del dataset + TravelGroup_TE. Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. TravelGroup_TE: tasa de transporte media del grupo de viaje (target encoding).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.9495 |
| ROC-AUC (validacion)  |  0.9928 |
| Accuracy (CV 5-fold)  |  0.953  |


## Hiperparametros

```
iterations: 200
depth: 5
learning_rate: 0.05011896346846189
l2_leaf_reg: 13.20487906665041
bagging_temperature: 0.6246118922833275
```


## Dataset

- **Features:** 36

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

- TravelGroup_TE

- HomePlanet_Earth

- HomePlanet_Europa

- HomePlanet_Mars

- HomePlanet_Unknown

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- Deck_A

- Deck_B

- Deck_C

- Deck_D

- Deck_E

- Deck_F

- Deck_G

- Deck_T

- Deck_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult
