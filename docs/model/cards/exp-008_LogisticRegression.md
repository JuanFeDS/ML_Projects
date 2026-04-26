# Model Card — Exp-008 | LogisticRegression


## Identificacion

- **Experimento:** Exp-008

- **Fecha:** 2026-04-09 23:03

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.7898

- **val_accuracy referencia:** 0.9454

- **Diferencia:** -0.1556

- **Artefacto:** models/experiments/exp-008_LogisticRegression.pkl


## Feature Set

- **Nombre:** fs-008_domain_rules_only

- **Descripcion:** Imputacion por 6 reglas fisicas del dataset SIN target encoding de grupo. Elimina el leakage de TravelGroup_TE (fs-007). Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. Mismo pipeline que fs-007 pero con el mismo espacio de features que fs-001.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7898 |
| ROC-AUC (validacion)  |  0.8703 |
| Accuracy (CV 5-fold)  |  0.7893 |


## Dataset

- **Features:** 35

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
