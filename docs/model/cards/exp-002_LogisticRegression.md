# Model Card — Exp-002 | LogisticRegression


## Identificacion

- **Experimento:** Exp-002

- **Fecha:** 2026-04-09 22:54

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.7868

- **val_accuracy referencia:** 0.7886

- **Diferencia:** -0.0018

- **Artefacto:** models/experiments/exp-002_LogisticRegression.pkl


## Feature Set

- **Nombre:** fs-002_cryo_interactions

- **Descripcion:** fs-001 + Route (HomePlanet+Destination), GroupCryoSleepRate, CryoSleepViolation, LuxurySpendingRatio, CabinNumber_DeckPercentile, GroupSpendingMean. Referencia: Exp-007 (val_accuracy=0.8156, no supero fs-001).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7868 |
| ROC-AUC (validacion)  |  0.8694 |
| Accuracy (CV 5-fold)  |  0.7896 |


## Dataset

- **Features:** 56

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

- GroupCryoSleepRate

- CryoSleepViolation

- LuxurySpendingRatio

- CabinNumber_DeckPercentile

- GroupSpendingMean

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

- Route_Earth_to_55 Cancri e

- Route_Earth_to_PSO J318.5-22

- Route_Earth_to_TRAPPIST-1e

- Route_Earth_to_Unknown

- Route_Europa_to_55 Cancri e

- Route_Europa_to_PSO J318.5-22

- Route_Europa_to_TRAPPIST-1e

- Route_Europa_to_Unknown

- Route_Mars_to_55 Cancri e

- Route_Mars_to_PSO J318.5-22

- Route_Mars_to_TRAPPIST-1e

- Route_Mars_to_Unknown

- Route_Unknown_to_55 Cancri e

- Route_Unknown_to_PSO J318.5-22

- Route_Unknown_to_TRAPPIST-1e

- Route_Unknown_to_Unknown
