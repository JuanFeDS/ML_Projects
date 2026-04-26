# Model Card — Exp-009 | LogisticRegression


## Identificacion

- **Experimento:** Exp-009

- **Fecha:** 2026-04-09 23:04

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.7886

- **val_accuracy referencia:** 0.9454

- **Diferencia:** -0.1568

- **Artefacto:** models/experiments/exp-009_LogisticRegression.pkl


## Feature Set

- **Nombre:** fs-009_percentile_cabin

- **Descripcion:** fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor distributional shift entre train y test. La percentil normaliza la posicion relativa dentro del deck, eliminando el shift de rango absoluto.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7886 |
| ROC-AUC (validacion)  |  0.8705 |
| Accuracy (CV 5-fold)  |  0.7895 |


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

- GroupSize

- HasSpending

- SpendingCategories

- TotalSpending_Log

- CabinNumber_DeckPercentile

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
