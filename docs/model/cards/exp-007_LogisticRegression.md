# Model Card — Exp-007 | LogisticRegression


## Identificacion

- **Experimento:** Exp-007

- **Fecha:** 2026-04-09 23:02

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.9454

- **val_accuracy referencia:** 0.7904

- **Diferencia:** +0.1550

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-007_domain_rules

- **Descripcion:** Imputacion por 6 reglas fisicas del dataset + TravelGroup_TE. Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. TravelGroup_TE: tasa de transporte media del grupo de viaje (target encoding).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.9454 |
| ROC-AUC (validacion)  |  0.9912 |
| Accuracy (CV 5-fold)  |  0.9493 |


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
