# Model Card — Exp-001 | LogisticRegression


## Identificacion

- **Experimento:** Exp-001

- **Fecha:** 2026-04-09 21:36

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **Nota:** Primer experimento — sin referencia previa

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-001_baseline

- **Descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7886 |
| ROC-AUC (validacion)  |  0.8692 |
| Accuracy (CV 5-fold)  |  0.789  |


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
