# Model Card — Exp-003 | LogisticRegression


## Identificacion

- **Experimento:** Exp-003

- **Fecha:** 2026-04-09 22:57

- **Modelo:** LogisticRegression

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.7892

- **val_accuracy referencia:** 0.7886

- **Diferencia:** +0.0006

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-003_solo_interactions

- **Descripcion:** fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), SpendingIntensity (TotalSpending/(SpendingCategories+1)). Features simples de alta senal, sin riesgo de multicolinealidad.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.7892 |
| ROC-AUC (validacion)  |  0.8702 |
| Accuracy (CV 5-fold)  |  0.7896 |


## Dataset

- **Features:** 38

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

- IsAlone

- IsChild

- SpendingIntensity

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
