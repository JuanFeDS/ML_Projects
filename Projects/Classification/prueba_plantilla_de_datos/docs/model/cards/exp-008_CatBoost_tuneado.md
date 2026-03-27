# Model Card — Exp-008 | CatBoost (tuneado)


## Identificacion

- **Experimento:** Exp-008

- **Fecha:** 2026-03-26 15:10

- **Modelo:** CatBoost (tuneado)

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8186

- **val_accuracy referencia:** 0.8227

- **Diferencia:** -0.0041

- **Artefacto:** models/experiments/exp-008_CatBoost_tuneado.pkl


## Feature Set

- **Nombre:** fs-003_solo_interactions

- **Descripcion:** fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), SpendingIntensity (TotalSpending/(SpendingCategories+1)). Features simples de alta señal, sin riesgo de multicolinealidad.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8186 |
| ROC-AUC (validacion)  |  0.9034 |
| Accuracy (CV 5-fold)  |  0.8127 |


## Hiperparametros

```
learning_rate: 0.1
l2_leaf_reg: 10
iterations: 200
depth: 6
bagging_temperature: 0.0
```


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
