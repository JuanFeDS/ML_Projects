# Model Card — Exp-014 | CatBoost


## Identificacion

- **Experimento:** Exp-014

- **Fecha:** 2026-04-09 23:57

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8233

- **val_accuracy referencia:** 0.8221

- **Diferencia:** +0.0012

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-005_structural_context

- **Descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8233 |
| ROC-AUC (validacion)  |  0.9022 |
| Accuracy (CV 5-fold)  |  0.8137 |


## Hiperparametros

```
iterations: 600
depth: 6
learning_rate: 0.04569223593850866
l2_leaf_reg: 17.920260567511946
bagging_temperature: 0.6047914680041317
```


## Dataset

- **Features:** 42

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

- SpendingEntropy

- GroupSpendingZScore

- CabinNeighborhoodDensity

- FamilySizeFromName

- GroupCryoAlignment

- GroupAgeDispersion

- CryoSleep_Encoded

- Side_Encoded

- SpendingCategoryProfile_TE

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
