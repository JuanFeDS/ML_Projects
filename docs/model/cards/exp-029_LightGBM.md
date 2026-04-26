# Model Card — Exp-029 | LightGBM


## Identificacion

- **Experimento:** Exp-029

- **Fecha:** 2026-04-13 23:01

- **Modelo:** LightGBM

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8135

- **val_accuracy referencia:** 0.8181

- **Diferencia:** -0.0046

- **Artefacto:** models/experiments/exp-029_LightGBM.pkl


## Feature Set

- **Nombre:** fs-017_lastname_te

- **Descripcion:** fs-004 + LastName con Target Encoding fold-aware (sklearn TargetEncoder, cv=5). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. El encoding se computa por fold dentro del CV para eliminar leakage en apellidos raros (~80-90% de test surnames son unseen en train). Pipeline interno aplica TargetEncoder(smooth='auto') a LastName y passthrough al resto.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8135 |
| ROC-AUC (validacion)  |  0.9024 |
| Accuracy (CV 5-fold)  |  0.8127 |


## Dataset

- **Features:** 25

- **Muestras de entrenamiento:** 8,514

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

- LastName

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
