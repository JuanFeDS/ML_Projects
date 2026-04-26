# Model Card — Exp-027 | CatBoost


## Identificacion

- **Experimento:** Exp-027

- **Fecha:** 2026-04-13 22:29

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.8176

- **val_accuracy referencia:** 0.8575

- **Diferencia:** -0.0399

- **Artefacto:** models/experiments/exp-027_CatBoost.pkl


## Feature Set

- **Nombre:** fs-017_lastname_te

- **Descripcion:** fs-004 + LastName con Target Encoding fold-aware (sklearn TargetEncoder, cv=5). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. El encoding se computa por fold dentro del CV para eliminar leakage en apellidos raros (~80-90% de test surnames son unseen en train). Pipeline interno aplica TargetEncoder(smooth='auto') a LastName y passthrough al resto.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8176 |
| ROC-AUC (validacion)  |  0.9066 |
| Accuracy (CV 5-fold)  |  0.8149 |


## Hiperparametros

```
iterations: 600
depth: 6
learning_rate: 0.021652690184947157
l2_leaf_reg: 16.121101290648085
bagging_temperature: 0.003097017597283175
```


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
