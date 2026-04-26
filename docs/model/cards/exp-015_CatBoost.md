# Model Card — Exp-015 | CatBoost


## Identificacion

- **Experimento:** Exp-015

- **Fecha:** 2026-04-10 00:14

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** ❌ No supero al modelo actual

- **val_accuracy este run:** 0.815

- **val_accuracy referencia:** 0.8233

- **Diferencia:** -0.0083

- **Artefacto:** models/experiments/exp-015_CatBoost.pkl


## Feature Set

- **Nombre:** fs-010_cryo_spending

- **Descripcion:** fs-004 + 4 features de interaccion CryoSleep x spending: CryoSpendingAnomaly (gasto cuando CryoSleep=True, anomalia fisica), GroupTransportedProxy (ratio de miembros del grupo sin gasto), SideSpendingDiff (asimetria de gasto entre lados P/S de la cabina), CryoSleepBinary (CryoSleep como numerico 1/0/-1). Motivacion: los errores mas frecuentes estan en pasajeros con patrones de gasto inconsistentes con su estado CryoSleep.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.815  |
| ROC-AUC (validacion)  |  0.9045 |
| Accuracy (CV 5-fold)  |  0.8102 |


## Hiperparametros

```
iterations: 600
depth: 9
learning_rate: 0.020589728197687916
l2_leaf_reg: 4.4546743769349115
bagging_temperature: 0.18340450985343382
```


## Dataset

- **Features:** 28

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

- CryoSpendingAnomaly

- GroupTransportedProxy

- SideSpendingDiff

- CryoSleepBinary

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
