# Model Card — Exp-011 | LightGBM


## Identificacion

- **Experimento:** Exp-011

- **Fecha:** 2026-04-09 23:40

- **Modelo:** LightGBM

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8203

- **val_accuracy referencia:** 0.805

- **Diferencia:** +0.0153

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-004_target_encoding

- **Descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8203 |
| ROC-AUC (validacion)  |  0.9046 |
| Accuracy (CV 5-fold)  |  0.8084 |


## Hiperparametros

```
n_estimators: 200
max_depth: 5
learning_rate: 0.09371983213998145
num_leaves: 53
subsample: 0.6216986075761003
colsample_bytree: 0.6535783131989047
reg_alpha: 1.4649640711267433
```


## Dataset

- **Features:** 24

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
