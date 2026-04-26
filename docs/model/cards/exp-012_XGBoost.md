# Model Card — Exp-012 | XGBoost


## Identificacion

- **Experimento:** Exp-012

- **Fecha:** 2026-04-09 23:42

- **Modelo:** XGBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8209

- **val_accuracy referencia:** 0.8203

- **Diferencia:** +0.0006

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-004_target_encoding

- **Descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8209 |
| ROC-AUC (validacion)  |  0.9023 |
| Accuracy (CV 5-fold)  |  0.8038 |


## Hiperparametros

```
n_estimators: 400
max_depth: 8
learning_rate: 0.04046301461698397
subsample: 0.7884745869448389
colsample_bytree: 0.6548503898583582
reg_alpha: 1.4070712978947977
reg_lambda: 3.4994103768847347
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
