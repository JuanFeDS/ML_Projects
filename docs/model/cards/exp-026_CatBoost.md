# Model Card — Exp-026 | CatBoost


## Identificacion

- **Experimento:** Exp-026

- **Fecha:** 2026-04-13 21:57

- **Modelo:** CatBoost

- **Tipo:** Clasificacion binaria

- **Target:** Transported (True/False)


## Estado

- **Resultado:** 🏆 Promovido a produccion

- **val_accuracy este run:** 0.8575

- **val_accuracy referencia:** 0.825

- **Diferencia:** +0.0325

- **Artefacto:** models/production/best_model.pkl


## Feature Set

- **Nombre:** fs-017_lastname_te

- **Descripcion:** fs-004 + LastName con Target Encoding suavizado (k=30). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. TE suavizado bayesiano: TE=(n*mean+30*global_mean)/(n+30) previene leakage en apellidos raros (n=1-2 → TE≈global_mean≈0.50).


## Metricas de rendimiento

| Metrica               |   Valor |
|:----------------------|--------:|
| Accuracy (validacion) |  0.8575 |
| ROC-AUC (validacion)  |  0.9379 |
| Accuracy (CV 5-fold)  |  0.8547 |


## Hiperparametros

```
iterations: 600
depth: 6
learning_rate: 0.06984284982015783
l2_leaf_reg: 17.09292634438464
bagging_temperature: 0.007697741191395324
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

- CryoSleep_Encoded

- Side_Encoded

- Deck_TE

- HomePlanet_TE

- LastName_TE

- Destination_55 Cancri e

- Destination_PSO J318.5-22

- Destination_TRAPPIST-1e

- Destination_Unknown

- AgeCategory_Adult

- AgeCategory_Child

- AgeCategory_Senior

- AgeCategory_Teen

- AgeCategory_YoungAdult
