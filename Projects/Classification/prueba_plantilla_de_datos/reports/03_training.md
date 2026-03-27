# Reporte de Entrenamiento — Spaceship Titanic


## Feature Set

- **Nombre:** fs-009_percentile_cabin


## Resultados Cross-Validation (todos los modelos)

Evaluacion con StratifiedKFold (5 folds). Ordenado por cv_accuracy_mean.

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8141 |            0.007  |            0.9047 |
| MoE_CatBoost         |             0.8115 |            0.0045 |            0.9016 |
| HistGradientBoosting |             0.8078 |            0.0065 |            0.8985 |
| LightGBM             |             0.8049 |            0.0041 |            0.8976 |
| GradientBoosting     |             0.8018 |            0.0057 |            0.8964 |
| RandomForest         |             0.8003 |            0.0042 |            0.8826 |
| XGBoost              |             0.7969 |            0.0027 |            0.8907 |
| ExtraTrees           |             0.7899 |            0.0052 |            0.8658 |
| LogisticRegression   |             0.7895 |            0.0098 |            0.8825 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |


## Mejor Modelo Seleccionado: CatBoost

El modelo con mayor cv_accuracy_mean fue tuneado con Optuna TPE (n_iter=25).


### Mejores hiperparametros encontrados

```
iterations: 600
depth: 9
learning_rate: 0.020589728197687916
l2_leaf_reg: 4.4546743769349115
bagging_temperature: 0.18340450985343382
```


## Evaluacion en Validacion: Tuneado vs Stacking vs MoE

Stacking construido con los 3 mejores modelos base: CatBoost, HistGradientBoosting, LightGBM. MoE entrena un experto CatBoost por segmento (cryo / activo).

| Modelo                    |   val_accuracy |   val_roc_auc |
|:--------------------------|---------------:|--------------:|
| CatBoost (tuneado)        |         0.8103 |        0.9036 |
| Stacking                  |         0.8015 |        0.8935 |
| MoE (CatBoost x segmento) |         0.8092 |        0.9033 |


### Classification Report — CatBoost (tuneado)

```
              precision    recall  f1-score   support

           0       0.80      0.82      0.81       845
           1       0.82      0.80      0.81       858

    accuracy                           0.81      1703
   macro avg       0.81      0.81      0.81      1703
weighted avg       0.81      0.81      0.81      1703

```


### Classification Report — Stacking

```
              precision    recall  f1-score   support

           0       0.78      0.83      0.81       845
           1       0.82      0.77      0.80       858

    accuracy                           0.80      1703
   macro avg       0.80      0.80      0.80      1703
weighted avg       0.80      0.80      0.80      1703

```


## Modelo Ganador Final

- **Modelo:** CatBoost (tuneado)

- **val_accuracy:** 0.8103

- **val_roc_auc:** 0.9036

El modelo ganador fue re-entrenado sobre el conjunto completo de train y guardado en `models/production/best_model.pkl`.


## Error Analysis — Tasa de error por segmento

Porcentaje de errores del modelo ganador en el conjunto de validacion, desglosado por variable.


### CryoSleep

| CryoSleep   |    n |   errors |   error_rate |
|:------------|-----:|---------:|-------------:|
| Active      | 1122 |      217 |       0.1934 |
| Cryo        |  564 |      103 |       0.1826 |
| Unknown     |   17 |        3 |       0.1765 |


### HomePlanet

| HomePlanet   |   n |   errors |   error_rate |
|:-------------|----:|---------:|-------------:|
| Unknown      |  14 |        6 |       0.4286 |
| Earth        | 931 |      246 |       0.2642 |
| Mars         | 334 |       38 |       0.1138 |
| Europa       | 424 |       33 |       0.0778 |


### Destination

| Destination   |    n |   errors |   error_rate |
|:--------------|-----:|---------:|-------------:|
| PSO J318.5-22 |  167 |       45 |       0.2695 |
| TRAPPIST-1e   | 1144 |      229 |       0.2002 |
| Unknown       |   34 |        5 |       0.1471 |
| 55 Cancri e   |  358 |       44 |       0.1229 |


### AgeCategory

| AgeCategory   |   n |   errors |   error_rate |
|:--------------|----:|---------:|-------------:|
| Child         | 164 |       44 |       0.2683 |
| YoungAdult    | 651 |      127 |       0.1951 |
| Teen          | 142 |       25 |       0.1761 |
| Adult         | 687 |      118 |       0.1718 |
| Senior        |  59 |        9 |       0.1525 |


### Deck

| Deck    |   n |   errors |   error_rate |
|:--------|----:|---------:|-------------:|
| G       | 523 |      147 |       0.2811 |
| E       | 167 |       31 |       0.1856 |
| F       | 541 |       97 |       0.1793 |
| D       | 105 |       17 |       0.1619 |
| A       |  45 |        7 |       0.1556 |
| Unknown |  22 |        3 |       0.1364 |
| C       | 154 |       12 |       0.0779 |
| B       | 146 |        9 |       0.0616 |


## Threshold Optimization

- **Umbral optimo:** 0.3965

- **val_accuracy con umbral optimo:** 0.8186

- **val_accuracy con umbral 0.50:** 0.8103

- **Ganancia:** 0.0083
