# Model Card — Exp-002 | Stacking

## Identificación

- **Experimento:** Exp-002
- **Fecha:** 2026-03-20
- **Modelo:** Stacking
- **Tipo:** Clasificación binaria
- **Target:** Transported (True/False)
- **Script:** `scripts/03_train.py`

## Métricas de rendimiento

| Métrica | Valor |
|---|---|
| Accuracy (validación) | 0.8115 |
| ROC-AUC (validación) | 0.8991 |
| Accuracy (CV 5-fold) | 0.8105 |

## Hiperparámetros

```
min_samples_leaf: 10
max_leaf_nodes: 15
max_iter: 200
max_depth: 5
learning_rate: 0.05
l2_regularization: 1.0
```

## Dataset

- **Features:** 35
- **Muestras de entrenamiento:** 6,811
- **Estrategia de validación:** StratifiedKFold (5 folds) + hold-out 20%

## Notas

- Modelo en producción actual.
- Artefacto: `models/production/best_model.pkl`.
