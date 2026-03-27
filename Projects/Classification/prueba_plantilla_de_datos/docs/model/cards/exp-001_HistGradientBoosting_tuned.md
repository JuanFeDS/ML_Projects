# Model Card — Exp-001 | HistGradientBoosting (tuned)

## Identificación

- **Experimento:** Exp-001
- **Fecha:** 2026-03-20
- **Modelo:** HistGradientBoosting (tuned)
- **Tipo:** Clasificación binaria
- **Target:** Transported (True/False)
- **Notebook:** `notebooks/exploratory/04.1_Advanced_Models.ipynb`

## Métricas de rendimiento

| Métrica | Valor |
|---|---|
| Accuracy (validación) | 0.8115 |
| ROC-AUC (validación) | 0.899 |
| Accuracy (CV 5-fold) | 0.8149 |

## Dataset

- **Features:** 35
- **Muestras de entrenamiento:** 6,811
- **Estrategia de validación:** StratifiedKFold (5 folds) + hold-out 20%

## Notas

- Experimento manual desde notebook, no desde `scripts/03_train.py`.
- Objetivo de 0.83 accuracy no alcanzado.
- Promovido a `models/experiments/best_advanced_model.pkl`.
