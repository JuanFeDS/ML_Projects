# Experimentation Log — Spaceship Titanic

Cada entrada es generada automáticamente por `scripts/03_train.py` al finalizar un run.
Todos los experimentos quedan registrados, promovidos o no.

---

## Exp-001 | 2026-03-20 | HistGradientBoosting (tuned) | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8115  _(ref: ninguna — primer experimento)_
- **val_roc_auc:** 0.899
- **cv_accuracy (ganador):** 0.8149
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-001_HistGradientBoosting_tuned.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** HistGradientBoosting (tuned)
- **nota:** Experimento manual desde `notebooks/exploratory/04.1_Advanced_Models.ipynb`

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-002 | 2026-03-20 | Stacking | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8115  _(ref: 0.8115, +0.0000)_
- **val_roc_auc:** 0.8991
- **cv_accuracy (ganador):** 0.8105
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-002_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `min_samples_leaf`: 10
  - `max_leaf_nodes`: 15
  - `max_iter`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `l2_regularization`: 1.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-003 | 2026-03-26 09:38 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8115  _(ref: 0.8115, +0.0000)_
- **val_roc_auc:** 0.8991
- **cv_accuracy (ganador):** 0.8105
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-003_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `min_samples_leaf`: 10
  - `max_leaf_nodes`: 15
  - `max_iter`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `l2_regularization`: 1.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-004 | 2026-03-26 09:41 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8115  _(ref: 0.8115, +0.0000)_
- **val_roc_auc:** 0.8991
- **cv_accuracy (ganador):** 0.8105
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-004_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `min_samples_leaf`: 10
  - `max_leaf_nodes`: 15
  - `max_iter`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `l2_regularization`: 1.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-005 | 2026-03-26 13:12 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8115  _(ref: 0.8115, +0.0000)_
- **val_roc_auc:** 0.8991
- **cv_accuracy (ganador):** 0.8105
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-005_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `min_samples_leaf`: 10
  - `max_leaf_nodes`: 15
  - `max_iter`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `l2_regularization`: 1.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-006 | 2026-03-26 13:35 | Stacking | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8227  _(ref: 0.8115, +0.0112)_
- **val_roc_auc:** 0.9052
- **cv_accuracy (ganador):** 0.8128
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-006_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline` _(pre-registro — asignado retroactivamente)_
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory.

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `learning_rate`: 0.1
  - `l2_leaf_reg`: 10
  - `iterations`: 200
  - `depth`: 6
  - `bagging_temperature`: 0.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-007 | 2026-03-26 14:15 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8156  _(ref: 0.8227, -0.0071)_
- **val_roc_auc:** 0.9029
- **cv_accuracy (ganador):** 0.8115
- **n_features:** 56
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-007_Stacking.pkl`

### Feature Set

- **nombre:** `fs-002_cryo_interactions` _(pre-registro — asignado retroactivamente)_
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + Route (HomePlanet+Destination), GroupCryoSleepRate, CryoSleepViolation, LuxurySpendingRatio, CabinNumber_DeckPercentile, GroupSpendingMean.
- **features anadidas vs parent (6):** `CabinNumber_DeckPercentile`, `CryoSleepViolation`, `GroupCryoSleepRate`, `GroupSpendingMean`, `LuxurySpendingRatio`, `Route`

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `learning_rate`: 0.1
  - `l2_leaf_reg`: 10
  - `iterations`: 200
  - `depth`: 6
  - `bagging_temperature`: 0.0

### Cross-Validation — todos los modelos

N/A (pre-registro del sistema de CV)

---

## Exp-008 | 2026-03-26 15:10 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8186  _(ref: 0.8227, -0.0041)_
- **val_roc_auc:** 0.9034
- **cv_accuracy (ganador):** 0.8127
- **n_features:** 38
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-008_CatBoost_tuneado.pkl`

### Feature Set

- **nombre:** `fs-003_solo_interactions`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), SpendingIntensity (TotalSpending/(SpendingCategories+1)). Features simples de alta senal, sin riesgo de multicolinealidad.
- **features anadidas vs parent (3):** `IsAlone`, `IsChild`, `SpendingIntensity`

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `learning_rate`: 0.1
  - `l2_leaf_reg`: 10
  - `iterations`: 200
  - `depth`: 6
  - `bagging_temperature`: 0.0

### Cross-Validation — todos los modelos

N/A (primer experimento con tabla de CV disponible parcialmente)

---

## Exp-009 | 2026-03-26 15:36 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8174  _(ref: 0.8227, -0.0053)_
- **val_roc_auc:** 0.9079
- **cv_accuracy (ganador):** 0.8109
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-009_CatBoost_tuneado.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `learning_rate`: 0.2
  - `l2_leaf_reg`: 10
  - `iterations`: 100
  - `depth`: 6
  - `bagging_temperature`: 1.0

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8109 |            0.0076 |            0.9045 |
| LightGBM             |             0.8084 |            0.009  |            0.8987 |
| GradientBoosting     |             0.8075 |            0.0069 |            0.8986 |
| HistGradientBoosting |             0.8059 |            0.0077 |            0.8986 |
| XGBoost              |             0.8038 |            0.0034 |            0.8936 |
| RandomForest         |             0.8031 |            0.0032 |            0.8838 |
| ExtraTrees           |             0.7905 |            0.0028 |            0.8677 |
| LogisticRegression   |             0.7899 |            0.0092 |            0.877  |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-010 | 2026-03-26 17:30 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8168  _(ref: 0.8227, -0.0059)_
- **val_roc_auc:** 0.9039
- **cv_accuracy (ganador):** 0.8137
- **n_features:** 42
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-010_CatBoost_tuneado.pkl`

### Feature Set

- **nombre:** `fs-005_structural_context`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).
- **features anadidas vs parent (8):** `CabinNeighborhoodDensity`, `FamilySizeFromName`, `GroupAgeDispersion`, `GroupCryoAlignment`, `GroupSpendingZScore`, `SpendingCategoryProfile`, `SpendingCategoryProfile_TE`, `SpendingEntropy`

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `learning_rate`: 0.05
  - `l2_leaf_reg`: 5
  - `iterations`: 400
  - `depth`: 6
  - `bagging_temperature`: 0.5

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8137 |            0.0078 |            0.9043 |
| HistGradientBoosting |             0.8105 |            0.0072 |            0.902  |
| LightGBM             |             0.8066 |            0.0062 |            0.9008 |
| GradientBoosting     |             0.8065 |            0.008  |            0.8988 |
| LogisticRegression   |             0.7991 |            0.0101 |            0.8841 |
| RandomForest         |             0.7983 |            0.008  |            0.8868 |
| XGBoost              |             0.798  |            0.0084 |            0.8925 |
| ExtraTrees           |             0.7937 |            0.0096 |            0.8747 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-011 | 2026-03-26 19:56 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8139  _(ref: 0.8227, -0.0088)_
- **val_roc_auc:** 0.9004
- **cv_accuracy (ganador):** 0.8115
- **n_features:** 56
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-011_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline`
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `learning_rate`: 0.1
  - `l2_leaf_reg`: 3
  - `iterations`: 400
  - `depth`: 6
  - `bagging_temperature`: 1.0

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8115 |            0.0087 |            0.9041 |
| MoE_CatBoost         |             0.8093 |            0.0063 |            0.9023 |
| HistGradientBoosting |             0.8063 |            0.0043 |            0.9002 |
| GradientBoosting     |             0.8025 |            0.0093 |            0.8982 |
| LightGBM             |             0.8025 |            0.0081 |            0.8985 |
| XGBoost              |             0.8014 |            0.0057 |            0.8915 |
| RandomForest         |             0.7984 |            0.0051 |            0.8861 |
| ExtraTrees           |             0.7959 |            0.0054 |            0.8737 |
| LogisticRegression   |             0.7896 |            0.0097 |            0.8816 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-012 | 2026-03-26 20:24 | Stacking | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8227  _(ref: 0.8227, +0.0000)_
- **val_roc_auc:** 0.9052
- **cv_accuracy (ganador):** 0.8128
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-012_Stacking.pkl`

### Feature Set

- **nombre:** `fs-001_baseline`
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).

### Modelo

- **algoritmo:** Stacking
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.05524411444827968
  - `l2_leaf_reg`: 8.778303618018892
  - `bagging_temperature`: 0.6339269511583692

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8128 |            0.0098 |            0.9058 |
| HistGradientBoosting |             0.8105 |            0.0085 |            0.901  |
| MoE_CatBoost         |             0.8081 |            0.0086 |            0.9026 |
| LightGBM             |             0.8063 |            0.0073 |            0.899  |
| GradientBoosting     |             0.8043 |            0.007  |            0.8983 |
| RandomForest         |             0.8031 |            0.0068 |            0.884  |
| XGBoost              |             0.8011 |            0.0061 |            0.8941 |
| LogisticRegression   |             0.789  |            0.0085 |            0.8821 |
| ExtraTrees           |             0.7865 |            0.0058 |            0.8653 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-013 | 2026-03-26 20:48 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8209  _(ref: 0.8227, -0.0018)_
- **val_roc_auc:** 0.9072
- **cv_accuracy (ganador):** 0.8128
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-013_CatBoost_(tuneado).pkl`

### Feature Set

- **nombre:** `fs-001_baseline`
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.05524411444827968
  - `l2_leaf_reg`: 8.778303618018892
  - `bagging_temperature`: 0.6339269511583692

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8128 |            0.0098 |            0.9058 |
| HistGradientBoosting |             0.8105 |            0.0085 |            0.901  |
| MoE_CatBoost         |             0.8081 |            0.0086 |            0.9026 |
| LightGBM             |             0.8063 |            0.0073 |            0.899  |
| GradientBoosting     |             0.8043 |            0.007  |            0.8983 |
| RandomForest         |             0.8031 |            0.0068 |            0.884  |
| XGBoost              |             0.8011 |            0.0061 |            0.8941 |
| LogisticRegression   |             0.789  |            0.0085 |            0.8821 |
| ExtraTrees           |             0.7865 |            0.0058 |            0.8653 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-014 | 2026-03-26 20:57 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8285  _(ref: 0.8227, +0.0058)_
- **val_roc_auc:** 0.9072
- **cv_accuracy (ganador):** 0.8128
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-014_CatBoost_(tuneado).pkl`

### Feature Set

- **nombre:** `fs-001_baseline`
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.05524411444827968
  - `l2_leaf_reg`: 8.778303618018892
  - `bagging_temperature`: 0.6339269511583692

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8128 |            0.0098 |            0.9058 |
| HistGradientBoosting |             0.8105 |            0.0085 |            0.901  |
| MoE_CatBoost         |             0.8081 |            0.0086 |            0.9026 |
| LightGBM             |             0.8063 |            0.0073 |            0.899  |
| GradientBoosting     |             0.8043 |            0.007  |            0.8983 |
| RandomForest         |             0.8031 |            0.0068 |            0.884  |
| XGBoost              |             0.8011 |            0.0061 |            0.8941 |
| LogisticRegression   |             0.789  |            0.0085 |            0.8821 |
| ExtraTrees           |             0.7865 |            0.0058 |            0.8653 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-015 | 2026-03-26 21:24 | MoE | ⚠️ INVALIDADO — data leakage

> **INVALIDADO.** `TravelGroup_TE` se calcula sobre el dataset completo antes del split,
> lo que incluye el propio label de cada muestra en su tasa de grupo. Para viajeros solos
> (GroupSize=1), `TravelGroup_TE` = su propio label → leakage puro. Score de Kaggle
> confirmó overfitting: 0.9495 val → 0.7989 test. El modelo en produccion sigue siendo
> **Exp-006** (Stacking, 0.8227). fs-008_domain_rules_only corrige esto.

### Metricas

- **val_accuracy:** 0.9495 _(INFLADO POR LEAKAGE — no valido)_
- **val_roc_auc:** 0.9928 _(INFLADO POR LEAKAGE — no valido)_
- **cv_accuracy (ganador):** 0.953 _(INFLADO POR LEAKAGE — no valido)_
- **n_features:** 36
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-015_MoE.pkl` _(no usar)_

### Feature Set

- **nombre:** `fs-007_domain_rules`
- **parent:** `fs-001_baseline`
- **descripcion:** Imputacion por 6 reglas fisicas del dataset + TravelGroup_TE. Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. TravelGroup_TE: tasa de transporte media del grupo de viaje (target encoding).
- **features anadidas vs parent (2):** `TravelGroup`, `TravelGroup_TE`

### Modelo

- **algoritmo:** MoE
- **hiperparametros optimos:**
  - `iterations`: 200
  - `depth`: 5
  - `learning_rate`: 0.05011896346846189
  - `l2_leaf_reg`: 13.20487906665041
  - `bagging_temperature`: 0.6246118922833275

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.953  |            0.0047 |            0.9932 |
| GradientBoosting     |             0.9521 |            0.0024 |            0.9931 |
| RandomForest         |             0.9502 |            0.006  |            0.9902 |
| MoE_CatBoost         |             0.9501 |            0.0033 |            0.9927 |
| LogisticRegression   |             0.9493 |            0.004  |            0.9922 |
| XGBoost              |             0.9464 |            0.0037 |            0.9913 |
| HistGradientBoosting |             0.9455 |            0.0047 |            0.992  |
| LightGBM             |             0.9449 |            0.0053 |            0.992  |
| ExtraTrees           |             0.9432 |            0.0052 |            0.9881 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-016 | 2026-03-26 21:35 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8197  _(ref: 0.9495, -0.1298)_
- **val_roc_auc:** 0.9051
- **cv_accuracy (ganador):** 0.814
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-016_CatBoost_(tuneado).pkl`

### Feature Set

- **nombre:** `fs-008_domain_rules_only`
- **parent:** `fs-001_baseline`
- **descripcion:** Imputacion por 6 reglas fisicas del dataset SIN target encoding de grupo. Elimina el leakage de TravelGroup_TE (fs-007). Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. Mismo pipeline que fs-007 pero con el mismo espacio de features que fs-001.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 7
  - `learning_rate`: 0.020589575682537137
  - `l2_leaf_reg`: 5.381997322907315
  - `bagging_temperature`: 0.24927702441464045

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.814  |            0.006  |            0.906  |
| HistGradientBoosting |             0.8106 |            0.0037 |            0.9012 |
| MoE_CatBoost         |             0.8097 |            0.0049 |            0.9036 |
| LightGBM             |             0.8065 |            0.0051 |            0.8984 |
| GradientBoosting     |             0.8041 |            0.0064 |            0.8989 |
| RandomForest         |             0.7993 |            0.0077 |            0.8843 |
| XGBoost              |             0.7949 |            0.0058 |            0.8924 |
| LogisticRegression   |             0.7893 |            0.0086 |            0.8824 |
| ExtraTrees           |             0.7889 |            0.0077 |            0.8658 |
| Baseline             |             0.5036 |            0.0001 |            0.5    |

---

## Exp-017 | 2026-03-26 22:09 | CatBoost (tuneado) | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8186  _(ref: 0.8227, -0.0041)_
- **val_roc_auc:** 0.9036
- **cv_accuracy (ganador):** 0.8141
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-017_CatBoost_(tuneado).pkl`

### Feature Set

- **nombre:** `fs-009_percentile_cabin`
- **parent:** `fs-008_domain_rules_only`
- **descripcion:** fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor distributional shift entre train y test. La percentil normaliza la posicion relativa dentro del deck, eliminando el shift de rango absoluto.
- **features anadidas vs parent (1):** `CabinNumber_DeckPercentile`
- **features eliminadas vs parent (1):** `CabinNumber`

### Modelo

- **algoritmo:** CatBoost (tuneado)
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 9
  - `learning_rate`: 0.020589728197687916
  - `l2_leaf_reg`: 4.4546743769349115
  - `bagging_temperature`: 0.18340450985343382

### Cross-Validation — todos los modelos

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

---
