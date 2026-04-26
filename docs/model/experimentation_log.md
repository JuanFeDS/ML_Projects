# Experimentation Log — Spaceship Titanic


## Exp-001 | 2026-04-09 21:36 | LogisticRegression | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.7886
- **val_roc_auc:** 0.8692
- **cv_accuracy (ganador):** 0.789
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-001_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-001_baseline`
- **parent:** ninguno (primer set)
- **descripcion:** Features base: Cabin→Deck/Side/CabinNumber, PassengerId→GroupSize, spending log+categorias, AgeCategory. Referencia: Exp-001 a Exp-006 (mejor val_accuracy=0.8227).

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |              0.789 |            0.0085 |            0.8821 |

---

## Exp-002 | 2026-04-09 22:54 | LogisticRegression | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.7868  _(ref: 0.7886, -0.0018)_
- **val_roc_auc:** 0.8694
- **cv_accuracy (ganador):** 0.7896
- **n_features:** 56
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-002_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-002_cryo_interactions`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + Route (HomePlanet+Destination), GroupCryoSleepRate, CryoSleepViolation, LuxurySpendingRatio, CabinNumber_DeckPercentile, GroupSpendingMean. Referencia: Exp-007 (val_accuracy=0.8156, no supero fs-001).
- **features anadidas vs parent (6):** `CabinNumber_DeckPercentile`, `CryoSleepViolation`, `GroupCryoSleepRate`, `GroupSpendingMean`, `LuxurySpendingRatio`, `Route`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7896 |            0.0097 |            0.8816 |

---

## Exp-003 | 2026-04-09 22:57 | LogisticRegression | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.7892  _(ref: 0.7886, +0.0006)_
- **val_roc_auc:** 0.8702
- **cv_accuracy (ganador):** 0.7896
- **n_features:** 38
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-003_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-003_solo_interactions`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + IsAlone (GroupSize==1), IsChild (Age<13), SpendingIntensity (TotalSpending/(SpendingCategories+1)). Features simples de alta senal, sin riesgo de multicolinealidad.
- **features anadidas vs parent (3):** `IsAlone`, `IsChild`, `SpendingIntensity`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7896 |            0.0091 |            0.8821 |

---

## Exp-004 | 2026-04-09 22:59 | LogisticRegression | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.7904  _(ref: 0.7892, +0.0012)_
- **val_roc_auc:** 0.8687
- **cv_accuracy (ganador):** 0.7899
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-004_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7899 |            0.0092 |             0.877 |

---

## Exp-005 | 2026-04-09 23:00 | LogisticRegression | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.7886  _(ref: 0.7904, -0.0018)_
- **val_roc_auc:** 0.8701
- **cv_accuracy (ganador):** 0.7991
- **n_features:** 42
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-005_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-005_structural_context`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).
- **features anadidas vs parent (8):** `CabinNeighborhoodDensity`, `FamilySizeFromName`, `GroupAgeDispersion`, `GroupCryoAlignment`, `GroupSpendingZScore`, `SpendingCategoryProfile`, `SpendingCategoryProfile_TE`, `SpendingEntropy`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7991 |            0.0101 |            0.8841 |

---

## Exp-006 | 2026-04-09 23:01 | LogisticRegression | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.7874  _(ref: 0.7904, -0.0030)_
- **val_roc_auc:** 0.8685
- **cv_accuracy (ganador):** 0.7893
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-006_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-006_group_imputation`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con imputacion group-aware para columnas de gasto. Pasajeros no-cryo con spending NaN reciben la mediana del TravelGroup (en lugar de 0), haciendo que TotalSpending_Log capture mejor su perfil real. El orden del pipeline cambia: fill categoricals → impute → create_spending_features.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7893 |            0.0089 |            0.8823 |

---

## Exp-007 | 2026-04-09 23:02 | LogisticRegression | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.9454  _(ref: 0.7904, +0.1550)_
- **val_roc_auc:** 0.9912
- **cv_accuracy (ganador):** 0.9493
- **n_features:** 36
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-007_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-007_domain_rules`
- **parent:** `fs-001_baseline`
- **descripcion:** Imputacion por 6 reglas fisicas del dataset + TravelGroup_TE. Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. TravelGroup_TE: tasa de transporte media del grupo de viaje (target encoding).
- **features anadidas vs parent (2):** `TravelGroup`, `TravelGroup_TE`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.9493 |             0.004 |            0.9922 |

---

## Exp-008 | 2026-04-09 23:03 | LogisticRegression | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.7898  _(ref: 0.9454, -0.1556)_
- **val_roc_auc:** 0.8703
- **cv_accuracy (ganador):** 0.7893
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-008_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-008_domain_rules_only`
- **parent:** `fs-001_baseline`
- **descripcion:** Imputacion por 6 reglas fisicas del dataset SIN target encoding de grupo. Elimina el leakage de TravelGroup_TE (fs-007). Reglas: HomePlanet por grupo, Deck A/B/C→Europa / G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0. Mismo pipeline que fs-007 pero con el mismo espacio de features que fs-001.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7893 |            0.0086 |            0.8824 |

---

## Exp-009 | 2026-04-09 23:04 | LogisticRegression | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.7886  _(ref: 0.9454, -0.1568)_
- **val_roc_auc:** 0.8705
- **cv_accuracy (ganador):** 0.7895
- **n_features:** 35
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-009_LogisticRegression.pkl`

### Feature Set

- **nombre:** `fs-009_percentile_cabin`
- **parent:** `fs-008_domain_rules_only`
- **descripcion:** fs-008 (domain rules) + CabinNumber reemplazado por CabinNumber_DeckPercentile. Motivacion: adversarial validation AUC=0.79, CabinNumber es la feature con mayor distributional shift entre train y test. La percentil normaliza la posicion relativa dentro del deck, eliminando el shift de rango absoluto.
- **features anadidas vs parent (1):** `CabinNumber_DeckPercentile`
- **features eliminadas vs parent (1):** `CabinNumber`

### Modelo

- **algoritmo:** LogisticRegression

### Cross-Validation — todos los modelos

| Modelo             |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------------|-------------------:|------------------:|------------------:|
| LogisticRegression |             0.7895 |            0.0098 |            0.8825 |

---

## Exp-010 | 2026-04-09 23:07 | RandomForest | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.805  _(ref: 0.9454, -0.1404)_
- **val_roc_auc:** 0.8871
- **cv_accuracy (ganador):** 0.8031
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-010_RandomForest.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** RandomForest
- **hiperparametros optimos:**
  - `n_estimators`: 400
  - `max_depth`: None
  - `min_samples_split`: 8
  - `max_features`: log2

### Cross-Validation — todos los modelos

| Modelo       |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:-------------|-------------------:|------------------:|------------------:|
| RandomForest |             0.8031 |            0.0032 |            0.8838 |

---

## Exp-011 | 2026-04-09 23:40 | LightGBM | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8203  _(ref: 0.805, +0.0153)_
- **val_roc_auc:** 0.9046
- **cv_accuracy (ganador):** 0.8084
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-011_LightGBM.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** LightGBM
- **hiperparametros optimos:**
  - `n_estimators`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.09371983213998145
  - `num_leaves`: 53
  - `subsample`: 0.6216986075761003
  - `colsample_bytree`: 0.6535783131989047
  - `reg_alpha`: 1.4649640711267433

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| LightGBM |             0.8084 |             0.009 |            0.8987 |

---

## Exp-012 | 2026-04-09 23:42 | XGBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8209  _(ref: 0.8203, +0.0006)_
- **val_roc_auc:** 0.9023
- **cv_accuracy (ganador):** 0.8038
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-012_XGBoost.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** XGBoost
- **hiperparametros optimos:**
  - `n_estimators`: 400
  - `max_depth`: 8
  - `learning_rate`: 0.04046301461698397
  - `subsample`: 0.7884745869448389
  - `colsample_bytree`: 0.6548503898583582
  - `reg_alpha`: 1.4070712978947977
  - `reg_lambda`: 3.4994103768847347

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| XGBoost  |             0.8038 |            0.0034 |            0.8936 |

---

## Exp-013 | 2026-04-09 23:50 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8221  _(ref: 0.8209, +0.0012)_
- **val_roc_auc:** 0.9069
- **cv_accuracy (ganador):** 0.8109
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-013_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 7
  - `learning_rate`: 0.04810373052442276
  - `l2_leaf_reg`: 18.989213385799708
  - `bagging_temperature`: 0.0065675086678651014

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8109 |            0.0076 |            0.9045 |

---

## Exp-014 | 2026-04-09 23:57 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8233  _(ref: 0.8221, +0.0012)_
- **val_roc_auc:** 0.9022
- **cv_accuracy (ganador):** 0.8137
- **n_features:** 42
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-014_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-005_structural_context`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 + 7 features estructurales/contextuales: SpendingEntropy (Shannon), GroupSpendingZScore (desviacion intragrupal), CabinNeighborhoodDensity (densidad ±50 cabinas por Deck), FamilySizeFromName (apellido compartido), GroupCryoAlignment (consenso CryoSleep en el grupo), GroupAgeDispersion (std Age por grupo), SpendingCategoryProfile → TE (patron de servicios usados).
- **features anadidas vs parent (8):** `CabinNeighborhoodDensity`, `FamilySizeFromName`, `GroupAgeDispersion`, `GroupCryoAlignment`, `GroupSpendingZScore`, `SpendingCategoryProfile`, `SpendingCategoryProfile_TE`, `SpendingEntropy`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 6
  - `learning_rate`: 0.04569223593850866
  - `l2_leaf_reg`: 17.920260567511946
  - `bagging_temperature`: 0.6047914680041317

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8137 |            0.0078 |            0.9043 |

---

## Exp-015 | 2026-04-10 00:14 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.815  _(ref: 0.8233, -0.0083)_
- **val_roc_auc:** 0.9045
- **cv_accuracy (ganador):** 0.8102
- **n_features:** 28
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-015_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-010_cryo_spending`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + 4 features de interaccion CryoSleep x spending: CryoSpendingAnomaly (gasto cuando CryoSleep=True, anomalia fisica), GroupTransportedProxy (ratio de miembros del grupo sin gasto), SideSpendingDiff (asimetria de gasto entre lados P/S de la cabina), CryoSleepBinary (CryoSleep como numerico 1/0/-1). Motivacion: los errores mas frecuentes estan en pasajeros con patrones de gasto inconsistentes con su estado CryoSleep.
- **features anadidas vs parent (4):** `CryoSleepBinary`, `CryoSpendingAnomaly`, `GroupTransportedProxy`, `SideSpendingDiff`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 9
  - `learning_rate`: 0.020589728197687916
  - `l2_leaf_reg`: 4.4546743769349115
  - `bagging_temperature`: 0.18340450985343382

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8102 |            0.0098 |            0.9032 |

---

## Exp-016 | 2026-04-10 22:09 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.818  _(ref: 0.8233, -0.0053)_
- **val_roc_auc:** 0.9021
- **cv_accuracy (ganador):** 0.814
- **n_features:** 43
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-016_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-011_child_route`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + 4 features dirigidas a los segmentos con mayor error en exp-013: IsChild (binario Age<13), GroupHasChild (grupo tiene algun nino), GroupChildRate (proporcion de ninos en el grupo), Route (HomePlanet+Destination, 9 combinaciones categoricas). Motivacion: ninos (28% error) y PSO J318.5-22 (30% error) son los segmentos mas dificiles; el contexto familiar y la ruta completa aportan informacion que el modelo no capturaba de forma individual.
- **features anadidas vs parent (4):** `GroupChildRate`, `GroupHasChild`, `IsChild`, `Route`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 7
  - `learning_rate`: 0.0748149645868013
  - `l2_leaf_reg`: 9.48753158546387
  - `bagging_temperature`: 0.0035936085696097256

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |              0.814 |            0.0077 |            0.9041 |

---

## Exp-017 | 2026-04-10 22:17 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.825  _(ref: 0.8233, +0.0017)_
- **val_roc_auc:** 0.9061
- **cv_accuracy (ganador):** 0.814
- **n_features:** 28
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-017_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-012_child_route_te`
- **parent:** `fs-011_child_route`
- **descripcion:** fs-011 con Route como Target Encoding en lugar de OHE. Motivacion: PSO J318.5-22 tiene 31% error y su tasa de transporte difiere mucho segun HomePlanet de origen. Route_TE codifica la tasa media de transporte por ruta (una columna numerica vs 9 OHE), capturando la señal ordinal que OHE no puede expresar. Features: IsChild, GroupHasChild, GroupChildRate (contexto familiar) + Deck_TE, HomePlanet_TE, Route_TE (3 target encodings).
- **features anadidas vs parent (1):** `Route_TE`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 400
  - `depth`: 7
  - `learning_rate`: 0.05567776420011066
  - `l2_leaf_reg`: 12.048063218106869
  - `bagging_temperature`: 0.7532380708632854

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |              0.814 |            0.0048 |            0.9042 |

---

## Exp-018 | 2026-04-10 22:39 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8238  _(ref: 0.825, -0.0012)_
- **val_roc_auc:** 0.906
- **cv_accuracy (ganador):** 0.8109
- **n_features:** 24
- **n_train_samples:** 6,811
- **artefacto:** `models/experiments/exp-018_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 400
  - `depth`: 6
  - `learning_rate`: 0.10744635528500016
  - `l2_leaf_reg`: 10.098119401157994
  - `bagging_temperature`: 0.10372772654542978

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8109 |            0.0076 |            0.9045 |

---

## Exp-019 | 2026-04-10 23:43 | TabNet | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.805  _(ref: 0.825, -0.0200)_
- **val_roc_auc:** 0.892523551399291
- **cv_accuracy (ganador):** 0.8115091015854374
- **n_features:** 24
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-019_TabNet.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** TabNet
- **hiperparametros optimos:**
  - `n_d`: 16
  - `n_steps`: 3
  - `gamma`: 1.3650761945240073
  - `n_independent`: 3
  - `n_shared`: 3
  - `momentum`: 0.011056215960993753
  - `batch_size`: 512
  - `learning_rate`: 0.0706918559511276

---

## Exp-020 | 2026-04-10 23:43 | TabNet | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.805  _(ref: 0.825, -0.0200)_
- **val_roc_auc:** 0.892523551399291
- **cv_accuracy (ganador):** 0.8115091015854374
- **n_features:** 24
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-020_TabNet.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding (media del target por categoria, con suavizado). Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan informacion ordinal que OHE no captura.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** TabNet
- **hiperparametros optimos:**
  - `n_d`: 16
  - `n_steps`: 3
  - `gamma`: 1.3650761945240073
  - `n_independent`: 3
  - `n_shared`: 3
  - `momentum`: 0.011056215960993753
  - `batch_size`: 512
  - `learning_rate`: 0.0706918559511276

---

## Exp-021 | 2026-04-11 08:38 | MoE | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8203  _(ref: 0.825, -0.0047)_
- **val_roc_auc:** 0.9061
- **cv_accuracy (ganador):** 0.8112
- **n_features:** 28
- **n_train_samples:** 6,876
- **artefacto:** `models/experiments/exp-021_MoE.pkl`

### Feature Set

- **nombre:** `fs-013_group_context`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + imputacion Age por grupo (mediana del TravelGroup antes de global) + 4 features de contexto colectivo inspiradas en soluciones top Kaggle: GroupAllCryo (todos en CryoSleep: 80.5% vs 42.4% transported), GroupAnyCryo (alguno en CryoSleep: grupos mixtos 60.4%), SpendShare (gasto individual / gasto total grupo, corr=-0.15 entre no-cryo), GroupSpendOthers_Log (gasto del resto del grupo, corr=+0.09).
- **features anadidas vs parent (4):** `GroupAllCryo`, `GroupAnyCryo`, `GroupSpendOthers_Log`, `SpendShare`

### Modelo

- **algoritmo:** MoE
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 6
  - `learning_rate`: 0.03208899494279826
  - `l2_leaf_reg`: 19.802111938430365
  - `bagging_temperature`: 0.33293772919177633

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8112 |             0.008 |            0.9041 |

---

## Exp-022 | 2026-04-12 18:45 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8145  _(ref: 0.825, -0.0105)_
- **val_roc_auc:** 0.8996
- **cv_accuracy (ganador):** 0.8112
- **n_features:** 34
- **n_train_samples:** 6,876
- **artefacto:** `models/experiments/exp-022_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-014_spend_clusters`
- **parent:** `fs-013_group_context`
- **descripcion:** fs-013 + 6 features derivadas del EDA 2026-04-12: EntertainmentSpend_Log (FoodCourt+VRDeck+Spa, r inter=0.42-0.46), ComfortSpend_Log (RoomService+ShoppingMall, r=0.36), EntVsComfort_Ratio, IsExtremeSpender (any service > p99 training), AgeVsPlanetMedian (Earth=23/Europa=33/Mars=28), GroupCryoSegment ordinal 0-3 (NoCryo 33.9% / Solo 45.2% / AnyCryo 60.4% / AllCryo 92.2%).
- **features anadidas vs parent (6):** `AgeVsPlanetMedian`, `ComfortSpend_Log`, `EntVsComfort_Ratio`, `EntertainmentSpend_Log`, `GroupCryoSegment`, `IsExtremeSpender`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.030265805407797112
  - `l2_leaf_reg`: 18.741644709165087
  - `bagging_temperature`: 0.5680093335492117

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8112 |            0.0078 |            0.9038 |

---

## Exp-023 | 2026-04-12 20:00 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8233  _(ref: 0.825, -0.0017)_
- **val_roc_auc:** 0.9044
- **cv_accuracy (ganador):** 0.8067
- **n_features:** 24
- **n_train_samples:** 6,876
- **artefacto:** `models/experiments/exp-023_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-015_domain_imputation`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 con imputación agresiva por reglas de dominio antes de engineering. apply_domain_rules resuelve por inferencia: HomePlanet NaN por grupo (100% homogéneo → 0 ambigüedad), HomePlanet NaN por Deck (A/B/C→Europa, G→Earth), Deck/Side NaN por grupo, CryoSleep NaN con gasto>0 → False, spending NaN con CryoSleep=True → 0. Misma dimensionalidad que fs-004 — diferencia es calidad de imputación.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 400
  - `depth`: 8
  - `learning_rate`: 0.04893857897388962
  - `l2_leaf_reg`: 18.10519218594773
  - `bagging_temperature`: 0.3208501786198171

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8067 |            0.0034 |            0.9037 |

---

## Exp-024 | 2026-04-12 21:10 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8233  _(ref: 0.825, -0.0017)_
- **val_roc_auc:** 0.9044
- **cv_accuracy (ganador):** 0.8067
- **n_features:** 24
- **n_train_samples:** 6,876
- **artefacto:** `models/experiments/exp-024_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-016_transductive`
- **parent:** `fs-015_domain_imputation`
- **descripcion:** fs-015 con imputación transductiva: train.csv + test.csv se combinan antes de aplicar apply_domain_rules e impute_age_by_group, de modo que grupos con miembros en ambos splits comparten información. Requiere ejecutar scripts/00_transductive_impute.py primero y pasar --train-path data/processed/train_transductive.csv a 02_features.py. Misma dimensionalidad que fs-004/fs-015.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 400
  - `depth`: 8
  - `learning_rate`: 0.04893857897388962
  - `l2_leaf_reg`: 18.10519218594773
  - `bagging_temperature`: 0.3208501786198171

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| CatBoost |             0.8067 |            0.0034 |            0.9037 |

---

## Exp-025 | 2026-04-13 21:31 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8182  _(ref: 0.825, -0.0068)_
- **val_roc_auc:** 0.9076
- **cv_accuracy (ganador):** 0.8154
- **n_features:** 24
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-025_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-004_target_encoding`
- **parent:** `fs-001_baseline`
- **descripcion:** fs-001 con Deck y HomePlanet reemplazados por Target Encoding. Deck (8 cats → 1 num) y HomePlanet (3 cats → 1 num) reducen dimensionalidad y aportan información ordinal que OHE no captura.
- **features anadidas vs parent (2):** `Deck_TE`, `HomePlanet_TE`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.13570390610987357
  - `l2_leaf_reg`: 19.825773038488364
  - `bagging_temperature`: 0.9960169893612135

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8154 |            0.0034 |            0.9075 |
| HistGradientBoosting |             0.8141 |            0.0072 |            0.9017 |
| LightGBM             |             0.8078 |            0.007  |            0.9028 |
| GradientBoosting     |             0.8075 |            0.009  |            0.8988 |
| XGBoost              |             0.8028 |            0.0057 |            0.8965 |
| RandomForest         |             0.8003 |            0.0071 |            0.8862 |
| ExtraTrees           |             0.7977 |            0.0076 |            0.8714 |
| LogisticRegression   |             0.7914 |            0.0101 |            0.8758 |
| Baseline             |             0.5036 |            0.0002 |            0.5    |

---

## Exp-026 | 2026-04-13 21:57 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8575  _(ref: 0.825, +0.0325)_
- **val_roc_auc:** 0.9379
- **cv_accuracy (ganador):** 0.8547
- **n_features:** 25
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-026_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-017_lastname_te`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + LastName con Target Encoding suavizado (k=30). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. TE suavizado bayesiano: TE=(n*mean+30*global_mean)/(n+30) previene leakage en apellidos raros (n=1-2 → TE≈global_mean≈0.50).
- **features anadidas vs parent (2):** `LastName`, `LastName_TE`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 6
  - `learning_rate`: 0.06984284982015783
  - `l2_leaf_reg`: 17.09292634438464
  - `bagging_temperature`: 0.007697741191395324

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8547 |            0.0052 |            0.9378 |
| HistGradientBoosting |             0.8507 |            0.006  |            0.9346 |
| LightGBM             |             0.849  |            0.0068 |            0.9336 |
| GradientBoosting     |             0.8434 |            0.0069 |            0.9276 |
| XGBoost              |             0.8417 |            0.0072 |            0.9293 |
| ExtraTrees           |             0.8377 |            0.005  |            0.914  |
| RandomForest         |             0.8373 |            0.0068 |            0.9199 |
| LogisticRegression   |             0.8318 |            0.0072 |            0.9129 |
| Baseline             |             0.5036 |            0.0002 |            0.5    |

---

## Exp-027 | 2026-04-13 22:29 | CatBoost | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8176  _(ref: 0.8575, -0.0399)_
- **val_roc_auc:** 0.9066
- **cv_accuracy (ganador):** 0.8149
- **n_features:** 25
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-027_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-017_lastname_te`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + LastName con Target Encoding fold-aware (sklearn TargetEncoder, cv=5). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. El encoding se computa por fold dentro del CV para eliminar leakage en apellidos raros (~80-90% de test surnames son unseen en train). Pipeline interno aplica TargetEncoder(smooth='auto') a LastName y passthrough al resto.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 6
  - `learning_rate`: 0.021652690184947157
  - `l2_leaf_reg`: 16.121101290648085
  - `bagging_temperature`: 0.003097017597283175

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8149 |            0.0065 |            0.9078 |
| LightGBM             |             0.8127 |            0.0059 |            0.9024 |
| HistGradientBoosting |             0.8125 |            0.0054 |            0.9029 |
| XGBoost              |             0.8063 |            0.0063 |            0.8982 |
| GradientBoosting     |             0.8057 |            0.0093 |            0.8988 |
| RandomForest         |             0.8012 |            0.0044 |            0.8904 |
| ExtraTrees           |             0.7984 |            0.0054 |            0.8815 |
| LogisticRegression   |             0.7905 |            0.0107 |            0.8759 |
| Baseline             |             0.5036 |            0.0002 |            0.5    |

---

## Exp-028 | 2026-04-13 22:41 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8181  _(ref: 0.8176, +0.0005)_
- **val_roc_auc:** 0.9073
- **cv_accuracy (ganador):** 0.8158
- **n_features:** 28
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-028_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-018_group_consistency`
- **parent:** `fs-017_lastname_te`
- **descripcion:** fs-017 + 3 features de coherencia interna de grupo (sin usar target). GroupAllSameDest: 1 si todos en el grupo comparten Destination. GroupAllSameHomePlanet: 1 si todos comparten HomePlanet. GroupConsistencyScore: suma ordinal 0-2 de cohesión grupal. Complementa LastName fold-aware TE con señal estructural del grupo.
- **features anadidas vs parent (3):** `GroupAllSameDest`, `GroupAllSameHomePlanet`, `GroupConsistencyScore`

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 600
  - `depth`: 5
  - `learning_rate`: 0.08242047873638254
  - `l2_leaf_reg`: 16.675777579204556
  - `bagging_temperature`: 0.25858137370944156

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8158 |            0.0076 |            0.9066 |
| LightGBM             |             0.8124 |            0.0076 |            0.9033 |
| HistGradientBoosting |             0.8117 |            0.0078 |            0.9024 |
| XGBoost              |             0.805  |            0.0066 |            0.8963 |
| GradientBoosting     |             0.805  |            0.0092 |            0.8986 |
| RandomForest         |             0.8    |            0.0051 |            0.8887 |
| ExtraTrees           |             0.7938 |            0.0055 |            0.8737 |
| LogisticRegression   |             0.7914 |            0.0097 |            0.8759 |
| Baseline             |             0.5036 |            0.0002 |            0.5    |

---

## Exp-029 | 2026-04-13 23:01 | LightGBM | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8135  _(ref: 0.8181, -0.0046)_
- **val_roc_auc:** 0.9024
- **cv_accuracy (ganador):** 0.8127
- **n_features:** 25
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-029_LightGBM.pkl`

### Feature Set

- **nombre:** `fs-017_lastname_te`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + LastName con Target Encoding fold-aware (sklearn TargetEncoder, cv=5). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. El encoding se computa por fold dentro del CV para eliminar leakage en apellidos raros (~80-90% de test surnames son unseen en train). Pipeline interno aplica TargetEncoder(smooth='auto') a LastName y passthrough al resto.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** LightGBM

### Cross-Validation — todos los modelos

| Modelo   |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------|-------------------:|------------------:|------------------:|
| LightGBM |             0.8127 |            0.0059 |            0.9024 |

---

## Exp-030 | 2026-04-13 23:01 | HistGradientBoosting | ❌ No supero al modelo actual

### Metricas

- **val_accuracy:** 0.8127  _(ref: 0.8181, -0.0054)_
- **val_roc_auc:** 0.9029
- **cv_accuracy (ganador):** 0.8125
- **n_features:** 25
- **n_train_samples:** 8,514
- **artefacto:** `models/experiments/exp-030_HistGradientBoosting.pkl`

### Feature Set

- **nombre:** `fs-017_lastname_te`
- **parent:** `fs-004_target_encoding`
- **descripcion:** fs-004 + LastName con Target Encoding fold-aware (sklearn TargetEncoder, cv=5). LastName es proxy de familia: comparten HomePlanet, destino y comportamiento. El encoding se computa por fold dentro del CV para eliminar leakage en apellidos raros (~80-90% de test surnames son unseen en train). Pipeline interno aplica TargetEncoder(smooth='auto') a LastName y passthrough al resto.
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** HistGradientBoosting

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| HistGradientBoosting |             0.8125 |            0.0054 |            0.9029 |

---

## Exp-031 | 2026-04-16 21:48 | CatBoost | 🏆 Promovido a produccion

### Metricas

- **val_accuracy:** 0.8352  _(ref: 0.8176, +0.0176)_
- **val_roc_auc:** 0.9246
- **cv_accuracy (ganador):** 0.8347
- **n_features:** 25
- **n_train_samples:** 9,472
- **artefacto:** `models/experiments/exp-031_CatBoost.pkl`

### Feature Set

- **nombre:** `fs-019_pseudo_labeled`
- **parent:** `fs-017_lastname_te`
- **descripcion:** fs-017 con pseudo-labeling: entrenado sobre train.csv + 985 filas de test.csv donde exp-027 predice con confianza >= 0.95 (711 True, 274 False, confianza media 97.6%). Pipeline idéntico a fs-017; diferencia es el tamaño del dataset de entrenamiento (8,693 → 9,678 filas, +11.3%).
- **cambios vs parent:** solo se modifico el tipo de encoding

### Modelo

- **algoritmo:** CatBoost
- **hiperparametros optimos:**
  - `iterations`: 400
  - `depth`: 4
  - `learning_rate`: 0.11971677241936392
  - `l2_leaf_reg`: 13.688056542097758
  - `bagging_temperature`: 0.9799558910910812

### Cross-Validation — todos los modelos

| Modelo               |   cv_accuracy_mean |   cv_accuracy_std |   cv_roc_auc_mean |
|:---------------------|-------------------:|------------------:|------------------:|
| CatBoost             |             0.8347 |            0.0071 |            0.9246 |
| HistGradientBoosting |             0.8299 |            0.0088 |            0.9207 |
| LightGBM             |             0.8282 |            0.0069 |            0.9204 |
| GradientBoosting     |             0.8239 |            0.01   |            0.9171 |
| XGBoost              |             0.823  |            0.0111 |            0.9152 |
| RandomForest         |             0.8224 |            0.0099 |            0.9105 |
| ExtraTrees           |             0.8173 |            0.003  |            0.9026 |
| LogisticRegression   |             0.8117 |            0.0095 |            0.8983 |
| Baseline             |             0.5257 |            0.0002 |            0.5    |

---
