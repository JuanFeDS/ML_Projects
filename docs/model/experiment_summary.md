# Resumen de Experimentos — Spaceship Titanic
_Última actualización: 2026-04-12_

## Contexto del problema

Clasificación binaria: predecir si un pasajero fue transportado a una dimensión alternativa.
Dataset: 8,693 train / 4,277 test. Target balanceado (50.4% / 49.6%).
Métrica: accuracy. Top del leaderboard público: ~0.83–0.84.

---

## Tabla de experimentos con score Kaggle

| Exp | Modelo | Feature Set | val_acc | Kaggle | Δ vs mejor |
|-----|--------|-------------|---------|--------|------------|
| 013 | CatBoost | fs-004 target encoding | 0.8221 | **0.80687** | — |
| 017 | CatBoost | fs-012 child + route TE | 0.8250 | 0.80617 | -0.00070 |
| 025 | CatBoost | fs-016 transductivo | 0.8233 | 0.80547 | -0.00140 |
| 023 | CatBoost | fs-015 domain rules | 0.8233 | 0.80547 | -0.00140 |
| ensemble | LGB+XGB+CB | fs-004 | — | 0.80219 | -0.00468 |
| 024 | LightGBM | fs-004 | 0.8203 | 0.80032 | -0.00655 |
| 022 | CatBoost | fs-014 spend clusters | 0.8145 | 0.79939 | -0.00748 |

---

## Todos los experimentos (incluidos no enviados a Kaggle)

| Exp | Modelo | Feature Set | val_acc | Kaggle | Nota |
|-----|--------|-------------|---------|--------|------|
| 001–009 | LogisticRegression | fs-001 a fs-009 | 0.788–0.790 | — | Baseline LR |
| 010 | RandomForest | fs-001 | 0.8050 | — | |
| 011 | LightGBM | fs-001 | 0.8203 | — | |
| 012 | XGBoost | fs-001 | 0.8209 | — | |
| **013** | **CatBoost** | **fs-004** | **0.8221** | **0.80687** | **Mejor Kaggle** |
| 014 | CatBoost | fs-005 structural | 0.8233 | 0.79962 | Sobreajusta val |
| 015 | CatBoost | fs-004 (más trials) | 0.8238 | — | Mismo techo |
| 016 | CatBoost | fs-011 child route OHE | — | — | |
| 017 | CatBoost | fs-012 child route TE | 0.8250 | 0.80617 | Val mejor, Kaggle peor |
| 018 | CatBoost | fs-004 (50 trials) | 0.8238 | — | Mismo techo |
| 019–020 | TabNet | fs-004 | 0.8033 | — | DLL issues Windows |
| 021 | MoE | fs-013 | 0.8203 | — | |
| 022 | CatBoost | fs-014 spend clusters | 0.8145 | 0.79939 | Features nuevas dañan |
| 023 | CatBoost | fs-015 domain rules | 0.8233 | 0.80547 | |
| 024 | LightGBM | fs-004 | 0.8203 | 0.80032 | LGB < CatBoost |
| 025 | CatBoost | fs-016 transductivo | 0.8233 | 0.80547 | Sin mejora vs 023 |

---

## Feature sets probados

| Feature Set | Descripción | Resultado vs fs-004 |
|-------------|-------------|---------------------|
| fs-001 | Baseline: Deck/Side/CabinNum, GroupSize, TotalSpending_Log, AgeCategory | Base |
| fs-002 | +Route, GroupCryoSleepRate, LuxurySpendingRatio | ❌ Peor |
| fs-003 | +IsAlone, IsChild, SpendingIntensity | ❌ Sin mejora |
| **fs-004** | TE en Deck + HomePlanet (OHE→1 num cada uno) | **✅ Mejor base** |
| fs-005 | +SpendingEntropy, GroupSpendingZScore, FamilySizeFromName... | ❌ Sobreajusta |
| fs-006 | Imputación group-aware spending | ❌ Sin mejora |
| fs-007 | Domain rules + TravelGroup_TE | ❌ Data leakage |
| fs-008 | Domain rules sin TravelGroup_TE | ❌ Sin mejora |
| fs-009 | +CabinNumber_DeckPercentile | ❌ Sin mejora |
| fs-010 | +CryoSpendingAnomaly, GroupTransportedProxy | ❌ Sin mejora |
| fs-011 | +IsChild, GroupHasChild, Route OHE | ❌ Sin mejora |
| fs-012 | fs-011 con Route como TE | ❌ Marginal (-0.0007 Kaggle) |
| fs-013 | +GroupAllCryo, GroupAnyCryo, SpendShare, GroupSpendOthers | ❌ Sin mejora |
| fs-014 | +EntertainmentSpend_Log, ComfortSpend_Log, GroupCryoSegment... | ❌ Peor |
| fs-015 | fs-004 + apply_domain_rules antes de engineering | ❌ -0.0014 Kaggle |
| fs-016 | fs-015 con imputación transductiva train+test | ❌ Igual a fs-015 |

---

## Modelos probados

| Modelo | Mejor val_acc sobre fs-004 | Kaggle | Notas |
|--------|---------------------------|--------|-------|
| LogisticRegression | 0.790 | — | Insuficiente |
| RandomForest | 0.805 | — | Insuficiente |
| LightGBM | 0.820 | 0.80032 | Peor que CatBoost |
| XGBoost | 0.821 | — | Comparable, no enviado |
| **CatBoost** | **0.825** | **0.80687** | **Mejor** |
| TabNet | 0.803 | — | DLL issues en Windows |
| MoE (CatBoost segmentado) | 0.820 | — | No supera CatBoost simple |
| Ensemble LGB+XGB+CB | — | 0.80219 | Diluye la señal de CatBoost |

---

## Patrones identificados

### 1. CatBoost + fs-004 es el techo local
Todo lo que se añade sobre fs-004 empeora o queda igual en Kaggle, incluso cuando val_accuracy sube. El modelo más simple generaliza mejor.

### 2. Gap val→Kaggle sistemático (~0.015–0.018)
| Exp | val_acc | Kaggle | Gap |
|-----|---------|--------|-----|
| 013 | 0.8221 | 0.80687 | 0.0152 |
| 017 | 0.8250 | 0.80617 | 0.0188 |
| 023 | 0.8233 | 0.80547 | 0.0178 |
| 025 | 0.8233 | 0.80547 | 0.0178 |

El gap aumenta cuando añadimos features o mejoramos la imputación, lo que sugiere **sobreajuste al split de validación fijo** (80/20 aleatorio, siempre el mismo).

### 3. Errores estructurales sin resolver
Los siguientes segmentos muestran error consistentemente alto en todos los experimentos:
- **PSO J318.5-22**: 30–33% error rate (solo 162 pasajeros en val, ~370 en test)
- **Niños (Age < 13)**: 19–28% error rate según el experimento
- **Pasajeros activos (CryoSleep=False)**: 18–20% error rate

### 4. La imputación transductiva no ayudó
Combinar train+test para imputar resolvió 56 NaN adicionales de HomePlanet y 55 de CryoSleep en el test. El score Kaggle fue idéntico a fs-015 sin transducción (0.80547), lo que indica que esos NaN residuales no eran el factor limitante.

---

## Hipótesis sobre el techo en 0.807

1. **Split de validación no representativo**: El 20% de val siempre es el mismo. El modelo puede estar sobreajustando a las características de esa partición específica. K-Fold estratificado daría una estimación más robusta.

2. **Distribución train/test ligeramente diferente**: La distribución de PSO J318.5-22 o de grupos específicos puede diferir entre train y test de Kaggle.

3. **Información que no capturamos**: Los top de Kaggle (0.83+) usan señales que no hemos identificado. Probable candidato: uso combinado de Name (apellido como proxy de familia + HomePlanet) con Target Encoding a nivel de apellido.

4. **Límite del modelo con este feature set**: CatBoost con fs-004 puede haber alcanzado su bayes error en estas features. Necesitamos features cualitativamente diferentes, no variaciones de las actuales.

---

## Lo que NO se ha probado

- [ ] **K-Fold estratificado** como estrategia de validación (en lugar del split fijo)
- [ ] **Target Encoding por apellido** (`LastName_TE`) — apellido correlaciona con HomePlanet (datos de familia), podría capturar señal sin data leakage si se hace correctamente con CV
- [ ] **Neural network (MLP sklearn)** — sin dependencia PyTorch, integración directa
- [ ] **Pseudo-labeling** — entrenar con predicciones del test como labels "blandos"
- [ ] **Feature selection activa** — eliminar features de fs-004 por importancia (¿CabinNumber y AgeCategory están añadiendo ruido?)
- [ ] **Stacking heterogéneo** — usar predicciones de CatBoost + LightGBM + LogReg como meta-features para un modelo nivel 2

---

## Configuración del mejor modelo (exp-013)

```
Modelo:    CatBoost
Feature set: fs-004_target_encoding
Artefacto: models/experiments/exp-013_CatBoost.pkl

Hiperparámetros:
  iterations:          600
  depth:               7
  learning_rate:       0.04810
  l2_leaf_reg:         18.989
  bagging_temperature: 0.00657

Features (24):
  Numéricas (14): Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
                  GroupSize, CabinNumber, TotalSpending_Log, SpendingCategories,
                  Deck_TE, HomePlanet_TE, HasSpending, CryoSleep_Encoded
  Categóricas (2): Destination, AgeCategory
  (más 8 intermedias que se dropean antes de entrenar)

Umbral de clasificación: 0.5 (ganancia marginal con umbral óptimo)
Val accuracy: 0.8221
Kaggle score: 0.80687
```
