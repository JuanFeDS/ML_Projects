# Estado de Experimentos — Spaceship Titanic

**Ultima actualizacion:** 2026-07-06 (noche — exp-055/056 ejecutados y subidos)
**Cobertura:** fs-001 a fs-019 | exp-001 a exp-056

---

## ⚠️ Habia tres "mejores modelos" distintos — resuelto el 2026-07-06

El proyecto tuvo **tres fuentes de verdad que dejaron de hablarse entre si** a partir de exp-032. El 2026-07-06 se reconfirmo exp-034 en Kaggle (submission `54408522`, score 0.80944 identico al de abril) y, al revisar el historial de submissions, se encontro que **exp-053/054 ya estaban confirmados en Kaggle desde el 2026-04-18/19 con score 0.80617** — dato que no se habia registrado en este documento. Esto cierra el pendiente "confirmar exp-053/054 en Kaggle":

| Fuente | Experimento | Metrica | Confiabilidad |
|---|---|---|---|
| `artifacts/production/model_metadata.json` (promocion oficial via `TrainingPipeline`) | **exp-031** — CatBoost, fs-019_pseudo_labeled | val_accuracy=0.8352 | ⚠️ Split random + pseudo-labeling sobre exp-027 (que ya tenia leakage de LastName). Nunca subido a Kaggle. Descartado como fuente de verdad. |
| Mejor score **Kaggle real confirmado** | **exp-034** — ensemble soft voting exp-027+exp-033 (pesos 0.5/0.5) | **0.80944** | ✅ Confirmado dos veces (2026-04-13 y 2026-07-06, mismo score exacto). Techo vigente del proyecto. |
| Estandar de **validacion honesta** (post-diagnostico de leakage, 2026-04-18) | **exp-053/054** — CatBoost + GroupKFold(LastName) | oof_acc=0.8144 / **Kaggle=0.80617** | ✅ Metodologicamente correcto, score Kaggle confirmado pero inferior a exp-034. No es optimismo falso: identifico un modelo con techo real mas bajo que el ensemble. |

**Conclusion:** exp-034 sigue siendo la fuente de verdad para submissions. GroupKFold sigue siendo el estandar de validacion obligatorio para cualquier iteracion nueva sobre fs-017 o descendientes — que su resultado en Kaggle haya sido mas bajo no invalida la metodologia, confirma que era la medicion honesta. Ver `docs/model/decision_fuente_de_verdad.md` (version corta, versionada en el repo).

**Por que importa:** los experimentos exp-032 a exp-054 (el "laboratorio" de scripts 05–13) nunca llamaron a `append_experiment_log` ni a `write_model_card` — no existen en `docs/model/experimentation_log.md` ni en `docs/model/cards/`. Su unico rastro estructurado son los JSON en `artifacts/experiments/*_metadata.json` (reconstruidos aqui) y las notas de sesion del 2026-04-18. El mecanismo de "produccion" nunca se entero de que exp-034/053 existen.

**Ademas:** el `val_accuracy` reportado por los scripts del laboratorio (exp-032 a exp-052) viene de un split random contaminado por el mismo leakage de `LastName` — por eso sube hasta 0.88 en el sweep de pesos (exp-044/045) sin que eso signifique nada real. Solo tratar como comparables entre si los numeros **dentro de la misma columna de metrica** de la tabla de la Seccion 4.2; no comparar val_accuracy de la Seccion 2 (split random, pre-GroupKFold) contra oof_acc de exp-053/054 (GroupKFold) como si midieran lo mismo.

---

## 1. Resumen ejecutivo

| Indicador | Valor |
|---|---|
| Feature sets explorados | 19 (fs-001 a fs-019) |
| Experimentos ejecutados | 54 (exp-001 a exp-054; exp-046 eliminado — artefacto parcial) |
| Mejor val_accuracy en pipeline oficial | 0.8352 (exp-031, CatBoost, fs-019 pseudo-labeled) — ver aviso de leakage arriba |
| Mejor score Kaggle confirmado | **0.80944** (exp-034, ensemble exp-027+exp-033) |
| Baseline de validacion honesta (GroupKFold) | oof_acc=0.8144, Kaggle=0.80617 (exp-053/054) — confirmado, inferior a exp-034 |
| Modelo ganador consistente | **CatBoost** (exp-013 en adelante, en cualquier variante) |
| Submissions generadas | 40 (incluye reconfirmacion de exp-034 el 2026-07-06) |
| Tecnicas que NO sumaron (veredicto propio) | TabNet (exp-019/020), MoE (exp-021), TabPFN (exp-047-051), stacking con meta-learner (exp-052) |

El proyecto alcanzo un plateau real (Kaggle) en **~0.809** usando CatBoost con features de Cabin, gasto, contexto familiar, target encoding de ruta y de apellido (LastName). El 2026-04-18 se diagnostico que el salto de OOF ~0.82→~0.86 al introducir `LastName_TE` era leakage de familias por split random, no señal real — GroupKFold(LastName) es el estandar de validacion acordado. Su score Kaggle (0.80617, confirmado 2026-07-06) quedo por debajo de exp-034, asi que el plateau vigente sigue siendo 0.80944.

---

## 2. Tabla maestra de feature sets

| FS | Experimentos | Mejor modelo | Val Acc | Val AUC | CV Acc | Trials | Estado |
|---|---|---|---|---|---|---|---|
| fs-001_baseline | exp-001, exp-013* | LogReg / CatBoost | 0.8285 | 0.9072 | 0.8128 | 25 | Baseline activo |
| fs-002_cryo_interactions | exp-002 | LogReg | 0.7868 | 0.8694 | 0.7896 | 0 | Descartado |
| fs-003_solo_interactions | exp-003 | LogReg | 0.7892 | 0.8702 | 0.7896 | 0 | Descartado |
| fs-004_target_encoding | exp-004/010/011/012/013/018 | CatBoost | 0.8238 | 0.9060 | 0.8109 | 50 | Referencia boosting |
| fs-005_structural_context | exp-005/014 | CatBoost | 0.8233 | 0.9022 | 0.8137 | 25 | Descartado |
| fs-006_group_imputation | exp-006 | LogReg | 0.7874 | 0.8685 | 0.7893 | 0 | Descartado |
| fs-007_domain_rules | exp-007 | LogReg | **0.9454** | 0.9912 | 0.9493 | 0 | ⚠ Data leakage |
| fs-008_domain_rules_only | exp-008 | LogReg | 0.7898 | 0.8703 | 0.7893 | 0 | Descartado |
| fs-009_percentile_cabin | exp-009 | LogReg | 0.7886 | 0.8705 | 0.7895 | 0 | Descartado |
| fs-010_cryo_spending | exp-015 | CatBoost | 0.8150 | 0.9045 | 0.8102 | 25 | Descartado |
| fs-011_child_route | exp-016 | CatBoost | 0.8180 | 0.9021 | 0.8140 | 25 | Superado por fs-012 |
| **fs-012_child_route_te** | **exp-017** | **CatBoost** | **0.8250** | **0.9061** | **0.8140** | **25** | Superado dentro del pipeline oficial |
| fs-013_group_context | exp-021 | MoE | 0.8203 | 0.9061 | 0.8112 | 50 | No supero produccion |
| fs-014_spend_clusters | exp-022 | CatBoost | 0.8145 | 0.8996 | 0.8112 | 25 | Descartado |
| fs-015_domain_imputation | exp-023/024 | CatBoost | 0.8233 | 0.9044 | 0.8067 | 25 | Sin mejora vs fs-004 |
| fs-016_transductive | exp-024 | CatBoost | 0.8233 | 0.9044 | 0.8067 | 25 | Sin mejora — requiere scripts/00 previo |
| fs-017_lastname_te | exp-026/027/029/030 | CatBoost | 0.8575* | 0.9379* | 0.8547* | 25 | ⚠ val inflado por leakage LastName (ver aviso) — origen tambien del "camino nativo" del laboratorio (exp-033+) |
| fs-018_group_consistency | exp-028 | CatBoost | 0.8181 | 0.9073 | 0.8158 | 25 | Sin mejora vs fs-017 |
| **fs-019_pseudo_labeled** | **exp-031** | **CatBoost** | **0.8352** | **0.9246** | **0.8347** | **0** | **Produccion oficial (`artifacts/production/`)** — ver aviso de leakage |

*exp-013 en adelante usaron CatBoost sobre fs-004 como punto de comparacion. A partir de fs-017, los val_accuracy estan inflados por el leakage de `LastName` diagnosticado el 2026-04-18 (ver aviso al inicio del documento).

---

## 3. Detalle de cada feature set

### fs-001_baseline

**Parent:** ninguno
**Features entrada al modelo:** 28 columnas (ver lista en model_metadata.json)

Pipeline base que transforma el dataset crudo en variables utilizables:

| Transformacion | Columnas generadas | Justificacion estadistica |
|---|---|---|
| `Cabin` → descomposicion | `Deck`, `CabinNumber`, `Side` | chi2 Deck=392, Side=91 (p<0.001) |
| `PassengerId` → grupo | `TravelGroup`, `GroupSize` | patron no lineal con target |
| Spending agregado | `TotalSpending_Log`, `HasSpending`, `SpendingCategories` | r_log=-0.47 vs r_raw=-0.20 |
| Encoding | `CryoSleep_Encoded`, `Side_Encoded` | binario directo |
| Categoricas | `AgeCategory` (5 bins), OHE Destination, TE Deck/HomePlanet | chi2 CryoSleep=1861 |

**Columnas descartadas:** PassengerId, Name, Cabin, TravelGroup (raw).

**Resultado:** Establece piso en 0.8285 con CatBoost tuneado. Es la base de todos los feature sets posteriores.

---

### fs-002_cryo_interactions

**Parent:** fs-001
**Features nuevas (+6):** `Route`, `GroupCryoSleepRate`, `CryoSleepViolation`, `LuxurySpendingRatio`, `CabinNumber_DeckPercentile`, `GroupSpendingMean`

Intento de capturar interacciones entre CryoSleep, el comportamiento del grupo y la posicion en el barco. `Route` concatena HomePlanet+Destination como variable categorica.

**Resultado:** 0.7868. Las interacciones son redundantes para arboles de decision que ya las aprenden internamente. La multicolinealidad daña modelos lineales.

---

### fs-003_solo_interactions

**Parent:** fs-001
**Features nuevas (+3):** `IsAlone`, `IsChild`, `SpendingIntensity`

Features simples e interpretables: viajero solo, menor de 13 años, y gasto por servicio utilizado.

**Resultado:** 0.7892. Mejora minima. `IsChild` tiene potencial (retomado en fs-011).

---

### fs-004_target_encoding

**Parent:** fs-001
**Cambio:** `Deck` y `HomePlanet` reemplazados de OHE a Target Encoding con suavizado bayesiano (smoothing=10)

Target Encoding introduce una señal ordinal real: si Deck B tiene 80% de transportados y Deck G tiene 30%, el modelo recibe esa diferencia directamente. OHE trata todas las cubiertas como igualmente distintas.

**Resultado:** 0.8238 con CatBoost (50 trials Optuna). Se convierte en la **referencia para todos los experimentos de modelos** (exp-010 a exp-018). El encoder serializado en `models/experiments/target_encoder_fs-004.pkl` garantiza que el mismo encoding se aplique a test.

---

### fs-005_structural_context

**Parent:** fs-001
**Features nuevas (+8):** `SpendingEntropy`, `GroupSpendingZScore`, `CabinNeighborhoodDensity`, `FamilySizeFromName`, `GroupCryoAlignment`, `GroupAgeDispersion`, `SpendingCategoryProfile`, `SpendingCategoryProfile_TE`

El conjunto mas ambicioso de features de contexto. Incluye entropia de Shannon del gasto, desviacion Z dentro del grupo, densidad de cabinas vecinas, tamanio de familia por apellido y dispersion de edad del grupo.

**Resultado:** 0.8233. Sin mejora real sobre fs-004. El ruido de grupos pequenos diluye la señal de las features de grupo. `FamilySizeFromName` tiene alta cardinalidad y no generaliza.

---

### fs-006_group_imputation

**Parent:** fs-001
**Cambio:** imputacion de gasto nulo con mediana del TravelGroup (en lugar de 0 global)

Para pasajeros activos (no CryoSleep) con spending NaN, usa la mediana del grupo como proxy mas realista que cero.

**Resultado:** 0.7874. La imputacion por grupo introduce sesgo cuando el grupo tiene comportamiento mixto (miembros con patrones de gasto muy distintos). Con solo ~2% de nulos en spending, la ganancia no compensa.

---

### fs-007_domain_rules ⚠ DATA LEAKAGE

**Parent:** fs-001
**Features nuevas (+1):** `TravelGroup_TE`
**Cambio:** imputacion de nulos aplicando 6 reglas fisicas antes de cualquier otra transformacion

Las 6 reglas: HomePlanet por grupo, Deck A/B/C→Europa, Deck G→Earth, Deck/Side por grupo, CryoSleep=True→spending=0, spending>0→CryoSleep=False, Age<=12→spending=0.

**Resultado:** 0.9454 en validacion — anomalia confirmada. Las reglas de CryoSleep codifican el target casi directamente (CryoSleep predice Transported con >80% de precision). La submission confirma el leakage: tasa de transporte predicha 50.2%, la mas baja de todas las submissions. **Este experimento queda invalidado.**

La leccion: las reglas de dominio son utiles para **imputar nulos** pero no como **features directas** cuando derivan del mismo mecanismo que genera el target.

---

### fs-008_domain_rules_only

**Parent:** fs-001
**Cambio:** mismo pipeline de imputacion de fs-007 pero sin `TravelGroup_TE`

Version depurada: aplica las reglas fisicas solo para resolver nulos, sin agregar features de encoding del grupo.

**Resultado:** 0.7898. Sin TravelGroup_TE la mejora desaparece completamente. Confirma que toda la ganancia de fs-007 era leakage.

---

### fs-009_percentile_cabin

**Parent:** fs-008
**Cambio:** `CabinNumber` reemplazado por `CabinNumber_DeckPercentile`

Motivado por adversarial validation: AUC=0.79 al distinguir train vs test. `CabinNumber` era la feature con mayor distributional shift (los rangos de numeros de cabina son distintos entre splits). El percentil dentro del deck normaliza la posicion relativa y deberia generalizar mejor.

**Resultado:** 0.7886. La normalizacion no aporta señal adicional. El numero de cabina en si mismo tiene poca importancia en modelos de arboles.

---

### fs-010_cryo_spending

**Parent:** fs-004
**Features nuevas (+4):** `CryoSpendingAnomaly`, `GroupTransportedProxy`, `SideSpendingDiff`, `CryoSleepBinary`

Interacciones CryoSleep x spending: flags de anomalia fisica (gasto mientras duerme), proxy del grupo transportado, asimetria de gasto entre lados P/S, y CryoSleep como numerico {-1, 0, 1}.

**Resultado:** 0.8150. Por debajo de fs-004. El modelo ya captura estas interacciones sin necesitar features explicitas.

---

### fs-011_child_route

**Parent:** fs-004
**Features nuevas (+4):** `IsChild`, `GroupHasChild`, `GroupChildRate`, `Route` (OHE, 9 categorias)

Dirigido a los segmentos con mayor tasa de error en exp-013: ninos (28% error) y destino PSO J318.5-22 (30% error). `Route` combina HomePlanet + Destination como categorica.

**Resultado:** 0.8180. Mejora real sobre fs-010. `IsChild` y el contexto familiar aportan. `Route` como OHE genera 9 columnas con poca frecuencia en algunas combinaciones.

---

### fs-012_child_route_te ← PRODUCCION

**Parent:** fs-011
**Features nuevas (+1):** `Route_TE` (reemplaza a Route OHE)

`Route_TE` codifica la tasa media de transporte por ruta HomePlanet→Destination como un solo numero. PSO J318.5-22 tiene tasas de transporte muy distintas segun el origen del pasajero — Route_TE captura esa señal ordinal que 9 columnas OHE diluian.

**Features del modelo en produccion (28):**

| Categoria | Features |
|---|---|
| Gasto (raw) | Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck |
| Gasto (derivado) | TotalSpending_Log, HasSpending, SpendingCategories |
| Cabin | CabinNumber, Side_Encoded, Deck_TE |
| Grupo | GroupSize, IsChild, GroupHasChild, GroupChildRate |
| Planeta/ruta | HomePlanet_TE, Route_TE |
| Destino (OHE) | Destination_55 Cancri e, Destination_PSO J318.5-22, Destination_TRAPPIST-1e, Destination_Unknown |
| Cryo | CryoSleep_Encoded |
| Edad | AgeCategory_Adult, AgeCategory_Child, AgeCategory_Senior, AgeCategory_Teen, AgeCategory_YoungAdult |

**Hiperparametros CatBoost:** iterations=400, depth=7, lr=0.0557, l2=12.05, bag_temp=0.753
**Threshold optimo:** 0.4106 (ajustado por busqueda de threshold, no 0.5 default)

---

### fs-013_group_context

**Parent:** fs-004
**Features nuevas (+4):** `GroupAllCryo`, `GroupAnyCryo`, `GroupSpendOthers_Log`, `SpendShare`

Motivado por el heatmap HomePlanet x CryoSleep del EDA: pasajeros europeos en CryoSleep tienen >86% de tasa de transporte. `GroupAllCryo`/`GroupAnyCryo` codifican el consenso criogenico del grupo. `SpendShare` es la fraccion de gasto del pasajero sobre el total del grupo.

**Resultado:** 0.8203. Por debajo de fs-012. El consenso grupal aporta señal pero no supera la combinacion de contexto familiar + Route_TE. El modelo ganador fue MoE (Mixture of Experts) en vez de CatBoost vanilla, aunque tampoco supero produccion.

---

### fs-014_spend_clusters

**Parent:** fs-013
**Features nuevas (+6):** `EntertainmentSpend_Log`, `ComfortSpend_Log`, `EntVsComfort_Ratio`, `IsExtremeSpender`, `AgeVsPlanetMedian`, `GroupCryoSegment`

Derivadas del EDA del 2026-04-12: agrupa gasto en "entretenimiento" (FoodCourt+VRDeck+Spa) vs "confort" (RoomService+ShoppingMall), flag de gasto extremo (>p99), edad relativa a la mediana del HomePlanet, y un ordinal 0-3 de consenso CryoSleep del grupo.

**Resultado:** 0.8145. Por debajo de fs-013. Los clusters de gasto no aportan sobre lo que CatBoost ya captura internamente.

---

### fs-015_domain_imputation

**Parent:** fs-004
**Cambio:** imputacion agresiva por reglas de dominio (HomePlanet por grupo/Deck, CryoSleep por gasto) aplicada antes del resto del pipeline, misma dimensionalidad que fs-004.

**Resultado:** 0.8233 — identico a fs-004 en la practica. La calidad de imputacion no es el cuello de botella en este punto del plateau.

---

### fs-016_transductive

**Parent:** fs-015
**Cambio:** imputacion transductiva combinando train+test (`scripts/00_transductive_impute.py`) antes de aplicar las reglas de dominio, para que grupos con miembros en ambos splits compartan informacion.

**Resultado:** 0.8233 — sin cambio medible sobre fs-015. Requiere un paso previo fuera del pipeline oficial (`00_transductive_impute.py`), lo que ya era una señal temprana de que el flujo se estaba ramificando.

---

### fs-017_lastname_te ⚠ ORIGEN DEL LEAKAGE DIAGNOSTICADO

**Parent:** fs-004
**Features nuevas (+2):** `LastName`, `LastName_TE`

Target Encoding de `LastName` (apellido, proxy de familia) con suavizado bayesiano (exp-026, k=30) y luego fold-aware con `sklearn.TargetEncoder` (exp-027). Salto aparente de val_accuracy de 0.8181→0.8575 (exp-026) sobre split random.

**Resultado:** el salto **no era señal real** — el diagnostico del 2026-04-18 confirmo que personas del mismo apellido caian en train y val simultaneamente, y el modelo memorizaba destinos familiares en vez de generalizar. Con `GroupKFold(LastName)` (exp-053), el oof_acc honesto cae a 0.8144 — muy por debajo del 0.8575 reportado aqui. Aun asi, `fs-017` es el feature set que mas se siguio usando: es la base de todo el "laboratorio" (exp-032 en adelante) y del camino de preprocesamiento nativo (`src/preprocessing/common.py`, `FS_NATIVE`).

---

### fs-018_group_consistency

**Parent:** fs-017
**Features nuevas (+3):** `GroupAllSameDest`, `GroupAllSameHomePlanet`, `GroupConsistencyScore`

Señal estructural de grupo sin usar el target: si todos comparten destino/planeta de origen.

**Resultado:** 0.8181 (val, misma advertencia de leakage que fs-017). No mejora sobre fs-017 por si sola.

---

### fs-019_pseudo_labeled ← PRODUCCION OFICIAL (`artifacts/production/`)

**Parent:** fs-017
**Cambio:** pseudo-labeling — se agregan al train 985 filas de test.csv donde exp-027 predice con confianza ≥0.95 (711 True, 274 False, confianza media 97.6%). Mismo pipeline que fs-017, dataset +11.3% (8,693→9,678 filas).

**Resultado:** 0.8352 val_accuracy — el numero mas alto jamas registrado por el mecanismo oficial de promocion, por eso quedo como "produccion". Pero hereda el leakage de `LastName` de fs-017 **y ademas** usa pseudo-etiquetas generadas por un modelo (exp-027) que ya estaba sobreestimado por ese mismo leakage — es la combinacion mas optimista de todo el historial y la menos confiable. Nunca se confirmo contra Kaggle.

---

## 4. Arbol de linaje

```
fs-001_baseline
├── fs-002_cryo_interactions      [descartado]
├── fs-003_solo_interactions      [descartado]
├── fs-004_target_encoding
│   ├── fs-010_cryo_spending      [descartado]
│   ├── fs-011_child_route
│   │   └── fs-012_child_route_te
│   ├── fs-013_group_context      [no supero]
│   │   └── fs-014_spend_clusters [descartado]
│   ├── fs-015_domain_imputation  [sin mejora]
│   │   └── fs-016_transductive   [sin mejora, requiere scripts/00]
│   └── fs-017_lastname_te ⚠ leakage diagnosticado 2026-04-18
│       ├── fs-018_group_consistency [sin mejora]
│       ├── fs-019_pseudo_labeled  ← PRODUCCION OFICIAL (leaky)
│       └── (camino nativo, fuera de FEATURE_SETS — ver Seccion 4.2)
│           └── src/preprocessing/common.py::FS_NATIVE = fs-017
│               ├── exp-033 CatBoost native
│               ├── exp-034 ensemble 027+033 ← MEJOR KAGGLE REAL (0.80944)
│               ├── exp-047 TabPFN            [descartado]
│               ├── exp-052 stacking LR       [descartado]
│               └── exp-053/054 GroupKFold(LastName) ← NUEVO ESTANDAR HONESTO
├── fs-005_structural_context     [descartado]
├── fs-006_group_imputation       [descartado]
├── fs-007_domain_rules           [leakage - invalidado]
│   └── fs-008_domain_rules_only  [descartado]
│       └── fs-009_percentile_cabin [descartado]
```

---

## 4.2 Laboratorio de iteracion — exp-032 a exp-054

**Por que esta seccion existe aparte:** estos experimentos corrieron desde `scripts/05` a `scripts/13` (fuera de `run.py`), reimplementando su propio preprocesamiento (`src/preprocessing/common.py`) en paralelo a `FEATURE_SETS`. Ninguno llama a `append_experiment_log` — su unico registro estructurado es `artifacts/experiments/exp-0XX_*_metadata.json`, reconstruido aqui. **exp-046 no aparece: era un artefacto parcial, eliminado manualmente el 2026-04-18.**

| Exp | Script origen | Que probo | Metrica registrada | Veredicto |
|---|---|---|---|---|
| 032 | 07_segmented_train | CatBoost separado por segmento CryoSleep (cryo/activo) | oof_acc combinado 0.8146 | No mejora sobre fs-017 vanilla |
| 033 | 08_catboost_native | CatBoost con categoricas nativas (Deck/HomePlanet/Destination/AgeCategory/LastName sin OHE/TE manual) | oof_acc 0.8177 | ✅ Se vuelve el segundo modelo base del mejor ensemble |
| 034–043 | 05_ensemble | Sweep de pesos soft-voting entre exp-027 y exp-033 (10 combinaciones) | val_accuracy 0.8602→0.8655 (split random, leaky) | **exp-034 (pesos 50/50) es el unico con Kaggle confirmado: 0.80944** — el resto son exploracion offline, val_accuracy no comparable a Kaggle |
| 044–045 | 05_ensemble | Ensemble de 3 modelos (027+029 LightGBM+033) | val_accuracy 0.8843 (mismo leakage, no confirmado en Kaggle) | No reemplazo a exp-034 |
| 047 | 10_tabpfn_train | TabPFN cloud (foundation model tabular, in-context learning) | oof_acc 0.8168 | ❌ Descartado — aprende la misma señal que CatBoost, sin diversidad de error |
| 048–051 | 11_ensemble_tabpfn | Soft voting 027+033+047 con distintos pesos | val_accuracy 0.8479–0.8561 | ❌ Peor que el ensemble sin TabPFN (0.80780 vs 0.80944 en Kaggle, por notas de sesion) |
| 052 | 12_stacking | Meta-learner LogisticRegression sobre OOF de 027+033+047 | oof_acc_meta 0.8204 | ❌ Descartado — no supero al soft voting simple |
| 053/054 | 13_groupkfold_train | CatBoost + GroupKFold(LastName), threshold=0.5 fijo | oof_acc 0.8144, oof_roc_auc 0.9065, **Kaggle 0.80617** (confirmado 2026-04-18/19) | ✅ Nuevo estandar de validacion honesta — confirmado en Kaggle, inferior a exp-034 (0.80944). exp-054 repite los mismos parametros que exp-053 (misma corrida, no hubo cambio real entre ambos) |
| 055 | 13_groupkfold_train --group-features | exp-053 + 3 group features (GroupCryoRate, GroupMeanSpending_Log, GroupSpendingRank) — propuesta del 2026-04-18, por fin implementada | oof_acc 0.8151, oof_roc_auc 0.9049 | Δ +0.0007 vs exp-053 — dentro del ruido (SE~0.004). No subido a Kaggle (regla: solo si oof>=0.820) |
| 056 | 13_groupkfold_train --group-features --tune 25 | exp-055 + re-tuning Optuna con GroupKFold como CV (antes venian de StratifiedKFold via exp-033). Params nuevos mas conservadores: depth 6, lr 0.031, 655 iter | oof_acc 0.8167, oof_roc_auc 0.9039, **Kaggle 0.80710** | Δ +0.0023 vs exp-053 en OOF, +0.0009 en Kaggle. Subido como dato de calibracion OOF→leaderboard (ver Seccion 6). Mejor que exp-053 pero sigue debajo de exp-034 |

**Conclusion del laboratorio:** el techo real del proyecto (Kaggle) sigue siendo **0.80944** (exp-034). Todo el trabajo posterior (segmentacion, TabPFN, stacking) confirmo lo mismo que el plateau ya sugeria: CatBoost sobre fs-017/fs-004 con encoding de ruta y apellido captura la señal disponible; variarle la arquitectura del modelo no suma. El diagnostico de leakage del 2026-04-18 fue el hallazgo mas valioso del periodo, no una tecnica de modelado.

---

## 5. Lo que se ha explorado

### Tecnicas de encoding
- [x] One-Hot Encoding (baseline)
- [x] Target Encoding con suavizado (Deck, HomePlanet, Route)
- [x] Encoding binario (CryoSleep, Side)
- [ ] Ordinal Encoding
- [ ] Embeddings categoricos (neural)

### Feature engineering
- [x] Descomposicion de Cabin (Deck/CabinNumber/Side)
- [x] Extraccion de grupo desde PassengerId
- [x] Spending agregado + log1p
- [x] Categorias de edad
- [x] Interacciones CryoSleep x spending
- [x] Features de grupo (size, cryo ratio, spending medio)
- [x] Features de ninos (IsChild, GroupHasChild, GroupChildRate)
- [x] Route como categorica y como TE
- [x] Percentil de cabina dentro del deck
- [x] Entropia de gasto
- [x] Z-score de gasto dentro del grupo
- [x] Consenso CryoSleep del grupo (GroupAllCryo, GroupAnyCryo)
- [x] Target Encoding de LastName (fs-017) — ⚠ requiere GroupKFold, no split random
- [x] Clusters de gasto entretenimiento/confort (fs-014) — no aporto
- [x] Coherencia de grupo (mismo destino/planeta) (fs-018) — no aporto
- [ ] Embeddings de nombre propio
- [ ] Features de posicion espacial avanzada (zonas del barco)
- [ ] GroupHomePlanet_TE (TE del planeta mayoritario del grupo)
- [ ] Destination x CryoSleep interaccion explicita
- [ ] Imputacion Age por mediana del TravelGroup (antes del global)
- [ ] Group features sobre GroupKFold honesto (GroupCryoRate, GroupMeanSpending, spending rank) — pendiente desde 2026-04-18, no intentado aun con validacion correcta

### Imputacion
- [x] Mediana/moda global
- [x] Reglas de dominio fisicas (CryoSleep→spending, Deck→HomePlanet)
- [x] Mediana por grupo de viaje (spending)
- [ ] KNN imputation
- [ ] Iterative imputer (MICE)
- [ ] Imputacion por modelo predictor

### Modelos
- [x] Logistic Regression (baseline)
- [x] Random Forest
- [x] LightGBM
- [x] XGBoost
- [x] CatBoost (ganador consistente, en toda variante de preprocesamiento)
- [x] TabNet (exp-019/020 — no supero CatBoost)
- [x] Mixture of Experts (exp-021 — no supero CatBoost; sigue corriendo por defecto en cada train)
- [x] TabPFN cloud foundation model (exp-047 — aprende la misma señal que CatBoost, sin diversidad)
- [x] HistGradientBoosting de sklearn (parte del catalogo desde exp-025)
- [x] Stacking / blending (exp-034-045 soft voting, exp-052 meta-learner LR — soft voting simple 027+033 es lo unico que se sostiene)
- [ ] Red neuronal tabular (MLP, FT-Transformer)

### Estrategias de entrenamiento
- [x] StratifiedKFold 5-fold + hold-out 20% (split random — confirmado con leakage de grupos en LastName)
- [x] Tuning Optuna (25-50 trials)
- [x] Threshold adjustment (0.41-0.51 segun experimento)
- [x] Pseudo-labeling con datos de test (fs-019/exp-031 — hereda el leakage de su modelo fuente, no confiable)
- [x] GroupKFold por LastName (exp-053/054) — nuevo estandar de validacion honesta
- [ ] Entrenamiento sobre train+val completo con params fijos
- [ ] Optuna con 100+ trials sobre fs-012
- [ ] Early stopping con mas iteraciones
- [ ] Re-tuning de hiperparametros usando GroupKFold como CV (los params actuales de exp-053/054 vienen de StratifiedKFold)

---

## 6. Analisis de donde estamos (actualizado 2026-07-05)

### El plateau real esta en Kaggle, no en val_accuracy
Los val_accuracy de la Seccion 2/4.2 subieron hasta 0.88 en algun punto, pero eso fue leakage, no progreso. El numero que importa es el confirmado contra Kaggle: **0.80944** (exp-034), sin movimiento desde el 2026-04-13. Todo lo intentado despues (segmentacion por CryoSleep, TabPFN, stacking, mas feature engineering sobre fs-013/014/018) confirmo el mismo techo en vez de romperlo.

### Diagnostico de leakage (2026-04-18) — el hallazgo mas importante del periodo
`LastName_TE` con split random hacia que miembros de la misma familia cayeran en train y val simultaneamente — el modelo memorizaba destinos familiares en vez de generalizar. Con `GroupKFold(LastName)` el oof_acc honesto de CatBoost cae de ~0.8177 a **0.8144**. Este es ahora el numero de referencia correcto para cualquier experimento futuro sobre `fs-017` o el camino nativo.

### Calibracion OOF→Kaggle (2026-07-06) — el gap es estable
Con dos puntos confirmados, el mapeo del OOF GroupKFold al leaderboard es consistente:

| Exp | OOF | Kaggle | Gap |
|---|---|---|---|
| 053 | 0.8144 | 0.80617 | -0.008 |
| 056 | 0.8167 | 0.80710 | -0.010 |

Las mejoras de OOF honesto **si se traducen** a Kaggle (direccion y magnitud consistentes). Con gap ~-0.009, batir el techo de exp-034 (0.80944) requiere **oof_acc >= ~0.819-0.820** — esto valida empiricamente la regla del flujo experimental (subir solo con oof>=0.820). Corolario: el margen restante sobre fs-017 es de ~0.003-0.004 de OOF; si ninguna hipotesis nueva lo cierra, el plateau es real y esta en las features, no en el modelo.

### El registro de experimentos se desconecto en exp-032
`docs/model/experimentation_log.md`, `docs/model/cards/` y `artifacts/production/` solo conocen hasta exp-031. Todo el trabajo de exp-032 a exp-054 (incluyendo el mejor resultado real, exp-034) vive unicamente en `artifacts/experiments/*.json` y en notas de sesion. Esto no es un problema de disciplina — es que los scripts del laboratorio nunca importaron `src.models.tracking` ni `append_experiment_log`.

### Señales de error conocidas (de exp-017, ultimo analisis de errores documentado)
- **Ninos (IsChild=1):** tasa de error ~28% — spending=0 por regla, señal casi exclusivamente demografica
- **Destino PSO J318.5-22:** tasa de error ~30% — comportamiento atipico segun HomePlanet de origen
- **Pasajeros solos en edad media (25-45):** perfil heterogeneo, mayor incertidumbre del modelo

No hay analisis de errores equivalente para exp-034 o exp-053 — pendiente si se quiere seguir iterando sobre el camino nativo/GroupKFold.

---

## 7. Propuestas para proximos experimentos (revisado — varias ya se ejecutaron)

### Ya ejecutadas desde la version anterior de este documento (2026-04-11)
- ~~Pseudo-labeling~~ → hecho en fs-019/exp-031, pero heredo el leakage de su fuente — no confiable tal cual
- ~~Stacking CatBoost+LightGBM+XGBoost~~ → hecho en exp-052 (meta-learner LR sobre 027+033+047) — no supero al soft voting simple
- ~~Mas trials Optuna~~ → no se re-hizo sobre fs-012, pero si se re-tuneo CatBoost dentro del camino nativo (exp-033, 546 iteraciones)
- ~~GroupHomePlanet_TE / interacciones de grupo~~ → exploradas parcialmente en fs-013/fs-018, sin mejora
- ~~Confirmar exp-053/054 en Kaggle~~ → **resuelto 2026-07-06**: ya estaba confirmado desde el 2026-04-18/19 con score 0.80617, inferior a exp-034 (0.80944). GroupKFold no era optimista falso, simplemente identifico un techo real mas bajo.
- ~~Decidir destino de TabNet/TabPFN/stacking~~ → hecho en la auditoria del 2026-07-05: codigo eliminado, veredicto negativo confirmado (ver `2026-07-05-session.md`)

### Ejecutadas el 2026-07-06 (exp-055/056)
- ~~Group features sobre GroupKFold honesto~~ → exp-055: +0.0007 OOF, dentro del ruido
- ~~Re-tuning con GroupKFold como CV~~ → exp-056: +0.0023 OOF acumulado, Kaggle 0.80710. Mejora real pero insuficiente. Los dos juntos cerraron ~la mitad del gap hacia 0.820

### Prioridad alta — siguientes hipotesis

**A. Ensemble exp-027 + exp-033 + exp-056**
El mejor Kaggle real (exp-034, 0.80944) es un ensemble de dos modelos. exp-056 aprende con regularizacion distinta (params GroupKFold-tuned, mas conservadores) y podria aportar diversidad de error. Barato: solo inferencia, sin reentrenar. Riesgo: exp-048-051 mostraron que agregar un tercer modelo sin diversidad real empeora.

**B. Calibracion de probabilidades**
Platt scaling o isotonic sobre el OOF de exp-056 — el threshold optimo varia entre experimentos (0.41-0.55) y una probabilidad calibrada podria ganar las ~0.003 de OOF que faltan sin tocar features.

**C. GroupKFold con otra granularidad**
n_splits=10, o agrupar por `TravelGroup` en vez de `LastName` (la señal de grupo de viaje es mas fuerte que la familiar segun el EDA).

### Prioridad media

**C. Fusionar `FEATURE_SETS` con el camino nativo**
No es una mejora de score, pero es la que mas devuelve inversion: mientras existan dos sistemas de preprocesamiento paralelos, cada iteracion nueva seguira reescribiendo logica en vez de sumar una entrada de configuracion. Ver Seccion 4.2 y `docs/restructuring-plan.md`. Ademas evitaria colisiones de ID como la del 2026-07-06 (`get_next_exp_id` asigno "032" a una reproduccion de exp-034 porque el log oficial no conoce el laboratorio).

**D. Calibracion de probabilidades**
El threshold optimo varia bastante entre experimentos (0.41 a 0.55) — Platt scaling o isotonic regression podrian estabilizarlo en vez de fijarlo por busqueda de grilla en cada corrida.

### Prioridad baja — exploratorio, sin intentar aun
- Embeddings de nombre propio / KNN imputation / MICE
- Red neuronal tabular (MLP, FT-Transformer) — distinto de TabNet, no descartado por veredicto propio
- Feature selection por SHAP sobre el modelo ganador final (recordar: SHAP solo sobre el ganador, no en exploracion)

---

## 8. Proximo paso recomendado

Estado tras el ciclo del 2026-07-06: exp-034 sigue siendo el techo (0.80944), pero exp-056 (0.80710) demostro que el OOF honesto se traduce a Kaggle con gap estable ~-0.009. La regla del flujo experimental queda: **subir solo con oof_acc >= 0.820**.

1. **Ensemble 027+033+056** (Propuesta A) — la via mas barata de probar; validar primero el OOF del ensemble antes de subir
2. **Calibracion de probabilidades** (Propuesta B) — si A no alcanza
3. **Fusionar los dos caminos de preprocesamiento** — no bloqueante, pero evita seguir acumulando deuda tecnica y colisiones de ID como la del 2026-07-06 (mitigadas por ahora con --exp-id manual en el script 13)
