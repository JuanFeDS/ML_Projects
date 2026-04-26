# De 0.789 a 0.825: construyendo un clasificador de supervivencia para el Spaceship Titanic

> Un recorrido por 21 experimentos, 13 conjuntos de features y varios modelos para resolver
> una de las competencias más populares de Kaggle.

---

## El problema

En el año 2912, la nave espacial *Titanic* —en su viaje inaugural desde el Sistema Solar hacia
tres exoplanetas habitables— colisionó con una anomalía espacio-temporal. La mitad de sus
casi 13.000 pasajeros fue transportada a otra dimensión.

El reto: **predecir cuáles pasajeros fueron transportados** a partir de los registros
recuperados de la nave. Un problema de clasificación binaria con datos tabulares reales y
varios desafíos propios: valores faltantes con estructura física, variables categóricas con
jerarquías implícitas y patrones de grupo que afectan el destino individual.

El dataset tiene **8.693 filas de entrenamiento** y **4.277 de test**, con 13 variables
originales que incluyen datos demográficos, de cabina y de gasto en servicios de lujo a bordo.

---

## El pipeline de trabajo

El proyecto sigue una arquitectura limpia: toda la lógica de transformación vive en `src/`,
los notebooks orquestan el flujo, y cada experimento queda registrado en un log con sus
métricas, hiperparámetros y artefactos. El seguimiento de experimentos usa **MLflow**.

El flujo tiene cinco etapas:

```
Exploración → Análisis vs target → Feature engineering → Entrenamiento → Predicciones
```

Cada feature set es versionado (`fs-001`, `fs-002`, ...) y cada experimento (`exp-001`,
`exp-002`, ...) queda ligado al feature set y modelo con el que fue entrenado. Esto permite
reproducibilidad total y comparación justa entre enfoques.

---

## Fase 1: entender los datos

Antes de tocar un modelo, el análisis exploratorio reveló patrones que guiaron todo el
trabajo posterior:

**CryoSleep es la señal más fuerte.** Los pasajeros en suspensión criogénica tienen una tasa
de transporte radicalmente distinta a los activos. Además, el dataset tiene una regla física
implícita: si `CryoSleep = True`, el gasto en todos los servicios debería ser 0. Muchos
valores faltantes en spending se pueden imputar con esta regla.

**La cabina esconde información.** La columna `Cabin` tiene formato `Deck/CabinNumber/Side`.
El Deck (la cubierta de la nave) correlaciona fuertemente con el planeta de origen —las
cubiertas A, B y C son casi exclusivamente pasajeros de Europa— y con la tasa de transporte.

**Los grupos viajan juntos.** El ID del pasajero codifica un número de grupo (`XXXX_YY`).
Los miembros del mismo grupo comparten planeta de origen, deck y patrones de comportamiento.
Esto abre la puerta a features de contexto colectivo.

**Hay valores faltantes con estructura.** Los NaN en spending de pasajeros no-cryo son
distintos a los de pasajeros cryo. Los NaN en Age, HomePlanet o Deck a menudo se pueden
inferir a partir de otros miembros del mismo grupo de viaje.

---

## Fase 2: feature engineering iterativo

Se construyeron 13 conjuntos de features, cada uno añadiendo o probando hipótesis específicas.

### El punto de partida: `fs-001_baseline`

Features base extraídas directamente de las columnas originales:
- `Deck`, `Side`, `CabinNumber` extraídos de `Cabin`
- `GroupSize` extraído del ID
- `TotalSpending_Log` (suma de gasto, transformada log1p)
- `SpendingCategories` (cuántos servicios usó el pasajero)
- `AgeCategory` (Child / Teen / YoungAdult / Adult / Senior)

### El descubrimiento clave: reglas físicas del dominio (`fs-007`)

La mayor ganancia de todo el proyecto no vino de un modelo más complejo, sino de
**imputar correctamente los valores faltantes usando las reglas físicas de la nave**:

1. Si `CryoSleep = True` → todos los gastos son 0
2. Si gasto > 0 → `CryoSleep = False`
3. Si `Age <= 12` → gasto = 0
4. `HomePlanet` se puede inferir por grupo o por deck (A/B/C → Europa, G → Earth)
5. `Deck` y `Side` se infieren por grupo de viaje

El resultado fue un salto artificial de 0.789 a **0.945** en validación... que resultó ser
data leakage. Al agregar `TravelGroup_TE` (target encoding del grupo), el modelo básicamente
memorizaba quién fue transportado por grupo. Útil como diagnóstico, pero no válido para
producción.

### Target encoding: `fs-004`

En lugar de One-Hot Encoding para las variables categóricas, se reemplazaron `Deck` y
`HomePlanet` por su **tasa media de transporte** (con suavizado para evitar leakage). Esto
redujo la dimensionalidad de 35 a 24 features y capturó información ordinal que OHE no puede
expresar. Fue el feature set base de casi todos los experimentos posteriores.

### Lo que no funcionó

| Feature set | Hipótesis | Resultado |
|---|---|---|
| `fs-002` | Route, GroupCryoRate, CryoSleepViolation | -0.002 vs baseline |
| `fs-005` | SpendingEntropy, GroupAgeDispersion, CabinNeighborhoodDensity | ❌ no mejoró |
| `fs-006` | Imputación de spending por mediana del grupo | ❌ no mejoró |
| `fs-010` | Interacciones CryoSleep × gasto | -0.007 vs mejor |
| `fs-011` | IsChild, GroupHasChild, Route OHE | -0.005 vs mejor |

### Lo que sí funcionó

| Feature set | Añadió | Val acc |
|---|---|---|
| `fs-004_target_encoding` | Deck_TE, HomePlanet_TE | 0.822 |
| `fs-012_child_route_te` | Route_TE (encoding de ruta completa) | **0.825** |
| `fs-013_group_context` | GroupAllCryo, GroupAnyCryo, SpendShare, GroupSpendOthers_Log + Age imputada por grupo | 0.820 |

`fs-013` introdujo features de comportamiento colectivo inspiradas en soluciones top de
Kaggle: si **todos** los miembros del grupo están en CryoSleep, la tasa de transporte sube
de 42.4% a 80.5%. `GroupAllCryo` y `GroupAnyCryo` capturan esta señal con más granularidad
que la variable individual.

---

## Fase 3: evolución de modelos

### De regresión logística a gradient boosting

El primer benchmark fue una **Regresión Logística**: rápida, interpretable, y un buen
termómetro para saber si las features aportan señal. Llegó hasta 0.790 de accuracy en
validación.

El salto real llegó al cambiar de familia de modelos:

| Experimento | Modelo | Feature Set | Val accuracy |
|---|---|---|---|
| exp-001 | Logistic Regression | fs-001 | 0.789 |
| exp-010 | Random Forest | fs-004 | 0.805 |
| exp-011 | LightGBM | fs-004 | 0.820 |
| exp-012 | XGBoost | fs-004 | 0.821 |
| exp-013 | **CatBoost** | **fs-004** | **0.822** |
| exp-017 | CatBoost | fs-012 | **0.825** |

Los tres modelos de gradient boosting (LightGBM, XGBoost, CatBoost) se entrenaron sobre el
mismo feature set `fs-004` con tuning de hiperparámetros usando **Optuna** (25-50 trials de
búsqueda bayesiana). CatBoost se llevó el primer puesto, probablemente por su manejo nativo
de variables categóricas.

### Tuning de hiperparámetros

Optuna realiza búsqueda bayesiana: cada trial evalúa una combinación de hiperparámetros
y los resultados informan los siguientes trials. Para CatBoost, los parámetros más relevantes
fueron `depth` (profundidad de árboles), `learning_rate` e `iterations`.

Los mejores parámetros de exp-013:
```
iterations: 600  |  depth: 7  |  learning_rate: 0.048
l2_leaf_reg: 19  |  bagging_temperature: 0.007
```

### Mixture of Experts (MoE)

El pipeline incluye un componente de **Mixture of Experts**: entrena dos submodelos
especializados, uno para pasajeros en CryoSleep y otro para pasajeros activos. El
clasificador final combina las predicciones según el estado cryo del pasajero.

La lógica es clara: el comportamiento de un pasajero inconsciente en criogenia es
fundamentalmente distinto al de uno activo gastando en servicios de lujo. El MoE ganó
al CatBoost estándar en varios experimentos por un margen de ~0.004.

### El experimento con TabNet (exp-019)

**TabNet** es una arquitectura de red neuronal diseñada específicamente para datos tabulares.
A diferencia de los MLPs convencionales, usa mecanismos de atención para seleccionar
features en cada capa de decisión, aproximando el comportamiento de los árboles de decisión
pero con la flexibilidad de las redes neuronales.

El resultado: **0.805** de val accuracy —2 puntos por debajo de CatBoost. Con 8.693 filas,
los modelos de boosting tienen ventaja estructural: TabNet brilla con datasets más grandes.
Además, instalar PyTorch en Windows presentó incompatibilidades de DLL que complicaron la
ejecución, y fue necesario degradar de torch 2.11 a 2.5.1.

### Ensemble

Se intentó un ensemble de votación suave entre LightGBM + XGBoost + CatBoost (exp-011,
012, 013). El resultado en Kaggle fue **0.80219** —peor que CatBoost solo (0.80687). El
ensemble diluyó la señal del modelo más fuerte en lugar de complementarla.

---

## Resultados en Kaggle

| Experimento | Modelo | Kaggle score |
|---|---|---|
| exp-013 | CatBoost + fs-004 | **0.80687** ← mejor |
| exp-017 | CatBoost + fs-012 | 0.80617 |
| ensemble 011+012+013 | LightGBM+XGBoost+CatBoost | 0.80219 |
| exp-014 | CatBoost + fs-005 | 0.79962 |

Curiosamente, exp-017 tiene mejor val accuracy (0.825 vs 0.822) pero peor score en Kaggle
(0.806 vs 0.807). Esto sugiere que las features de ruta (`Route_TE`) capturan algo real
en entrenamiento pero generalizan peor en test —un caso clásico de overfitting a la muestra
de validación.

---

## Análisis de errores

El modelo comete más errores en segmentos específicos:

| Segmento | Error rate |
|---|---|
| Destino PSO J318.5-22 | 33.3% |
| Pasajeros "Unknown" (sin destino) | 23.8% |
| Pasajeros activos (no-cryo) | 19.3% |
| Niños (Age < 13) | 21.3% |
| Pasajeros en CryoSleep | 17.4% |

**PSO J318.5-22** es el destino más raro (162 pasajeros en validación) y el más difícil.
Los **niños** tienen patrones de gasto distintos que el modelo no captura bien con las
features actuales.

---

## Lecciones aprendidas

**Más features no es mejor.** Varios feature sets con docenas de variables nuevas no mejoraron
al baseline. La señal adicional muchas veces viene acompañada de ruido o multicolinealidad.

**El dominio importa más que el algoritmo.** La imputación con reglas físicas (CryoSleep
implica gasto = 0) fue más valiosa que probar un arquitectura de red neuronal.

**Hay que separar val de Kaggle.** El modelo con mejor val accuracy no siempre es el mejor
en test público. Validar con múltiples seeds y estrategias de split reduce este riesgo.

**El ensemble no siempre ayuda.** Solo funciona cuando los modelos tienen errores
complementarios. Aquí los tres boosters se equivocaban en los mismos pasajeros.

**Target encoding requiere cuidado.** `TravelGroup_TE` fue el ejemplo más claro: métricas
perfectas en validación por leakage, inútil en producción.

---

## Estado actual y próximos pasos

- **21 experimentos** completados
- **Mejor score Kaggle: 0.80687** (top ~25% de la competencia en el momento de envío)
- **Mejor val accuracy: 0.825** (exp-017, CatBoost + fs-012)

El techo actual parece estar alrededor de **0.807 en Kaggle**. Los mejores scores públicos
de la competencia rondan **0.83-0.84**, por lo que hay ~2-3 puntos de margen.

Las líneas de trabajo más prometedoras para romper ese techo:

1. **Feature engineering para PSO J318.5-22** — el segmento con 33% de error, probablemente
   tratable con una feature binaria de destino raro combinada con target encoding específico.
2. **Pseudo-labeling** — usar las predicciones del modelo en test con alta confianza como
   etiquetas para reentrenamiento, una técnica común en los top submissions de Kaggle.
3. **Stacking** — en lugar de votación suave, usar las probabilidades de LightGBM, XGBoost
   y CatBoost como features de un meta-modelo de segundo nivel.

---

*Proyecto desarrollado en Python con scikit-learn, LightGBM, XGBoost, CatBoost, Optuna y
MLflow. El código completo sigue una arquitectura lista para producción: lógica en `src/`,
notebooks como orquestadores, experimentos versionados y reproducibles.*
