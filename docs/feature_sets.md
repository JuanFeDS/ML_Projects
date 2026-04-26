# Feature Sets — Spaceship Titanic

Un **feature set** es una versión concreta del dataset listo para entrenar un modelo.
Define qué columnas se crean, cómo se transforman y qué se descarta antes de que el modelo vea los datos.

Cada feature set es inmutable: una vez creado no se modifica. Si una idea no funciona, queda registrada de todas formas para no repetirla. Para experimentar con algo nuevo se crea una entrada nueva con el siguiente número.

El código de todos los feature sets vive en `src/features/feature_sets.py`. Los datos procesados se generan con:

```bash
python run.py --stage features --feature-set <nombre>
```

---

## fs-001_baseline

**Parent:** ninguno (es el punto de partida)
**Columnas de entrada al modelo:** 10 numéricas + 4 categóricas (OHE) ≈ 18 columnas tras encoding

Este es el feature set de referencia. Extrae las señales más directas que tiene el dataset y establece el piso de rendimiento que los siguientes feature sets deben superar.

### Qué hace

El dataset crudo tiene columnas como `Cabin` (formato `"B/0045/S"`) o `PassengerId` (formato `"0001_01"`) que no se pueden usar directamente. Este feature set las descompone y crea variables útiles:

- **`Cabin` → `Deck`, `CabinNumber`, `Side`**: la cubierta y el lado del barco son estadísticamente significativos para predecir si el pasajero fue transportado (chi²=392 y chi²=91 respectivamente).
- **`PassengerId` → `TravelGroup`, `GroupSize`**: el tamaño del grupo de viaje muestra un patrón no lineal con el target — los grupos medianos (3-6 personas) tienen más probabilidad de ser transportados.
- **Spending agregado → `TotalSpending_Log`, `SpendingCategories`**: sumar los 5 servicios de gasto y aplicar logaritmo eleva la correlación con el target de r=−0.20 a r=−0.47. `SpendingCategories` cuenta cuántos servicios distintos usó el pasajero.
- **`Age` → `AgeCategory`**: categorizar la edad en rangos (Child, Teen, YoungAdult, Adult, Senior) captura la relación no lineal que tiene con el target mejor que el valor numérico crudo.

### Columnas descartadas
`PassengerId`, `Name`, `Cabin`, `TravelGroup`, `CryoSleep`, `VIP`, `Side`, `TotalSpending` — o bien fueron descompuestas en otras columnas, o tienen poca señal, o introducen problemas de encoding.

---

## fs-002_cryo_interactions

**Parent:** fs-001_baseline
**Columnas nuevas:** +5 numéricas, +1 categórica (`Route`)

Intenta capturar la interacción entre el estado de CryoSleep del pasajero, el comportamiento de su grupo y su posición en el barco.

### Qué añade sobre fs-001

- **`Route`**: concatena `HomePlanet` y `Destination` (ej. `"Europa_55 Cancri e"`). La hipótesis es que ciertas rutas tienen tasas de transporte distintas más allá de lo que capturan las dos columnas por separado.
- **`GroupCryoSleepRate`**: qué porcentaje de los compañeros de viaje del pasajero está en CryoSleep. Un pasajero activo rodeado de gente dormida puede ser una señal.
- **`CryoSleepViolation`**: flag binario — pasajero activo en un grupo donde la mayoría está en CryoSleep.
- **`LuxurySpendingRatio`**: qué fracción del gasto total va a servicios de lujo (Spa + VRDeck). Diferencia perfiles de consumo.
- **`CabinNumber_DeckPercentile`**: posición relativa de la cabina dentro de su cubierta (0 = más cercana al frente, 1 = al fondo).
- **`GroupSpendingMean`**: media de gasto de todos los compañeros de viaje del pasajero.

**Resultado histórico:** val_accuracy=0.8156, no superó fs-001 (0.8227). Las interacciones con CryoSleep añadieron ruido más que señal en este caso.

---

## fs-003_solo_interactions

**Parent:** fs-001_baseline
**Columnas nuevas:** +3 numéricas

Apuesta por features simples y directamente interpretables, sin riesgo de multicolinealidad ni overfitting por complejidad.

### Qué añade sobre fs-001

- **`IsAlone`**: 1 si el pasajero viaja solo (`GroupSize == 1`), 0 si no. Los pasajeros solos tienen un patrón de transporte distinto a los que viajan en grupo.
- **`IsChild`**: 1 si `Age < 13`. Los niños no tienen acceso a servicios de gasto (spending siempre 0), lo que hace que su perfil sea muy diferente al resto.
- **`SpendingIntensity`**: `TotalSpending / (SpendingCategories + 1)`. Mide cuánto gasta el pasajero *por servicio que usa*. Un pasajero que gasta mucho en un solo servicio tiene un perfil distinto a uno que distribuye el gasto entre varios.

---

## fs-004_target_encoding

**Parent:** fs-001_baseline
**Columnas modificadas:** `Deck` y `HomePlanet` reemplazadas por Target Encoding

En fs-001, `Deck` (8 categorías) y `HomePlanet` (3 categorías) se codifican con One-Hot Encoding, generando columnas binarias. Este feature set las reemplaza por una sola columna numérica cada una, donde el valor es la **tasa media de transporte** de esa categoría en el dataset de entrenamiento.

### Por qué puede mejorar

OHE trata todas las categorías como igualmente distintas entre sí. Target Encoding introduce una señal ordinal real: si la Cubierta B tiene 80% de transportados y la G tiene 30%, el modelo recibe directamente esa diferencia en lugar de tener que aprenderla a partir de columnas binarias.

El encoder se serializa en `models/experiments/target_encoder_fs-004_target_encoding.pkl` para poder aplicar la misma transformación al dataset de test sin recalcular.

**Riesgo:** target leakage si no se aplica correctamente (el encoder debe ajustarse solo sobre los datos de entrenamiento, nunca sobre validación o test).

---

## fs-005_structural_context

**Parent:** fs-001_baseline
**Columnas nuevas:** +6 numéricas + 1 Target Encoding

El conjunto más ambicioso de features derivadas. Añade contexto estructural del pasajero dentro de su grupo, su deck y su patrón de consumo.

### Qué añade sobre fs-001

- **`SpendingEntropy`**: mide qué tan distribuido está el gasto entre los 5 servicios usando entropía de Shannon. Entropía alta = gasta un poco en todo; baja = concentra en uno o dos servicios.
- **`GroupSpendingZScore`**: cuántas desviaciones estándar se aleja el gasto del pasajero respecto a la media de su grupo. Detecta pasajeros anómalos dentro de su propio grupo.
- **`CabinNeighborhoodDensity`**: cuántos pasajeros hay en las 50 cabinas contiguas del mismo deck. Mide si el pasajero está en una zona densa o aislada del barco.
- **`FamilySizeFromName`**: cuenta cuántos pasajeros comparten el mismo apellido. Aproxima el tamaño familiar más allá del grupo de viaje formal.
- **`GroupCryoAlignment`**: qué tan alineado está el pasajero con el consenso de CryoSleep de su grupo. Un pasajero que duerme cuando todos están activos (o viceversa) puede ser señal.
- **`GroupAgeDispersion`**: desviación estándar de la edad dentro del grupo. Grupos con rango etario amplio (mezcla de niños y adultos) vs. grupos homogéneos.
- **`SpendingCategoryProfile_TE`**: Target Encoding del perfil de servicios usados (combinación de qué servicios usó el pasajero).

---

## fs-006_group_imputation

**Parent:** fs-001_baseline
**Columnas:** mismas que fs-001

Mismo espacio de features que fs-001, pero con una estrategia de imputación diferente para los valores de gasto nulos.

### El problema que resuelve

En fs-001, los nulos en columnas de gasto (`RoomService`, `Spa`, etc.) se imputan con 0. Eso es correcto para pasajeros en CryoSleep (que físicamente no pueden gastar), pero para pasajeros activos con gasto nulo puede ser incorrecto — simplemente no tenemos el dato.

Este feature set imputa esos nulos con la **mediana del gasto del grupo de viaje** del pasajero. La hipótesis es que los compañeros de viaje tienen perfiles de consumo similares, así que la mediana del grupo es una mejor estimación que cero.

**Diferencia clave:** el orden del pipeline cambia: `extraer features → rellenar categóricas → imputar gasto por grupo → calcular TotalSpending_Log`.

---

## fs-007_domain_rules

**Parent:** fs-001_baseline
**Columnas nuevas:** +1 numérica (`TravelGroup_TE`)

Antes de hacer cualquier cálculo, aplica 6 reglas físicas del universo del dataset para imputar columnas clave. La idea es que muchos nulos no son aleatorios — tienen una respuesta correcta que se puede deducir por lógica.

### Las 6 reglas

| Regla | Lógica |
|---|---|
| HomePlanet por grupo | Si todos los compañeros de viaje tienen el mismo planeta de origen, el pasajero también |
| Deck A/B/C → Europa | Históricamente, esas cubiertas son casi exclusivamente de pasajeros de Europa |
| Deck G → Earth | Igual, Deck G es predominantemente de Tierra |
| Deck/Side por grupo | Dentro de un grupo, todos ocupan la misma cubierta y lado |
| CryoSleep=True → spending=0 | Un pasajero hibernando no puede gastar en servicios |
| spending>0 → CryoSleep=False | Si gastó algo, no estaba durmiendo |
| Age≤12 → spending=0 | Los menores no tienen acceso a los servicios del barco |

Además añade `TravelGroup_TE`: la tasa de transporte media del grupo de viaje (Target Encoding sobre `TravelGroup`).

---

## fs-008_domain_rules_only

**Parent:** fs-001_baseline
**Columnas:** mismas que fs-001

Idéntico a fs-007 en cuanto al pipeline de imputación por reglas, pero **sin** `TravelGroup_TE`. Elimina el potencial leakage que introduce encodear el grupo directamente con el target, manteniendo el beneficio de las imputaciones físicas.

Es el "fs-007 limpio" — mismo conocimiento del dominio, sin la feature potencialmente problemática.

---

## fs-009_percentile_cabin

**Parent:** fs-008_domain_rules_only
**Columnas modificadas:** `CabinNumber` reemplazado por `CabinNumber_DeckPercentile`

### El problema que resuelve

Un análisis de **adversarial validation** (entrenar un clasificador para distinguir train de test) dio AUC=0.79 — lo ideal es 0.50. Eso significa que train y test son distinguibles, lo que indica que alguna feature tiene una distribución diferente entre los dos splits.

La feature más responsable de ese shift resultó ser `CabinNumber`: los números de cabina en el dataset de test están en rangos distintos a los de train (el barco tiene secciones distintas en cada split). Un modelo que aprende "cabina 300 = más probable de ser transportado" en train no puede generalizar eso a test.

**La solución:** reemplazar el número absoluto de cabina por su **percentil relativo dentro del deck**. En lugar de "cabina número 300 del Deck B", el modelo recibe "cabina en el 40% más bajo del Deck B". Eso es invariante al rango absoluto y generaliza mejor.

---

## Comparativa rápida

| Feature set | Base | Cambio principal | Resultado conocido |
|---|---|---|---|
| fs-001_baseline | — | Features base del dataset | 0.8227 val_accuracy |
| fs-002_cryo_interactions | fs-001 | Interacciones CryoSleep + grupo | 0.8156 (no mejoró) |
| fs-003_solo_interactions | fs-001 | IsAlone, IsChild, SpendingIntensity | — |
| fs-004_target_encoding | fs-001 | Deck y HomePlanet como Target Encoding | — |
| fs-005_structural_context | fs-001 | 7 features contextuales de grupo y deck | — |
| fs-006_group_imputation | fs-001 | Imputación de gasto por mediana del grupo | — |
| fs-007_domain_rules | fs-001 | 6 reglas físicas + TravelGroup_TE | — |
| fs-008_domain_rules_only | fs-001 | 6 reglas físicas sin TE de grupo | — |
| fs-009_percentile_cabin | fs-008 | CabinNumber → percentil relativo por deck | — |
