# CLAUDE.md — Template de Proyecto ML

Este archivo describe la metodología, la arquitectura de notebooks y los patrones de código
que definen este template. Se aplica a cualquier proyecto derivado de esta estructura.

---

## Flujo de análisis: cadena de notebooks

Los notebooks exploratorios forman una cadena narrativa. Cada uno parte de donde termina el anterior.
**No repetir análisis** — referenciar los hallazgos del notebook previo con una tabla o bloque de contexto.

```
NB01 — Exploración Inicial
  ↓  (estructura del dataset, tipos, nulos, duplicados, balance del target)
NB02 — Análisis de Variables vs Target
  ↓  (poder discriminativo de cada feature, decisiones documentadas)
NB03 — Feature Engineering
  ↓  (ejecuta las decisiones de NB02; lógica vive en src/features/engineering.py)
NB04 — Model Training
       (carga el dataset de NB03; entrena, compara, evalúa, guarda el modelo)
```

### Convención de referencias cruzadas

Al inicio de NB02, NB03 y NB04 debe haber un bloque de contexto que resume
qué viene del notebook anterior. Ejemplo para NB03:

```markdown
**Fundamentos — decisiones de NB02:**

| Feature     | Acción        | Justificación               |
|-------------|---------------|-----------------------------|
| `Cabin`     | Extraer Deck  | chi²=392.3 (p<0.001)        |
| `VIP`       | Descartar     | corr=-0.037, sin señal      |
```

---

## Arquitectura: notebooks como orquestadores, src/ como lógica

**Regla:** La lógica de transformación vive en `src/`. Los notebooks llaman funciones y visualizan.
Esto elimina duplicación de código y hace el proyecto testeable.

```
src/features/engineering.py   ← funciones de feature engineering
src/data/preprocessing.py     ← funciones de limpieza y preprocesamiento
src/data/eda.py               ← funciones de análisis exploratorio
```

En los notebooks:
```python
import sys
sys.path.insert(0, '../../')   # raíz del proyecto

from src.features.engineering import extract_cabin_features, build_feature_set
```

Para generar reportes desde notebooks: usar **papermill** + **nbconvert**.
```bash
papermill 01.Initial_exploration.ipynb output/report_01.ipynb -p dataset_path data/raw/train.csv
jupyter nbconvert output/report_01.ipynb --to html
```

---

## Estructura de directorios

```
proyecto/
├── data/
│   ├── raw/          ← datos originales (no modificar, no commitear)
│   └── processed/    ← datasets transformados por NB03
├── docs/
│   ├── data/         ← diccionario de datos, fuentes, calidad
│   ├── model/        ← model card, log de experimentos
│   └── templates/    ← eda_template.md
├── models/
│   ├── experiments/  ← modelos de variantes y experimentos (.pkl, .json)
│   └── production/   ← modelo final promovido a producción (.pkl, .json)
├── notebooks/
│   └── exploratory/
│       ├── 01.Initial_exploration.ipynb
│       ├── 02.Analisis_Target.ipynb
│       ├── 03.feature_engineering.ipynb
│       ├── 04.Model_Training.ipynb
│       └── 05.Predictions.ipynb
├── reports/          ← reportes HTML/MD generados (salida de scripts/)
├── scripts/          ← scripts ejecutables por etapa (01_eda.py … run_pipeline.py)
├── src/
│   ├── config/       ← settings.py (env vars), logger.py
│   ├── data/         ← preprocessing.py, eda.py, quality_checks.py
│   ├── features/     ← engineering.py, constants.py  ← LÓGICA PRINCIPAL
│   ├── models/       ← catalogue.py, training.py, predict.py
│   ├── pipelines/    ← orquestación end-to-end (data_pipeline.py)
│   ├── reports/      ← generación de plots y reportes (builder.py, *_plots.py)
│   └── api/          ← FastAPI (main.py, models.py)
├── run.py            ← CLI entry point
├── CLAUDE.md         ← este archivo
└── README.md
```

---

## Patrones de código

### Feature engineering: función por transformación
```python
def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre descriptivo. Docstring con justificación estadística."""
    df_copy = df.copy()  # nunca mutar el input
    # ... transformación ...
    return df_copy
```

### Pipeline completo: función `build_feature_set`
```python
def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Orquesta todas las transformaciones en orden."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    # ...
    return df_out
```

### Nulos: estrategia explícita en código
```python
# Siempre comentar la justificación de la estrategia
df['col_cat'].fillna('Unknown', inplace=True)  # Preservar como categoría
df['col_num'].fillna(0, inplace=True)           # Ausencia = cero
df.dropna(subset=['Age'], inplace=True)         # Evitar sesgo en categorización
```

---

## Qué documenta NB02 (Análisis vs Target)

Cada variable debe tener:
- **Categóricas:** tasa del target por categoría + chi-square test (chi², p-valor)
- **Numéricas:** distribución por clase + Mann-Whitney U test (p-valor) + correlación de Pearson
- **Features derivadas:** validar que la transformación propuesta mejora la señal antes de construirla

Al final: tabla de decisiones con columnas `MANTENER | DESCARTAR | TRANSFORMAR`.

---

## Modelo de evaluación (NB04)

1. Baseline con `DummyClassifier` → piso mínimo a superar
2. Comparación con CV 5-fold → `LogisticRegression`, `RandomForest`, `GradientBoosting`
3. GridSearchCV sobre el ganador
4. Evaluación final: accuracy, ROC-AUC, confusion matrix, classification report
5. Feature importance → verificar coherencia con NB02
6. Guardar: `models/best_model.pkl` + `models/model_metadata.json`

---

## Usar este template en un nuevo proyecto

1. Copiar la estructura de directorios
2. Reemplazar datos en `data/raw/`
3. Adaptar las funciones de `src/features/engineering.py` al dominio del nuevo problema
4. Ejecutar los notebooks en orden (NB01 → NB02 → NB03 → NB04)
5. Las decisiones de NB02 guían NB03; NB03 alimenta NB04

**No adaptar los notebooks antes de tener los datos.** El orden natural es:
explorar primero → decidir → construir → modelar.
