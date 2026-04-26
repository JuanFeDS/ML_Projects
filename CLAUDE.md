# CLAUDE.md

Este archivo provee orientación a Claude Code (claude.ai/code) para trabajar con el código de este repositorio.

## Descripción del Repositorio

Proyecto ML para el reto de clasificación Spaceship Titanic (Kaggle). El proyecto sigue una
estructura lista para producción: `src/` contiene toda la lógica reutilizable, `notebooks/`
orquesta el flujo de análisis, y directorios separados gestionan datos, modelos, docs y reportes.

## Configuración del Entorno

Activar el entorno virtual antes de ejecutar cualquier cosa:

```bash
source .venv/Scripts/activate   # Windows Git Bash
```

Instalar dependencias (gestionado con uv + pyproject.toml):

```bash
uv pip install -e .
```

## Ejecutar Notebooks

Lanzar Jupyter desde la raíz del repositorio:

```bash
jupyter notebook
# o
jupyter lab
```

## Flujo de Análisis: Cadena de Notebooks

Los notebooks exploratorios forman una cadena narrativa. Cada uno parte de donde termina el anterior.
**No repetir análisis** — referenciar los hallazgos del notebook previo con una tabla o bloque de contexto.

```
NB01 — Exploración Inicial
  ↓  (estructura del dataset, tipos, nulos, duplicados, balance del target)
NB02 — Análisis de Variables vs Target
  ↓  (poder discriminativo de cada feature, decisiones documentadas)
NB03 — Feature Engineering
  ↓  (ejecuta las decisiones de NB02; lógica vive en src/features/engineering.py)
NB04 — Entrenamiento de Modelos
       (carga el dataset de NB03; entrena, compara, evalúa, guarda el modelo)
NB05 — Predicciones
       (carga el modelo de producción; genera el archivo de submission)
```

### Convención de Referencias Cruzadas

Al inicio de NB02, NB03, NB04 y NB05 debe haber un bloque de contexto que resume
qué viene del notebook anterior. Ejemplo para NB03:

```markdown
**Fundamentos — decisiones de NB02:**

| Feature  | Acción        | Justificación          |
|----------|---------------|------------------------|
| `Cabin`  | Extraer Deck  | chi²=392.3 (p<0.001)   |
| `VIP`    | Descartar     | corr=-0.037, sin señal |
```

## Arquitectura: Notebooks como Orquestadores, src/ como Lógica

**Regla:** Toda la lógica de transformación vive en `src/`. Los notebooks llaman funciones y visualizan resultados.
Esto elimina duplicación de código y hace el proyecto testeable.

```
src/features/engineering.py   ← funciones de feature engineering
src/data/preprocessing.py     ← funciones de limpieza y preprocesamiento
src/data/eda.py               ← funciones de análisis exploratorio
```

En los notebooks:

```python
import sys
sys.path.insert(0, '../../')  # raíz del proyecto

from src.features.engineering import extract_cabin_features, build_feature_set
```

Para generar reportes desde notebooks usar **papermill** + **nbconvert**:

```bash
papermill 01.Initial_exploration.ipynb output/report_01.ipynb -p dataset_path data/raw/train.csv
jupyter nbconvert output/report_01.ipynb --to html
```

## Estructura de Directorios

```
ML_Projects/
├── data/
│   ├── raw/          ← datos originales (no modificar, no commitear)
│   ├── processed/    ← datasets transformados por NB03
│   ├── features/     ← conjuntos de features generados
│   └── submissions/  ← archivos de submission para Kaggle
├── docs/
│   ├── data/         ← data_dictionary.md, data_sources.md, data_quality.md
│   ├── model/
│   │   └── cards/    ← una tarjeta markdown por experimento
│   ├── templates/    ← eda_template.md
│   ├── evaluacion.md ← notas de evaluación del proyecto
│   └── notas.md      ← notas de trabajo
├── logs/             ← logs de la aplicación
├── models/
│   ├── experiments/  ← artefactos de experimentos (.pkl, .json)
│   └── production/   ← modelo final promovido a producción (.pkl, .json)
├── notebooks/
│   └── exploratory/
│       ├── 01.Initial_exploration.ipynb
│       ├── 02.Analisis_Target.ipynb
│       ├── 03.feature_engineering.ipynb
│       ├── 04.Model_Training.ipynb
│       └── 05.Predictions.ipynb
├── reports/          ← reportes HTML/MD generados por scripts/
├── scripts/          ← scripts ejecutables por etapa del pipeline
├── src/
│   ├── config/       ← settings.py (variables via .env), logger.py
│   ├── data/         ← preprocessing.py, eda.py, quality_checks.py
│   ├── features/     ← engineering.py, constants.py, feature_sets.py
│   ├── models/       ← catalogue.py, training.py, predict.py, tracking.py, moe.py
│   ├── pipelines/    ← orquestación end-to-end (data_pipeline.py)
│   ├── reports/      ← generación de plots y reportes (builder.py, *_plots.py)
│   └── api/          ← FastAPI (main.py, models.py)
├── venv/             ← entorno virtual compartido
├── run.py            ← punto de entrada CLI
├── CLAUDE.md         ← este archivo
└── README.md
```

## Seguimiento de Experimentos (MLflow)

El proyecto usa MLflow para el seguimiento de experimentos. La configuración reside en `src/config/settings.py`.

Comandos útiles:

- **Ver UI**: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (ejecutar en la raíz del proyecto)
- **Tracking manual**: usar el context manager `mlrun` de `src.models.tracking`.

`src/models/training.py` integra MLflow automáticamente en:

- `evaluate_models`: crea un run por cada modelo comparado.
- `tune_model`: registra cada trial de Optuna como un run anidado.

## Estilo de Código

- **Formato**: black (line-length 88) + isort (perfil black)
- **Tipos**: mypy en modo estricto
- **Linting**: pylint, puntuación mínima 8.0
- **Docstrings**: estilo Google con secciones Args, Returns, Raises
- **Logging**: formato lazy con `%` — `logger.info("msg %s", val)`

Ejecutar verificaciones:

```bash
black src/
isort src/
mypy src/
pylint src/
pytest
```

## Patrones de Código

### Feature engineering: una función por transformación

```python
def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre descriptivo. Docstring con justificación estadística."""
    df_copy = df.copy()  # nunca mutar el input
    # ... transformación ...
    return df_copy
```

### Pipeline completo: `build_feature_set`

```python
def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Orquesta todas las transformaciones en orden."""
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    # ...
    return df_out
```

### Nulos: estrategia explícita con justificación en el código

```python
df['col_cat'].fillna('Unknown', inplace=True)  # Preservar como categoría
df['col_num'].fillna(0, inplace=True)           # Ausencia = cero
df.dropna(subset=['Age'], inplace=True)         # Evitar sesgo en categorización
```

## Agentes y Skills

- Para commits, usar **siempre el skill `/make-commits`** (Skill tool), nunca el subagente `make-commits` (Agent tool). El skill corre en la conversación principal y no consume tokens extra de spawning.

## Convenciones de Commits

Este repositorio usa Conventional Commits adaptado a ciencia de datos:

| Tipo        | Emoji | Cuándo usar                                         |
|-------------|-------|-----------------------------------------------------|
| `feat:`     | ✨    | Nuevo modelo, función o script                      |
| `fix:`      | 🐛    | Corrección de error o lógica                        |
| `data:`     | 🗃️    | Cambios en datos (limpieza, fuente, actualizaciones)|
| `sql:`      | 🗄️    | Cambios en consultas o scripts SQL                  |
| `eda:`      | 📊    | Trabajo de análisis exploratorio                    |
| `model:`    | 🤖    | Entrenamiento, evaluación o mejoras de modelos      |
| `test:`     | ✅    | Adición o modificación de tests                     |
| `docs:`     | 📝    | Documentación y explicaciones en notebooks          |
| `viz:`      | 📈    | Visualizaciones nuevas o ajustadas                  |
| `refactor:` | ♻️    | Reestructuración de código sin cambiar funcionalidad|
| `chore:`    | 🔧    | Tareas menores y actualizaciones técnicas           |
| `env:`      | 📦    | Cambios en dependencias o entorno                   |
| `ci:`       | ⚙️    | Cambios en CI/CD o scripts de automatización        |

## Librerías Clave

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `xgboost`,
`catboost`, `optuna`, `mlflow`, `fastapi`, `uvicorn`, `pydantic`, `oracledb`, `python-dotenv`.

## Reglas de .gitignore

Los archivos CSV, TXT y XLSX están excluidos de git. Los datos crudos deben almacenarse
localmente pero no commitearse. Los artefactos de modelos (`.pkl`) y checkpoints de
notebooks también están excluidos.
