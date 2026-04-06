# Spaceship Titanic — Proyecto ML

Reto de clasificación de Kaggle: predecir qué pasajeros fueron transportados a una dimensión
alternativa. El proyecto está estructurado como un pipeline de producción con trazabilidad
completa de experimentos via MLflow y generación automática de reportes en cada etapa.

---

## Arquitectura

El proyecto separa responsabilidades en tres capas que trabajan en conjunto:

- **`run.py`** es el único punto de entrada. Orquesta el pipeline llamando a los scripts por etapa.
- **`scripts/`** contiene un script por etapa del pipeline. Cada uno es un orquestador delgado: parsea argumentos, llama a `src/` y reporta estado.
- **`src/`** contiene toda la lógica reutilizable. No tiene argumentos CLI ni prints de progreso — solo funciones y clases.

Los notebooks en `notebooks/exploratory/` son el espacio de exploración donde se toman las decisiones analíticas. La lógica que resulta de esa exploración termina implementada en `src/`.

---

## Pipeline

El flujo completo tiene cuatro etapas encadenadas:

```
EDA  →  Feature Engineering  →  Entrenamiento  →  Predicción
```

Cada etapa genera reportes automáticos en `reports/` (`.md` + `.html`) y registra métricas en MLflow. El entrenamiento además actualiza `docs/model/` con una tarjeta por experimento y el log acumulado de todos los runs.

### Entrenamiento iterativo

La etapa de entrenamiento está diseñada para ser iterativa. Cada run:

1. Evalúa todos los modelos del catálogo con cross-validation.
2. Tunea el mejor con Optuna (25 trials).
3. Construye un Stacking y un Mixture of Experts sobre el tuneado.
4. Elige el ganador por `val_accuracy` con umbral optimizado.
5. Promueve a `models/production/` solo si supera al modelo actual.
6. Registra el experimento en `docs/model/experimentation_log.md` y genera una model card en `docs/model/cards/`.

Los artefactos de todos los experimentos quedan en `models/experiments/` para poder reproducir cualquier run anterior.

---

## Estructura de directorios

```
ML_Projects/
├── data/
│   ├── raw/            ← datos originales (no se commitean)
│   ├── features/       ← datasets transformados por feature set
│   └── submissions/    ← archivos de submission para Kaggle
├── docs/
│   ├── data/           ← diccionario, fuentes y calidad de datos
│   └── model/
│       ├── cards/      ← una tarjeta .md por experimento
│       ├── model_card.md          ← card del modelo en producción
│       └── experimentation_log.md ← historial acumulado de runs
├── models/
│   ├── experiments/    ← artefactos de cada run (.pkl)
│   └── production/     ← modelo promovido (best_model.pkl + metadata)
├── notebooks/exploratory/   ← NB01 → NB02 → NB03 → NB04 → NB05
├── reports/            ← reportes HTML/MD generados por el pipeline
├── scripts/            ← orquestadores por etapa (01_eda … 04_predict)
├── src/
│   ├── config/         ← settings.py, logger.py
│   ├── data/           ← preprocessing, EDA, quality checks
│   ├── features/       ← engineering, feature sets, constantes
│   ├── models/         ← catálogo, entrenamiento, tracking MLflow, MoE
│   ├── pipelines/      ← orquestación de cada etapa
│   └── reports/        ← builders de reportes MD/HTML
└── run.py              ← punto de entrada CLI
```

---

## Comandos

```bash
# Pipeline completo
python run.py

# Etapa individual
python run.py --stage eda
python run.py --stage features --feature-set fs-001_baseline
python run.py --stage train    --feature-set fs-001_baseline
python run.py --stage predict

# Pipeline parcial
python run.py --skip-eda    --feature-set fs-001_baseline
python run.py --from-train  --feature-set fs-001_baseline
python run.py --predict-only
```

---

## Seguimiento de experimentos

MLflow registra cada run con sus métricas, parámetros y jerarquía parent → child.

```bash
# Inicializar (primera vez o entorno nuevo)
python run.py --init

# Ver UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```
