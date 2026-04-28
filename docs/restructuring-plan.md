# Plan de Reestructuración de Carpetas

## Problemas identificados

1. **`src/scripts/`** — nombre confuso con `scripts/` en la raíz. `common.py` es preprocesamiento, no un script.
2. **`src/models/`** — 14 archivos planos con 5+ responsabilidades mezcladas.
3. **`src/data/`** — casi vacío (solo `transductive.py`). `transductive.py` debería estar en `src/features/`.
4. **`models/` raíz vs `src/models/`** — overlap de nombres entre artefactos y código.

---

## Estructura propuesta

### `src/models/` — dividir en 4 subfolders, ningún archivo suelto

```
src/models/
├── training/
│   ├── catalogue.py        # ← src/models/catalogue.py
│   ├── tuning.py           # ← src/models/tuning.py
│   ├── pipeline_utils.py   # ← src/models/pipeline_utils.py
│   ├── catboost_native.py  # ← src/models/catboost_native.py
│   ├── tabnet_training.py  # ← src/models/tabnet_training.py
│   └── tabnet_wrapper.py   # ← src/models/tabnet_wrapper.py
├── evaluation/
│   ├── evaluation.py       # ← src/models/evaluation.py
│   ├── errors.py           # ← src/models/errors.py
│   └── stacking_oof.py     # ← src/models/stacking_oof.py
├── ensembles/
│   ├── ensembles.py        # ← src/models/ensembles.py
│   └── moe.py              # ← src/models/moe.py
└── inference/
    ├── predict.py          # ← src/models/predict.py
    └── artifact_store.py   # ← src/models/artifact_store.py
```

El `src/models/__init__.py` sigue re-exportando todo igual — los scripts externos no ven el cambio.

### `src/scripts/` → renombrar a `src/preprocessing/`

```
src/preprocessing/
└── common.py    # ← src/scripts/common.py (preprocess_native_*)
```

Actualizar imports en: `scripts/08`, `09`, `10`, `11`, `12`, `13`.

### `src/data/` — mover `transductive.py`

- `transductive.py` → `src/features/transductive.py` (es generación de features, no carga de datos)
- Si no hay más contenido para `src/data/`, eliminar el módulo.

### `models/` raíz → renombrar a `artifacts/`

```
artifacts/
├── experiments/   # ← models/experiments/
└── production/    # ← models/production/
```

Actualizar `src/config/settings.py` — todas las rutas `EXPERIMENTS_DIR`, `MODEL_PATH`, etc.

---

## Orden de ejecución sugerido

1. Reestructurar `src/models/` (mayor impacto, más imports que actualizar)
2. Renombrar `src/scripts/` → `src/preprocessing/`
3. Mover `transductive.py` y limpiar `src/data/`
4. Renombrar `models/` → `artifacts/` y actualizar `settings.py`
5. Correr `pytest` para verificar que no hay imports rotos

---

---

## Imágenes pendientes para el artículo del portafolio

Referenciadas en `docs/spaceship-titanic.mdx` pero aún no generadas.
Ruta esperada por el portafolio: `public/images/projects/spaceship-titanic/`

| Archivo | Descripción | Cómo generarla |
|---|---|---|
| `banner.png` | Imagen principal del artículo | Diseño manual / generada con IA |
| `dataset-overview.png` | Distribución de variables clave y balance del target | Script EDA — plots de `src/reports/eda/` |
| `architecture-diagram.png` | Diagrama de las 3 capas del pipeline | Diagrama manual (draw.io / Excalidraw) |
| `feature-engineering-layers.png` | Mapa de los 3 niveles de features | Diagrama manual o gráfico generado |
| `model-comparison.png` | Métricas de los 6 modelos base antes del ensemble | Extraer de MLflow o regenerar con `evaluate_models` |
| `ensemble-diagram.png` | Diagrama de las 3 estrategias de ensemble | Diagrama manual |
| `mlflow-ui.png` | Captura de la UI de MLflow con los runs del proyecto | Screenshot de `mlflow ui` corriendo localmente |

Las que dependen de ejecución del pipeline (`dataset-overview`, `model-comparison`) deberían generarse y exportarse como parte del flujo normal — idealmente desde `scripts/01_eda.py` y `scripts/03_train.py`. Las diagramas arquitecturales son manuales.

---

## Archivos con mayor cantidad de imports a actualizar

- `src/pipelines/training_pipeline.py` — importa de `src/models/` extensamente
- `scripts/08_catboost_native.py` al `13_groupkfold_train.py` — importan de `src/scripts/common`
- `src/models/__init__.py` — re-export central, actualizar paths internos
- `src/config/settings.py` — rutas de `models/` → `artifacts/`
