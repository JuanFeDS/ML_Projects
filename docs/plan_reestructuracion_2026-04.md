# Plan de Reestructuración — ML_Projects

**Fecha:** 2026-04-05  
**Objetivo:** Lograr armonía entre `scripts/`, `src/` y `run.py`, eliminar redundancias y dejar MLflow operativo desde cero.

---

## Diagnóstico

### Problemas identificados

| # | Problema | Archivos involucrados |
|---|----------|-----------------------|
| 1 | Dos puntos de entrada que hacen casi lo mismo | `run.py`, `scripts/run_pipeline.py` |
| 2 | `run.py` no expone los flags útiles de `run_pipeline.py` (`--skip-eda`, `--from-train`, `--predict-only`) | `run.py` |
| 3 | `include_ai=True` en `03_train.py` — IA no está implementada aún | `scripts/03_train.py` |
| 4 | `builder.py` (794 líneas) mezcla lógica de experimentos, tarjetas y reportes | `src/reports/builder.py` |
| 5 | Scalers y encoders sueltos en `models/` (no en `models/experiments/`) | `models/scaler_*.pkl`, `models/target_encoder_*.pkl` |
| 6 | `Projects/` directorio huérfano con un solo archivo | `Projects/` |
| 7 | MLflow configurado pero sin DB ni experimento inicializado | `src/models/tracking.py`, `src/config/settings.py` |
| 8 | `scripts/run_pipeline.py` es legacy — se puede eliminar | `scripts/run_pipeline.py` |

---

## Plan por Fases

### FASE 1 — Limpiar entrada y eliminar redundancias
**Meta:** Un solo punto de entrada (`run.py`) con todos los flags necesarios.

- [x] **1.1** Absorber los flags de `run_pipeline.py` en `run.py`  
  (`--skip-eda`, `--from-train`, `--predict-only`)
- [x] **1.2** Eliminar `scripts/run_pipeline.py`
- [x] **1.3** Quitar `include_ai=True` de `scripts/03_train.py` (pasar a `False`)

**Resultado esperado:**
```bash
python run.py --stage all
python run.py --stage train --feature-set fs-009_percentile_cabin
python run.py --skip-eda --feature-set fs-009_percentile_cabin
python run.py --predict-only
```

---

### FASE 2 — MLflow desde cero
**Meta:** MLflow funcional con UI y tracking persistente.

- [x] **2.1** Verificar que `mlflow` está instalado — creado `requirements.txt`
- [x] **2.2** Confirmar que `settings.py` apunta a `sqlite:///mlflow.db`
- [x] **2.3** `--init` integrado directamente en `run.py` (sin script separado)
- [x] **2.4** Agregar `mlflow.db` y `mlruns/` al `.gitignore`
- [x] **2.5** Documentar UI y comandos en `README.md`

---

### FASE 3 — Reorganizar artefactos de modelos
**Meta:** Todo artefacto en su carpeta correcta.

- [x] **3.1** Mover `models/scaler_fs-*.pkl` → `models/experiments/`
- [x] **3.2** Mover `models/target_encoder_fs-*.pkl` → `models/experiments/`
- [x] **3.3** Actualizar `settings.py`: `get_scaler_path()` y `get_target_encoder_path()`  
  apuntan a `models/experiments/`
- [x] **3.4** Eliminar directorio `Projects/`

---

### FASE 4 — Dividir `builder.py`
**Meta:** Separar responsabilidades del monolito de 794 líneas.

Estructura propuesta:
```
src/reports/
├── builder.py          ← solo ReportFactory (dispatcher de reportes)
├── experiment_log.py   ← append_experiment_log, get_next_exp_id, is_duplicate_experiment
├── model_cards.py      ← write_experiment_card, write_model_card
├── eda_reports.py      ← (ya existe, sin cambios)
├── feature_reports.py  ← (ya existe, sin cambios)
├── training_report.py  ← (ya existe, sin cambios)
└── prediction_report.py ← (ya existe, sin cambios)
```

- [x] **4.1** Extraer funciones de log/cards a `experiment_log.py` y `model_cards.py`
- [x] **4.2** `builder.py` queda con `MarkdownReport`, `HTMLReport`, `ReportFactory` (413 líneas vs 794)
- [x] **4.3** Actualizar imports en `training_pipeline.py` y `eda_reports.py`

---

### FASE 5 — Verificación end-to-end
**Meta:** El pipeline completo corre sin errores desde `run.py`.

- [ ] **5.1** `python run.py --init` — inicializa MLflow
- [ ] **5.2** `python run.py --stage eda` — genera reporte EDA
- [ ] **5.3** `python run.py --stage features --feature-set fs-001_baseline`
- [ ] **5.4** `python run.py --stage train --feature-set fs-001_baseline`
- [ ] **5.5** `python run.py --stage predict`
- [ ] **5.6** `python run.py --stage all` — pipeline completo
- [ ] **5.7** Verificar UI de MLflow: runs visibles con jerarquía parent → child

---

## Reglas de diseño (invariantes)

- `scripts/` = orquestadores delgados. Solo parsean args, llaman `src/`, imprimen estado.
- `src/` = toda la lógica reutilizable. Sin prints de progreso, sin argparse.
- `run.py` = único punto de entrada para el usuario.
- Notebooks = referencia y exploración. No se modifican.

---

---

### FASE 6 — Dividir `tracking_report.py` (969 líneas) — CRÍTICO
**Meta:** Separar las 4 responsabilidades del monolito más grande del repo.

Estructura propuesta:
```
src/models/
└── tracking_loader.py          ← extracción y parsing de datos MLflow

src/reports/experiments/
├── charts.py                   ← visualizaciones Plotly (leaderboard, progression, etc.)
├── cards_renderer.py           ← renderización HTML de tarjetas de experimento
└── tracking_report.py          ← orquestador delgado: llama loader → charts → renderer
```

- [ ] **6.1** Crear `src/models/tracking_loader.py` con `load_experiment_data()` y helpers de parsing
- [ ] **6.2** Crear `src/reports/experiments/charts.py` con las 6 funciones Plotly
- [ ] **6.3** Crear `src/reports/experiments/cards_renderer.py` con `_render_experiment_card()` y CSS
- [ ] **6.4** Reducir `tracking_report.py` a orquestador ≤100 líneas
- [ ] **6.5** Verificar que `build_tracking_report()` funciona end-to-end

---

### FASE 7 — Extraer assets de `builder.py` (834 líneas) — ALTO
**Meta:** CSS y JS fuera del código Python; `builder.py` queda como lógica pura.

Estructura propuesta:
```
src/reports/
├── assets.py                   ← constantes: _HTML_CSS, _HTML_JS, _LOGO_SVG
└── builder.py                  ← solo clases y factory (objetivo: <400 líneas)
```

- [ ] **7.1** Crear `src/reports/assets.py` y mover `_HTML_CSS`, `_HTML_JS`, `_LOGO_SVG`
- [ ] **7.2** Actualizar `builder.py` para importar desde `assets.py`
- [ ] **7.3** Verificar generación de reportes HTML sin regresiones

---

### FASE 8 — Centralizar duplicación en scripts — ALTO
**Meta:** Eliminar copias de `_preprocess()` y constantes hardcodeadas.

- [ ] **8.1** Crear `src/scripts/common.py` con `preprocess_native()`, `CAT_FEATURES`, `NUMERIC_FEATURES`
- [ ] **8.2** Actualizar `scripts/08_catboost_native.py` y `scripts/12_stacking.py` para importar desde `common.py`
- [ ] **8.3** Mover `_get_git_commit()` y `_create_git_tag()` de `training_pipeline.py` → `src/config/vcs.py`

---

### FASE 9 — Dividir `derived.py` por dominio (524 líneas) — MEDIO
**Meta:** Submódulos temáticos; `__init__.py` como aggregator.

Estructura propuesta:
```
src/features/engineering/
├── derived_group.py            ← group_spending, group_context, group_consistency
├── derived_interaction.py      ← solo_interaction, cryo_spending_interaction, spend_cluster
├── derived_demographic.py      ← child_route, structural_context
└── __init__.py                 ← re-exporta todo (sin cambios en callers)
```

- [ ] **9.1** Crear `derived_group.py`, `derived_interaction.py`, `derived_demographic.py`
- [ ] **9.2** Actualizar `__init__.py` para re-exportar desde los tres submódulos
- [ ] **9.3** Eliminar `derived.py` original

---

### FASE 10 — Refactorizar `training_pipeline.py` a OOP — MEDIO
**Meta:** `TrainingPipeline` con inyección de dependencias; eliminar hardcodeo de `MODELS`.

```python
class TrainingPipeline:
    def __init__(self, models: dict, feature_sets: dict, config: Settings): ...
    def evaluate(self, ...): ...
    def tune(self, ...): ...
    def build_ensemble(self, ...): ...
    def run(self, ...): ...
```

- [ ] **10.1** Crear clase `TrainingPipeline` en `src/pipelines/training_pipeline.py`
- [ ] **10.2** Extraer `_get_git_commit()` / `_create_git_tag()` (prerequisito: Fase 8.3)
- [ ] **10.3** Actualizar `scripts/03_train.py` para instanciar `TrainingPipeline`
- [ ] **10.4** Verificar pipeline end-to-end

---

## Estado

- [x] Fase 1
- [x] Fase 2
- [x] Fase 3
- [x] Fase 4
- [ ] Fase 5 — verificación end-to-end (pendiente desde antes)
- [ ] Fase 6 — dividir `tracking_report.py` (969 líneas) **← SIGUIENTE**
- [ ] Fase 7 — extraer assets de `builder.py`
- [ ] Fase 8 — centralizar duplicación en scripts
- [ ] Fase 9 — dividir `derived.py` por dominio
- [ ] Fase 10 — `TrainingPipeline` OOP con inyección de dependencias
