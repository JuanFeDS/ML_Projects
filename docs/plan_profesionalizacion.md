# Plan de Profesionalización del Repositorio

Fecha: 2026-04-26  
Objetivo: Llevar el repo a estándares de producción para portafolio técnico.  
Audiencia: Recruiters técnicos y data scientists.

---

## Criterios acordados

- **Idioma**: español en toda la documentación
- **Tests**: cobertura > 80% con pytest
- **Linting**: pylint ≥ 90, black
- **CI/CD**: GitHub Actions (black, pylint, pytest --cov)
- **Scripts experimentales**: se conservan como parte del proceso visible
- **Archivos de trabajo**: viven en `ignore/` (gitignoreado)
- **Notebooks**: se ejecutan y se guarda el output — diferido a Fase 7
- **README**: técnico, orientado a arquitectura, sin sección de replicación paso a paso

---

## Fases

### Fase 1 — Limpieza del repo ✓
- [x] Crear carpeta `ignore/`
- [x] Mover `docs/notas.md`, `docs/evaluacion.md`, `docs/plan_reestructuracion.md`, `docs/estado_experimentos.md` a `ignore/`
- [x] Mover session files `2026-04-*.md` a `ignore/`
- [x] Revisar y completar `.gitignore` (eliminado `*.md` global, añadido `ignore/`)
- [x] Modularidad: split `src/models/training.py` → `evaluation.py`, `tuning.py`, `ensembles.py`, `errors.py`, `pipeline_utils.py`
- [x] `run_training_pipeline()` (God function, 443 líneas) → clase `TrainingPipeline` con 10 métodos
- [x] Eliminado `_create_git_tag` privado importado desde training_pipeline (bug) → `src.config.vcs`
- [x] Helpers duplicados en `07_segmented_train.py` eliminados → `pipeline_utils.py`

### Fase 2 — Calidad de código
- [ ] Pasar `black` sobre `src/` y `scripts/`
- [ ] Alcanzar `pylint ≥ 90` en todo `src/`
- [ ] Revisar docstrings (Google style, español, consistentes)
- [ ] Limpiar TODOs y comentarios muertos

### Fase 3 — Tests
- [ ] Auditar cobertura actual con `pytest --cov`
- [ ] Escribir tests hasta superar 80% en `src/`

### Fase 4 — FastAPI
- [ ] Crear `src/api/` con endpoints: `GET /health`, `POST /predict`
- [ ] Modelos Pydantic para input/output
- [ ] Cargar `models/production/best_model.pkl`
- [ ] Tests del API incluidos

### Fase 5 — CI/CD
- [ ] Workflow GitHub Actions: `black --check`, `pylint`, `pytest --cov`
- [ ] Badge de estado en el README

### Fase 6 — README
- [ ] Reescribir: qué es el proyecto, arquitectura, pipeline, resultados

### Fase 7 — Notebooks *(diferido)*
- [ ] Ejecutar notebooks y guardar output
- [ ] Revisar presentación y narrativa
