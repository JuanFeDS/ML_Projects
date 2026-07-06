# Decisión — Fuente de verdad del modelo

**Fecha:** 2026-07-06
**Contexto:** auditoría de repositorio del 2026-07-05 (ver `2026-07-05-session.md`)

## El problema

Desde exp-032 conviven tres "mejores modelos" que dejaron de hablarse entre sí:

| Fuente | Experimento | Métrica | Estado |
|---|---|---|---|
| `artifacts/production/model_metadata.json` (promoción vía `TrainingPipeline`) | exp-031 | val_accuracy=0.8352 | ❌ Descartado como fuente de verdad |
| Mejor score Kaggle real | exp-034 (ensemble soft voting exp-027+exp-033, 50/50) | **0.80944** | ✅ Fuente de verdad para submissions |
| Estándar de validación honesta | exp-053/054 (CatBoost + GroupKFold sobre LastName) | oof_acc=0.8144 | ✅ Fuente de verdad para metodología, sin confirmar en Kaggle |

## Por qué exp-031 queda descartado

`exp-031` hereda el leakage de `LastName_TE` diagnosticado el 2026-04-18 (miembros de la misma familia caían en train y validación por split random) **y además** entrena sobre pseudo-etiquetas generadas por un modelo (exp-027) que ya estaba inflado por ese mismo leakage. Su val_accuracy (0.8352) es el más alto jamás registrado por el pipeline oficial, pero nunca se subió a Kaggle — es la combinación más optimista y menos confiable de todo el historial.

## Criterio de uso

- **Si se necesita un submission hoy:** usar el ensemble de **exp-034** (soft voting 50/50 entre exp-027 y exp-033). Es el único candidato con score confirmado contra el leaderboard real de Kaggle.
- **Si se está iterando sobre features o modelos nuevos:** validar con **GroupKFold(LastName)** (metodología de exp-053/054), no con StratifiedKFold/split random. Cualquier val_accuracy sobre fs-017 o descendientes obtenido con split random está inflado por el leakage de apellidos.
- **`artifacts/production/` y `docs/model/model_card.md`** siguen apuntando a exp-031 y no deben tomarse como referencia hasta que se promueva un modelo reconstruido sobre exp-034 o un sucesor validado con GroupKFold + Kaggle.

## Pendiente (no bloqueante)

Confirmar exp-053/054 en Kaggle. Si su score real supera 0.80944, reemplaza a exp-034 como fuente de verdad única (metodología honesta + score real). Si no lo supera, exp-034 se mantiene como techo del proyecto y GroupKFold queda solo como estándar de validación para trabajo futuro.

## Detalle completo

Ver `ignore/estado_experimentos.md` (no versionado — historial completo de los 19 feature sets y 54 experimentos).
