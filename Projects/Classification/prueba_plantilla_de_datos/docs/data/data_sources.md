# Data Sources — Spaceship Titanic

## Fuente principal

| Campo | Detalle |
|---|---|
| **Competencia** | Spaceship Titanic — Kaggle Getting Started |
| **URL** | https://www.kaggle.com/competitions/spaceship-titanic |
| **Tipo** | Clasificación binaria — competencia de práctica |
| **Licencia** | Kaggle competition rules |

## Archivos del dataset

| Archivo | Ubicación | Descripción |
|---|---|---|
| `train.csv` | `data/raw/train.csv` | 8,693 pasajeros con etiqueta `Transported`. Usado para entrenamiento y validación. |
| `test.csv` | `data/raw/test.csv` | 4,277 pasajeros sin etiqueta. Usado para generar el submission de Kaggle. |
| `sample_submission.csv` | `data/raw/sample_submission.csv` | Formato esperado del archivo de entrega. |

## Datos procesados generados por el pipeline

| Archivo | Generado por | Descripción |
|---|---|---|
| `data/processed/train_clean.csv` | `scripts/01_eda.py` | Dataset con nulos imputados y columnas irrelevantes eliminadas. |
| `data/processed/train_features.csv` | `scripts/02_features.py` | Dataset con features derivadas (Deck, TotalSpending_Log, GroupSize, etc.). |
| `data/processed/train_scaled.csv` | `scripts/02_features.py` | Dataset final escalado, listo para entrenar. |
| `data/processed/submission.csv` | `scripts/04_predict.py` | Predicciones sobre test.csv en formato de entrega de Kaggle. |

## Notas

- Los archivos de `data/raw/` y `data/processed/` están excluidos de git (ver `.gitignore`).
- Para reproducir: descargar los datos de Kaggle y colocarlos en `data/raw/`.
