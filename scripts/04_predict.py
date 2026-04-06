"""
Script de prediccion — Spaceship Titanic.

Carga el modelo en produccion, preprocesa test.csv y genera submission.csv.
El feature set y las features numericas se leen del metadata del modelo
(models/production/model_metadata.json).

Ejecutar desde la raiz del proyecto:
    python scripts/04_predict.py
"""
import json
import sys

sys.path.insert(0, ".")  # scripts run from project root
sys.stdout.reconfigure(encoding='utf-8')
# pylint: disable=wrong-import-position

import mlflow
import joblib
import pandas as pd

from src.config.settings import (
    MODEL_METADATA,
    MODEL_PATH,
    SCALER_PATH,
    SUBMISSIONS_DIR,
    TEST_RAW,
    get_submission_path,
    get_target_encoder_path,
)
from src.features.feature_sets import FEATURE_SETS
from src.models.predict import generate_submission, preprocess_test
from src.models.tracking import mlrun
from src.reports.prediction_report import build_prediction_md


def _load_feature_set(meta_prod: dict):
    """Devuelve el FeatureSetConfig del modelo en produccion.

    Para modelos anteriores al registro de feature sets, usa fs-001_baseline.

    Args:
        meta_prod: Diccionario con el metadata del modelo en produccion.

    Returns:
        FeatureSetConfig correspondiente al modelo.
    """
    fs_name = meta_prod.get("feature_set_name", "fs-001_baseline")
    if fs_name not in FEATURE_SETS:
        print(f"  ⚠ Feature set '{fs_name}' no encontrado, usando fs-001_baseline.")
        fs_name = "fs-001_baseline"
    return fs_name, FEATURE_SETS[fs_name]


def main() -> None:
    """Ejecuta el pipeline de prediccion."""
    print("=" * 60)
    print("04_predict.py — Prediccion")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cargar modelo y scaler
    # ------------------------------------------------------------------
    print("Cargando modelo...")
    model = joblib.load(MODEL_PATH)
    model_type = type(model).__name__
    print(f"  Tipo de modelo: {model_type}")

    print("Cargando scaler...")
    scaler = joblib.load(SCALER_PATH)

    # ------------------------------------------------------------------
    # 2. Leer metadata, feature set y target encoder
    # ------------------------------------------------------------------
    print("Leyendo metadata del modelo...")
    meta_prod: dict = {}
    if MODEL_METADATA.exists():
        with open(MODEL_METADATA, encoding="utf-8") as f:
            meta_prod = json.load(f)

    fs_name, fs = _load_feature_set(meta_prod)
    print(f"  Feature set: {fs_name}")

    feature_cols: list | None = None
    if hasattr(model, "feature_names_in_"):
        feature_cols = model.feature_names_in_.tolist()
    if feature_cols is None:
        feature_cols = meta_prod.get("feature_names")
    if feature_cols is None:
        raise FileNotFoundError(
            "No se encontraron feature_names en el modelo ni en el metadata. "
            "Verifica que el modelo fue entrenado con 03_train.py."
        )
    print(f"  Features esperadas: {len(feature_cols)}")

    target_encoder = None
    te_path = get_target_encoder_path(fs_name)
    if te_path.exists():
        target_encoder = joblib.load(te_path)
        print(f"  Target encoder cargado: {list(target_encoder.keys())}")

    # ------------------------------------------------------------------
    # 3. Cargar test.csv y preprocesar
    # ------------------------------------------------------------------
    print("Cargando test.csv...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    print(f"  Registros en test: {len(df_test)}")

    print("Preprocesando test...")
    x_test = preprocess_test(
        df_test,
        fs=fs,
        feature_cols=feature_cols,
        scaler=scaler,
        target_encoder=target_encoder,
    )
    print(f"  Shape de X_test: {x_test.shape}")

    # ------------------------------------------------------------------
    # 4. Generar predicciones (con umbral optimo si esta guardado)
    # ------------------------------------------------------------------
    threshold = meta_prod.get("best_threshold", 0.5)
    print(f"Generando predicciones (umbral={threshold:.4f})...")
    submission = generate_submission(model, x_test, test_ids, threshold=threshold)

    # ------------------------------------------------------------------
    # 5. Guardar submission con etiqueta del experimento
    # ------------------------------------------------------------------
    exp_id = meta_prod.get("exp_id", "unknown") if meta_prod else "unknown"
    exp_label = f"exp-{exp_id}"
    submission_path = get_submission_path(exp_id)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"  Submission guardado: {submission_path}")

    # ------------------------------------------------------------------
    # 6. Estadisticas y reporte
    # ------------------------------------------------------------------
    n_total = len(submission)
    n_true = int(submission["Transported"].sum())
    n_false = n_total - n_true
    pct_true = n_true / n_total * 100
    pct_false = n_false / n_total * 100

    print("\nDistribucion de predicciones:")
    print(f"  True  (transportado):    {n_true:>5} ({pct_true:.1f}%)")
    print(f"  False (no transportado): {n_false:>5} ({pct_false:.1f}%)")
    print(f"  Total:                   {n_total}")

    build_prediction_md(
        model_type=model_type,
        exp_label=exp_label,
        fs_name=fs_name,
        n_total=n_total,
        n_true=n_true,
        n_false=n_false,
        pct_true=pct_true,
        pct_false=pct_false,
        submission_path=str(submission_path),
        threshold=threshold,
    )

    # ------------------------------------------------------------------
    # 7. MLflow tracking
    # ------------------------------------------------------------------
    with mlrun("predictions", tags={"stage": "predict", "exp_id": exp_id}):
        mlflow.log_params({
            "exp_id": exp_id,
            "threshold": threshold,
            "fs_name": fs_name,
            "model_type": model_type,
        })
        mlflow.log_metrics({
            "pct_transported": pct_true,
            "n_predictions": float(n_total),
        })

    print("\nScript de prediccion completado.")


if __name__ == "__main__":
    main()
