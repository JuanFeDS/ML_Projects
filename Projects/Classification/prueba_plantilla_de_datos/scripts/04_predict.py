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

import joblib
import pandas as pd

from src.config.settings import (
    MODEL_METADATA,
    MODEL_PATH,
    REPORTS_DIR,
    SCALER_PATH,
    SUBMISSIONS_DIR,
    TEST_RAW,
    get_submission_path,
    get_target_encoder_path,
)
from src.features.feature_sets import FEATURE_SETS
from src.models.predict import generate_submission, preprocess_test
from src.reports.builder import MarkdownReport


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

    _build_markdown_report(
        model_type=model_type,
        exp_label=exp_label,
        fs_name=fs_name,
        n_total=n_total,
        n_true=n_true,
        n_false=n_false,
        pct_true=pct_true,
        pct_false=pct_false,
        submission_path=str(submission_path),
    )

    print("\nScript de prediccion completado.")


def _build_markdown_report(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model_type: str,
    exp_label: str,
    fs_name: str,
    n_total: int,
    n_true: int,
    n_false: int,
    pct_true: float,
    pct_false: float,
    submission_path: str,
) -> None:
    """Genera Reports/04_predictions.md.

    Args:
        model_type: Nombre de la clase del modelo usado.
        exp_label: Etiqueta del experimento (e.g. 'exp-006_Stacking').
        fs_name: Nombre del feature set usado en entrenamiento.
        n_total: Total de predicciones generadas.
        n_true: Cantidad de predicciones True (transportado).
        n_false: Cantidad de predicciones False (no transportado).
        pct_true: Porcentaje de True sobre el total.
        pct_false: Porcentaje de False sobre el total.
        submission_path: Ruta donde se guardo submission.csv.
    """
    md = MarkdownReport("Reporte de Predicciones — Spaceship Titanic")

    md.add_section("Modelo Utilizado")
    md.add_metric("Experimento", exp_label)
    md.add_metric("Tipo de modelo", model_type)
    md.add_metric("Feature set", fs_name)
    md.add_metric("Ruta del modelo", str(MODEL_PATH))

    md.add_section("Predicciones Generadas")
    md.add_metric("Registros predichos", n_total)

    dist_df = pd.DataFrame([
        {
            "Transported": "True  (transportado)",
            "Conteo": n_true,
            "Porcentaje": f"{pct_true:.1f}%",
        },
        {
            "Transported": "False (no transportado)",
            "Conteo": n_false,
            "Porcentaje": f"{pct_false:.1f}%",
        },
        {
            "Transported": "Total",
            "Conteo": n_total,
            "Porcentaje": "100.0%",
        },
    ])
    md.add_table(dist_df, index=False)

    md.add_section("Archivo Generado")
    md.add_metric("Ruta", submission_path)
    md.add_text(
        "El archivo `submission.csv` contiene las columnas `PassengerId` y "
        "`Transported` (valores booleanos), listo para subir a Kaggle."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md.save(str(REPORTS_DIR / "04_predictions.md"))


if __name__ == "__main__":
    main()
