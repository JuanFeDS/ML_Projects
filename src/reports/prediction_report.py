"""
Generación del reporte de predicciones para el pipeline Spaceship Titanic.

Función para construir reports/04_predictions.md a partir de los resultados
del script de predicción.
"""
import pandas as pd

from src.config.settings import MODEL_PATH, REPORTS_DIR
from src.reports.builder import MarkdownReport


def build_prediction_md(
    model_type: str,
    exp_label: str,
    fs_name: str,
    n_total: int,
    n_true: int,
    n_false: int,
    pct_true: float,
    pct_false: float,
    submission_path: str,
    threshold: float = 0.5,
    ai_insights: str = "",
) -> None:
    """Genera reports/04_predictions.md.

    Args:
        model_type: Nombre de la clase del modelo usado.
        exp_label: Etiqueta del experimento (e.g. 'exp-006').
        fs_name: Nombre del feature set usado en entrenamiento.
        n_total: Total de predicciones generadas.
        n_true: Cantidad de predicciones True (transportado).
        n_false: Cantidad de predicciones False (no transportado).
        pct_true: Porcentaje de True sobre el total.
        pct_false: Porcentaje de False sobre el total.
        submission_path: Ruta donde se guardó submission.csv.
        threshold: Umbral de clasificación utilizado.
        ai_insights: Párrafo de análisis generado por Claude (opcional).
    """
    md = MarkdownReport("Reporte de Predicciones — Spaceship Titanic")

    md.add_section("Modelo Utilizado")
    md.add_metric("Experimento", exp_label)
    md.add_metric("Tipo de modelo", model_type)
    md.add_metric("Feature set", fs_name)
    md.add_metric("Umbral de clasificacion", f"{threshold:.4f}")
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

    if ai_insights:
        md.add_section("Análisis")
        md.add_text(ai_insights)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md.save(str(REPORTS_DIR / "04_predictions.md"))
