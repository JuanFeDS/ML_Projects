"""
Generación de análisis narrativo con Claude API para los reportes del pipeline.

Cada función recibe un diccionario de contexto con los resultados de una etapa
del pipeline y retorna un párrafo en español generado por Claude.

Si la API no está disponible (sin key, sin red, error inesperado), todas las
funciones retornan cadena vacía — el pipeline continúa sin la sección de análisis.
"""
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 350
_SYSTEM = (
    "Eres un data scientist analizando resultados de un proyecto de clasificación ML. "
    "Responde siempre en español. Sé conciso (máximo 150 palabras), con tono técnico "
    "pero claro. No uses listas ni encabezados: escribe un único párrafo fluido."
)


def _get_client():
    """Retorna un cliente Anthropic o None si no está disponible."""
    try:
        import anthropic
        from src.config.settings import ANTHROPIC_API_KEY
        if not ANTHROPIC_API_KEY:
            return None
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:  # pylint: disable=broad-except
        return None


def _call_claude(prompt: str) -> str:
    """Llama a la API de Claude con el prompt dado.

    Args:
        prompt: Texto del mensaje de usuario.

    Returns:
        Texto generado por Claude, o cadena vacía ante cualquier error.
    """
    client = _get_client()
    if client is None:
        return ""
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("No se pudo generar análisis con Claude API: %s", exc)
        return ""


def get_eda_insights(context: Dict[str, Any]) -> str:
    """Genera un párrafo interpretativo sobre el análisis exploratorio.

    Args:
        context: Diccionario con las claves:
            - n_rows (int): Número de filas del dataset.
            - n_cols (int): Número de columnas.
            - null_pct_avg (float): Porcentaje promedio de nulos.
            - target_balance (dict): {"True": float, "False": float} en porcentaje.
            - top_chi2 (list[dict]): Top 3 variables categóricas por chi².
              Cada dict tiene "feature", "chi2", "p".
            - top_numeric_corr (list[dict]): Top 3 variables numéricas por |r|.
              Cada dict tiene "feature", "r", "p_mw".
            - decisions (list[dict]): Decisiones por feature.
              Cada dict tiene "feature" y "accion" (MANTENER/TRANSFORMAR/DESCARTAR).

    Returns:
        Párrafo de análisis en español, o cadena vacía si la API falla.
    """
    prompt = (
        f"Analiza estos resultados del EDA de un dataset de clasificación:\n\n"
        f"- Dataset: {context.get('n_rows', '?')} filas × {context.get('n_cols', '?')} columnas\n"
        f"- Nulos promedio: {context.get('null_pct_avg', 0):.1f}%\n"
        f"- Balance del target: {json.dumps(context.get('target_balance', {}))}\n"
        f"- Top variables categóricas (chi²): {json.dumps(context.get('top_chi2', []))}\n"
        f"- Top variables numéricas (correlación): {json.dumps(context.get('top_numeric_corr', []))}\n"
        f"- Decisiones de features: {json.dumps(context.get('decisions', []))}\n\n"
        "Interpreta los hallazgos más importantes: calidad del dataset, qué variables "
        "tienen mayor poder predictivo y si las decisiones de features parecen apropiadas."
    )
    return _call_claude(prompt)


def get_features_insights(context: Dict[str, Any]) -> str:
    """Genera un párrafo interpretativo sobre el feature engineering aplicado.

    Args:
        context: Diccionario con las claves:
            - fs_name (str): Nombre del feature set.
            - fs_description (str): Descripción del feature set.
            - n_features_before (int): Columnas antes del pipeline.
            - n_features_after (int): Features finales tras encoding/OHE.
            - n_samples (int): Filas del dataset resultante.
            - transformations (list[str]): Lista de transformaciones aplicadas.

    Returns:
        Párrafo de análisis en español, o cadena vacía si la API falla.
    """
    prompt = (
        f"Analiza este proceso de feature engineering:\n\n"
        f"- Feature set: {context.get('fs_name', '?')}\n"
        f"- Descripción: {context.get('fs_description', '?')}\n"
        f"- Features antes: {context.get('n_features_before', '?')} columnas → "
        f"después: {context.get('n_features_after', '?')} features\n"
        f"- Muestras resultantes: {context.get('n_samples', '?')}\n"
        f"- Transformaciones aplicadas: {json.dumps(context.get('transformations', []))}\n\n"
        "Evalúa si la expansión/reducción de dimensionalidad es razonable, qué "
        "transformaciones aportan más valor y si hay algún riesgo en el pipeline aplicado."
    )
    return _call_claude(prompt)


def get_training_insights(context: Dict[str, Any]) -> str:
    """Genera un análisis de los resultados del entrenamiento de modelos.

    Args:
        context: Diccionario con las claves:
            - fs_name (str): Feature set usado.
            - cv_results (list[dict]): Lista de {"model": str, "cv_accuracy": float}.
            - best_model (str): Nombre del mejor modelo en CV.
            - best_cv_accuracy (float): Su accuracy en CV.
            - tuned_val_accuracy (float): Val accuracy del modelo tuneado.
            - tuned_val_roc_auc (float): Val ROC-AUC del modelo tuneado.
            - stacking_val_accuracy (float): Val accuracy del stacking.
            - moe_val_accuracy (float): Val accuracy del MoE.
            - winner_name (str): Nombre del modelo ganador final.
            - winner_val_accuracy (float): Val accuracy del ganador.
            - winner_val_roc_auc (float): Val ROC-AUC del ganador.
            - best_threshold (float): Umbral óptimo encontrado.
            - threshold_gain (float): Ganancia de accuracy con el umbral óptimo.
            - top_features (list[str]): Top 5 features por importancia (puede estar vacío).

    Returns:
        Párrafo de análisis en español, o cadena vacía si la API falla.
    """
    prompt = (
        f"Analiza estos resultados de entrenamiento de modelos ML:\n\n"
        f"- Feature set: {context.get('fs_name', '?')}\n"
        f"- Resultados CV: {json.dumps(context.get('cv_results', []))}\n"
        f"- Mejor en CV: {context.get('best_model', '?')} "
        f"(accuracy={context.get('best_cv_accuracy', 0):.4f})\n"
        f"- Tuneado → val_accuracy={context.get('tuned_val_accuracy', 0):.4f}, "
        f"ROC-AUC={context.get('tuned_val_roc_auc', 0):.4f}\n"
        f"- Stacking → val_accuracy={context.get('stacking_val_accuracy', 0):.4f}\n"
        f"- MoE → val_accuracy={context.get('moe_val_accuracy', 0):.4f}\n"
        f"- Ganador: {context.get('winner_name', '?')} "
        f"(val_accuracy={context.get('winner_val_accuracy', 0):.4f}, "
        f"ROC-AUC={context.get('winner_val_roc_auc', 0):.4f})\n"
        f"- Threshold óptimo: {context.get('best_threshold', 0.5):.4f} "
        f"(ganancia: {context.get('threshold_gain', 0):+.4f})\n"
        f"- Top features: {context.get('top_features', [])}\n\n"
        "Explica por qué ganó el modelo seleccionado, qué dice la diferencia entre "
        "CV y validación sobre el ajuste del modelo, y qué recomendarías para la "
        "siguiente iteración."
    )
    return _call_claude(prompt)


def get_prediction_insights(context: Dict[str, Any]) -> str:
    """Genera un párrafo sobre la distribución de predicciones generadas.

    Args:
        context: Diccionario con las claves:
            - model_name (str): Nombre del modelo usado.
            - exp_id (str): ID del experimento de referencia.
            - fs_name (str): Feature set del modelo.
            - n_samples (int): Número de muestras predichas.
            - threshold (float): Umbral de clasificación usado.
            - pct_transported (float): Porcentaje predicho como transportado (0-100).
            - pct_not_transported (float): Porcentaje predicho como no transportado (0-100).

    Returns:
        Párrafo de análisis en español, o cadena vacía si la API falla.
    """
    prompt = (
        f"Analiza esta distribución de predicciones de un modelo de clasificación:\n\n"
        f"- Modelo: {context.get('model_name', '?')} (experimento {context.get('exp_id', '?')})\n"
        f"- Feature set: {context.get('fs_name', '?')}\n"
        f"- Muestras predichas: {context.get('n_samples', '?')}\n"
        f"- Umbral de clasificación: {context.get('threshold', 0.5):.4f}\n"
        f"- Transportados: {context.get('pct_transported', 0):.1f}%\n"
        f"- No transportados: {context.get('pct_not_transported', 0):.1f}%\n"
        f"- Balance en train: ~50.4% transportados / 49.6% no transportados\n\n"
        "Evalúa si la distribución de predicciones es coherente con el balance "
        "del dataset de entrenamiento y qué implicaciones tiene el umbral elegido."
    )
    return _call_claude(prompt)
