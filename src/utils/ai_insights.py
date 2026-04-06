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


