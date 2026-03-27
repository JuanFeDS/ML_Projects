"""
Script de entrenamiento — Spaceship Titanic.

Evalua multiples modelos con CV, tunea el mejor, construye stacking,
selecciona el modelo ganador y guarda los artefactos.

Ejecutar desde la raiz del proyecto:
    python scripts/03_train.py
    python scripts/03_train.py --feature-set fs-002_cryo_interactions
"""
import argparse
import json
import shutil
import sys

sys.path.insert(0, ".")  # scripts run from project root
sys.stdout.reconfigure(encoding='utf-8')
# pylint: disable=wrong-import-position

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config.settings import (
    DOCS_DIR,
    EXPERIMENTS_DIR,
    MODEL_METADATA,
    MODEL_PATH,
    PRODUCTION_DIR,
    REPORTS_DIR,
    SCALER_PATH,
    get_scaler_path,
    get_train_scaled,
)
from src.features.constants import TARGET
from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS
from src.models.catalogue import MODELS, PARAM_SPACES
from src.models.training import (
    analyze_errors,
    build_moe,
    build_stacking,
    evaluate_models,
    evaluate_on_validation,
    optimize_threshold,
    tune_model,
)
from src.reports.builder import (
    HTMLReport,
    MarkdownReport,
    append_experiment_log,
    get_next_exp_id,
    is_duplicate_experiment,
    write_experiment_card,
    write_model_card,
)
from src.reports.training_plots import (
    cv_accuracy_bar,
    feature_importance_bar,
    validation_metrics_bar,
)


def _parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenamiento — Spaceship Titanic")
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=list(FEATURE_SETS.keys()),
        help=(
            "Feature set a usar. Ver src/features/feature_sets.py. "
            f"Default: {DEFAULT_FEATURE_SET}"
        ),
    )
    return parser.parse_args()


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Ejecuta el pipeline completo de entrenamiento."""
    args = _parse_args()
    fs_name = args.feature_set
    fs = FEATURE_SETS[fs_name]

    train_scaled_path = get_train_scaled(fs_name)
    scaler_pkl_path = get_scaler_path(fs_name)

    print("=" * 60)
    print("🤖 03_train.py — Entrenamiento")
    print(f"  Feature set: {fs_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Carga de datos
    # ------------------------------------------------------------------
    print("📂 Cargando datos...")
    df = pd.read_csv(train_scaled_path)
    print(f"  Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    y = df[TARGET]
    x_df = df.drop(columns=[TARGET])

    # ------------------------------------------------------------------
    # 2. Split train / validation
    # ------------------------------------------------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        x_df, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {x_train.shape[0]} | Val: {x_val.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Cross-validation strategy
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 4. Evaluar todos los modelos con CV
    # ------------------------------------------------------------------
    print("\n📊 Evaluando modelos con CV...")
    cv_results = evaluate_models(MODELS, x_train, y_train, cv)
    print(cv_results.to_string())

    # ------------------------------------------------------------------
    # 5. Seleccionar mejor modelo tunable (excluir Baseline y MoE)
    # ------------------------------------------------------------------
    _NOT_TUNABLE = {"Baseline", "MoE_CatBoost"}
    best_name = next(
        n for n in cv_results.index
        if n not in _NOT_TUNABLE and n in PARAM_SPACES
    )
    print(f"  Mejor modelo CV (tunable): {best_name}")

    best_cv_score = float(cv_results.loc[best_name, "cv_accuracy_mean"])

    # ------------------------------------------------------------------
    # 6. Tunear el mejor modelo
    # ------------------------------------------------------------------
    print(f"\n🔧 Tuneando {best_name} (n_iter=25)...")
    tuned_model, best_params, tuned_cv_score = tune_model(
        MODELS[best_name], PARAM_SPACES[best_name], x_train, y_train, cv, n_iter=25
    )
    print(f"  Mejor score CV tuneado: {tuned_cv_score:.4f}")
    print(f"  Mejores hiperparametros: {best_params}")

    # ------------------------------------------------------------------
    # 7. Stacking con los 3 mejores modelos base (excluye MoE del top-3)
    # ------------------------------------------------------------------
    print("\n🏗️  Construyendo Stacking...")
    top_names = [
        n for n in cv_results.index if n not in ("Baseline", "MoE_CatBoost")
    ][:3]
    print(f"  Base estimators: {top_names}")
    base_estimators = [(name, MODELS[name]) for name in top_names]
    stacking_model, stacking_cv_score = build_stacking(
        base_estimators, x_train, y_train, cv
    )
    print(f"  Stacking CV accuracy: {stacking_cv_score:.4f}")

    # ------------------------------------------------------------------
    # 7b. MoE con el modelo tuneado como experto base
    # ------------------------------------------------------------------
    print("\n🔀 Construyendo Mixture of Experts (experto: tuneado)...")
    moe_model, moe_cv_score = build_moe(tuned_model, x_train, y_train, cv)
    sizes = moe_model.get_segment_sizes(x_train)
    print(f"  Segmento cryo:   {sizes['cryo']:,} muestras")
    print(f"  Segmento activo: {sizes['active']:,} muestras")
    print(f"  MoE CV accuracy: {moe_cv_score:.4f}")

    # ------------------------------------------------------------------
    # 8. Evaluar en validacion
    # ------------------------------------------------------------------
    print("\n🧪 Evaluando en validacion...")
    tuned_val = evaluate_on_validation(tuned_model, x_train, y_train, x_val, y_val)
    stacking_val = evaluate_on_validation(
        stacking_model, x_train, y_train, x_val, y_val
    )
    moe_val = evaluate_on_validation(moe_model, x_train, y_train, x_val, y_val)
    print(
        f"  Tuneado  -> val_acc={tuned_val['val_accuracy']:.4f} | "
        f"roc_auc={tuned_val['val_roc_auc']:.4f}"
    )
    print(
        f"  Stacking -> val_acc={stacking_val['val_accuracy']:.4f} | "
        f"roc_auc={stacking_val['val_roc_auc']:.4f}"
    )
    print(
        f"  MoE      -> val_acc={moe_val['val_accuracy']:.4f} | "
        f"roc_auc={moe_val['val_roc_auc']:.4f}"
    )

    # ------------------------------------------------------------------
    # 9. Seleccionar ganador final
    # ------------------------------------------------------------------
    candidates = [
        (f"{best_name} (tuneado)", tuned_model, tuned_val),
        ("Stacking", stacking_model, stacking_val),
        ("MoE", moe_model, moe_val),
    ]
    winner_name, winner_model, winner_val = max(
        candidates, key=lambda t: t[2]["val_accuracy"]
    )

    print(f"\n🏆 Modelo ganador: {winner_name}")

    # ------------------------------------------------------------------
    # 9b. Error analysis sobre el ganador (ya entrenado en evaluate_on_validation)
    # ------------------------------------------------------------------
    print("\n🔍 Analizando errores del modelo ganador...")
    y_pred_winner = winner_val["y_pred"]
    y_proba_winner = winner_val["y_proba"]
    error_tables = analyze_errors(
        x_val, y_val, pd.Series(y_pred_winner, index=y_val.index)
    )
    for seg, tbl in error_tables.items():
        print(f"\n  Error rate por {seg}:")
        print(tbl.to_string(index=False))

    # ------------------------------------------------------------------
    # 9c. Threshold optimization sobre el ganador
    # ------------------------------------------------------------------
    print("\n⚖️  Optimizando umbral de clasificacion...")
    best_threshold, threshold_acc = optimize_threshold(y_val, y_proba_winner)
    gain = round(threshold_acc - winner_val["val_accuracy"], 4)
    print(f"  Umbral optimo:  {best_threshold:.4f} → val_accuracy: {threshold_acc:.4f}")
    print(f"  Default (0.50): val_accuracy: {winner_val['val_accuracy']:.4f}")
    print(f"  Ganancia:       {gain:+.4f}")

    # ------------------------------------------------------------------
    # 10. Re-entrenar el ganador sobre x_train completo
    # ------------------------------------------------------------------
    print("🔄 Re-entrenando sobre x_train completo...")
    winner_model.fit(x_train, y_train)

    # ------------------------------------------------------------------
    # 11. Guardar artefactos + comparar vs produccion actual
    # ------------------------------------------------------------------
    log_path = str(DOCS_DIR / "model" / "experimentation_log.md")
    exp_id = get_next_exp_id(log_path)

    feature_names = x_train.columns.tolist()

    # Calcular cambios vs feature set parent
    features_added: list = []
    features_removed: list = []
    if fs.parent and fs.parent in FEATURE_SETS:
        parent_fs = FEATURE_SETS[fs.parent]
        current_cols = set(
            fs.numeric_features + fs.categorical_cols + list(fs.target_encode_cols)
        )
        parent_cols = set(
            parent_fs.numeric_features
            + parent_fs.categorical_cols
            + list(parent_fs.target_encode_cols)
        )
        features_added = sorted(current_cols - parent_cols)
        features_removed = sorted(parent_cols - current_cols)

    # Accuracy efectiva: la mejor entre default (0.5) y umbral optimizado
    effective_acc = max(winner_val["val_accuracy"], threshold_acc)
    effective_threshold = best_threshold if threshold_acc > winner_val["val_accuracy"] else 0.5

    metadata = {
        "exp_id": exp_id,
        "model_name": winner_name,
        "feature_set_name": fs_name,
        "feature_set_description": fs.description,
        "feature_set_parent": fs.parent,
        "features_added": features_added,
        "features_removed": features_removed,
        "numeric_features": fs.numeric_features,
        "val_accuracy": effective_acc,
        "val_accuracy_default_threshold": winner_val["val_accuracy"],
        "val_roc_auc": winner_val["val_roc_auc"],
        "cv_accuracy": best_cv_score,
        "n_features": x_train.shape[1],
        "n_train_samples": x_train.shape[0],
        "best_params": best_params,
        "best_threshold": effective_threshold,
        "feature_names": feature_names,
    }

    # Verificar si este experimento ya fue registrado
    if is_duplicate_experiment(metadata, log_path):
        print(
            "  ⏭️  Experimento identico ya registrado — "
            "se omiten artefacto, log y card."
        )
        promoted = False
    else:
        # Guardar artefacto en experiments/ con el ID del experimento
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = winner_name.replace(" ", "_").replace("(", "").replace(")", "")
        exp_artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_{safe_name}.pkl"
        joblib.dump(winner_model, exp_artifact)
        print(f"  💾 Artefacto guardado: {exp_artifact}")

        # Cargar metricas del modelo actual en produccion (si existe)
        current_best_acc: float | None = None
        if MODEL_METADATA.exists():
            with open(MODEL_METADATA, encoding="utf-8") as f:
                current_meta = json.load(f)
            current_best_acc = current_meta.get("val_accuracy")

        # Promover solo si supera al modelo actual (usando accuracy efectiva con umbral)
        new_acc = effective_acc
        promoted = current_best_acc is None or new_acc > current_best_acc

        if promoted:
            PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(winner_model, MODEL_PATH)
            if scaler_pkl_path.exists():
                shutil.copy2(scaler_pkl_path, SCALER_PATH)
            with open(MODEL_METADATA, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            label = "NUEVO MEJOR MODELO" if current_best_acc else "primer modelo"
            print(f"  🚀 [{label}] Promovido a produccion: {MODEL_PATH}")
        else:
            print(
                f"  ❌ No promovido — val_accuracy {new_acc:.4f} "
                f"no supera {current_best_acc:.4f} (produccion actual)"
            )

        cards_dir = str(DOCS_DIR / "model" / "cards")
        append_experiment_log(
            metadata=metadata,
            path=log_path,
            exp_id=exp_id,
            promoted=promoted,
            current_best_acc=current_best_acc,
            cv_results=cv_results,
            features_added=features_added,
            features_removed=features_removed,
        )
        write_experiment_card(
            metadata=metadata,
            feature_names=feature_names,
            exp_id=exp_id,
            cards_dir=cards_dir,
            promoted=promoted,
            current_best_acc=current_best_acc,
        )
        if promoted:
            write_model_card(
                metadata=metadata,
                feature_names=feature_names,
                path=str(DOCS_DIR / "model" / "model_card.md"),
            )

    # ------------------------------------------------------------------
    # 12. Reportes operacionales (siempre se regeneran)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _build_markdown_report(
        cv_results=cv_results,
        best_name=best_name,
        best_params=best_params,
        tuned_val=tuned_val,
        stacking_val=stacking_val,
        moe_val=moe_val,
        winner_name=winner_name,
        winner_val=winner_val,
        top_names=top_names,
        fs_name=fs_name,
        error_tables=error_tables,
        best_threshold=best_threshold,
        threshold_acc=threshold_acc,
    )
    _build_html_report(
        cv_results=cv_results,
        tuned_val=tuned_val,
        stacking_val=stacking_val,
        moe_val=moe_val,
        best_name=best_name,
        winner_model=winner_model,
        feature_names=x_train.columns.tolist(),
        error_tables=error_tables,
        best_threshold=best_threshold,
        threshold_acc=threshold_acc,
    )

    print("\n✅ Pipeline de entrenamiento completado.")


def _build_markdown_report(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cv_results: pd.DataFrame,
    best_name: str,
    best_params: dict,
    tuned_val: dict,
    stacking_val: dict,
    moe_val: dict,
    winner_name: str,
    winner_val: dict,
    top_names: list,
    fs_name: str,
    error_tables: dict,
    best_threshold: float,
    threshold_acc: float,
) -> None:
    """Genera Reports/03_training.md.

    Args:
        cv_results: Resultados de CV de todos los modelos.
        best_name: Nombre del mejor modelo seleccionado.
        best_params: Hiperparametros optimos encontrados.
        tuned_val: Metricas del modelo tuneado en validacion.
        stacking_val: Metricas del stacking en validacion.
        moe_val: Metricas del MoE en validacion.
        winner_name: Nombre del modelo ganador final.
        winner_val: Metricas del ganador en validacion.
        top_names: Nombres de los 3 mejores modelos base del stacking.
        fs_name: Nombre del feature set usado.
        error_tables: Tablas de tasa de error por segmento.
        best_threshold: Umbral optimo encontrado.
        threshold_acc: Accuracy con el umbral optimo.
    """
    md = MarkdownReport("Reporte de Entrenamiento — Spaceship Titanic")

    md.add_section("Feature Set")
    md.add_metric("Nombre", fs_name)

    md.add_section("Resultados Cross-Validation (todos los modelos)")
    md.add_text(
        "Evaluacion con StratifiedKFold (5 folds). Ordenado por cv_accuracy_mean."
    )
    cv_display = cv_results.reset_index().rename(columns={"index": "Modelo"})
    md.add_table(cv_display, index=False)

    md.add_section(f"Mejor Modelo Seleccionado: {best_name}")
    md.add_text(
        "El modelo con mayor cv_accuracy_mean fue tuneado con "
        "Optuna TPE (n_iter=25)."
    )
    md.add_subsection("Mejores hiperparametros encontrados")
    params_str = "\n".join(f"{k}: {v}" for k, v in best_params.items())
    md.add_code(params_str, lang="")

    md.add_section("Evaluacion en Validacion: Tuneado vs Stacking vs MoE")
    md.add_text(
        f"Stacking construido con los 3 mejores modelos base: {', '.join(top_names)}. "
        "MoE entrena un experto CatBoost por segmento (cryo / activo)."
    )
    val_df = pd.DataFrame([
        {
            "Modelo": f"{best_name} (tuneado)",
            "val_accuracy": tuned_val["val_accuracy"],
            "val_roc_auc": tuned_val["val_roc_auc"],
        },
        {
            "Modelo": "Stacking",
            "val_accuracy": stacking_val["val_accuracy"],
            "val_roc_auc": stacking_val["val_roc_auc"],
        },
        {
            "Modelo": "MoE (CatBoost x segmento)",
            "val_accuracy": moe_val["val_accuracy"],
            "val_roc_auc": moe_val["val_roc_auc"],
        },
    ])
    md.add_table(val_df, index=False)

    md.add_subsection(f"Classification Report — {best_name} (tuneado)")
    md.add_code(tuned_val["classification_report"], lang="")

    md.add_subsection("Classification Report — Stacking")
    md.add_code(stacking_val["classification_report"], lang="")

    md.add_section("Modelo Ganador Final")
    md.add_metric("Modelo", winner_name)
    md.add_metric("val_accuracy", winner_val["val_accuracy"])
    md.add_metric("val_roc_auc", winner_val["val_roc_auc"])
    md.add_text(
        "El modelo ganador fue re-entrenado sobre el conjunto completo de train "
        "y guardado en `models/production/best_model.pkl`."
    )

    md.add_section("Error Analysis — Tasa de error por segmento")
    md.add_text("Porcentaje de errores del modelo ganador en el conjunto de validacion, desglosado por variable.")
    for seg, tbl in error_tables.items():
        md.add_subsection(seg)
        md.add_table(tbl, index=False)

    md.add_section("Threshold Optimization")
    md.add_metric("Umbral optimo", best_threshold)
    md.add_metric("val_accuracy con umbral optimo", threshold_acc)
    md.add_metric("val_accuracy con umbral 0.50", winner_val["val_accuracy"])
    md.add_metric("Ganancia", round(threshold_acc - winner_val["val_accuracy"], 4))

    md.save(str(REPORTS_DIR / "03_training.md"))


def _build_html_report(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cv_results: pd.DataFrame,
    tuned_val: dict,
    stacking_val: dict,
    moe_val: dict,
    best_name: str,
    winner_model: object,
    feature_names: list,
    error_tables: dict,
    best_threshold: float,
    threshold_acc: float,
) -> None:
    """Genera Reports/03_training.html.

    Args:
        cv_results: Resultados de CV de todos los modelos.
        tuned_val: Metricas del modelo tuneado en validacion.
        stacking_val: Metricas del stacking en validacion.
        moe_val: Metricas del MoE en validacion.
        best_name: Nombre del mejor modelo seleccionado.
        winner_model: Estimador ganador entrenado.
        feature_names: Lista de nombres de features del modelo.
        error_tables: Tablas de tasa de error por segmento.
        best_threshold: Umbral optimo encontrado.
        threshold_acc: Accuracy con el umbral optimo.
    """
    html = HTMLReport("Reporte de Entrenamiento — Spaceship Titanic")

    winner_acc = max(
        tuned_val["val_accuracy"],
        stacking_val["val_accuracy"],
        moe_val["val_accuracy"],
    )
    winner_roc = max(
        tuned_val["val_roc_auc"],
        stacking_val["val_roc_auc"],
        moe_val["val_roc_auc"],
    )
    n_models_evaluated = len(cv_results)

    html.add_intro(
        f"Se evaluaron <b>{n_models_evaluated} modelos</b> mediante StratifiedKFold (5 folds). "
        f"El mejor modelo en CV (<b>{best_name}</b>) fue tuneado con Optuna TPE "
        f"(25 trials) y comparado contra Stacking y un Mixture of Experts (MoE) "
        f"que entrena un CatBoost especializado por segmento (cryo / activo). "
        f"El modelo ganador alcanza una precision de <b>{winner_acc:.1%}</b> y un "
        f"ROC-AUC de <b>{winner_roc:.4f}</b> en el conjunto de validacion (hold-out 20%)."
    )
    html.add_metrics_grid([
        (f"{winner_acc:.1%}", "val accuracy"),
        (f"{winner_roc:.4f}", "val ROC-AUC"),
        (f"{tuned_val['val_accuracy']:.1%}", f"{best_name} tuneado"),
        (f"{stacking_val['val_accuracy']:.1%}", "stacking"),
        (f"{moe_val['val_accuracy']:.1%}", "MoE"),
        (n_models_evaluated, "modelos evaluados"),
    ])

    html.add_section("Cross-Validation Accuracy (todos los modelos)")
    html.add_text(
        "Comparacion de todos los clasificadores con StratifiedKFold 5-fold sobre el "
        "conjunto de entrenamiento (80% del total). El Baseline (DummyClassifier) sirve "
        "como piso minimo de referencia."
    )
    html.add_figure(
        cv_accuracy_bar(cv_results), title="CV Accuracy — todos los modelos"
    )

    html.add_section("Validacion: Tuneado vs Stacking vs MoE")
    html.add_text(
        f"Evaluacion final en hold-out (20%). Se compara el modelo tuneado "
        f"({best_name}), el Stacking y el MoE para seleccionar el ganador."
    )
    val_models = [f"{best_name} (tuneado)", "Stacking", "MoE"]
    val_acc = [
        tuned_val["val_accuracy"],
        stacking_val["val_accuracy"],
        moe_val["val_accuracy"],
    ]
    val_roc = [
        tuned_val["val_roc_auc"],
        stacking_val["val_roc_auc"],
        moe_val["val_roc_auc"],
    ]
    html.add_figure(
        validation_metrics_bar(val_models, val_acc, val_roc),
        title="Val Accuracy y ROC-AUC",
    )

    fi_fig = feature_importance_bar(winner_model, feature_names)
    if fi_fig is not None:
        html.add_section("Feature Importance (top 20)")
        html.add_text(
            "Importancia de features del modelo ganador. Permite verificar coherencia "
            "con el analisis estadistico realizado en el EDA."
        )
        html.add_figure(fi_fig, title="Feature Importance")

    html.add_section("Error Analysis — Tasa de error por segmento")
    html.add_text(
        "Porcentaje de predicciones incorrectas del modelo ganador desglosado por "
        "variable. Permite identificar que segmentos son mas dificiles de clasificar."
    )
    for seg, tbl in error_tables.items():
        html.add_text(f"<b>{seg}</b>")
        html.add_text(tbl.to_html(index=False, border=0))

    html.add_section("Threshold Optimization")
    threshold_gain = round(threshold_acc - winner_acc, 4)
    html.add_text(
        f"Busqueda de umbral optimo en [0.30, 0.70] (200 puntos). "
        f"<b>Umbral optimo: {best_threshold:.4f}</b> → val_accuracy: "
        f"<b>{threshold_acc:.4f}</b> (vs {winner_acc:.4f} con 0.50, "
        f"ganancia: {threshold_gain:+.4f})."
    )

    html.save(str(REPORTS_DIR / "03_training.html"))


if __name__ == "__main__":
    main()
