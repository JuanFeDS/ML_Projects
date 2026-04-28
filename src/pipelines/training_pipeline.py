"""
Pipeline de entrenamiento orientado a objetos.

Carga datos escalados, ejecuta CV / tuning / ensambles, gestiona artefactos
y documentacion en docs/. Los reportes operacionales (reports/) se emiten
via ReportFactory a partir del dict retornado por TrainingPipeline.run().
"""

from __future__ import annotations

import json
import shutil
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
from src.config.vcs import create_git_tag, get_git_commit
from src.features.constants import TARGET
from src.features.feature_sets import FEATURE_SETS
from src.models.ensembles import build_moe, build_stacking
from src.models.evaluation import analyze_errors, compute_oof_metrics, evaluate_models, optimize_threshold
from src.models.tracking import mlrun
from src.models.training import (
    MODELS,
    PARAM_SPACES,
    TuneConfig,
    add_model_prefix,
    make_fold_te_pipeline,
    prefix_param_space,
    tune_model,
)
from src.reports.experiments.log import (
    ExperimentContext,
    append_experiment_log,
    get_next_exp_id,
    is_duplicate_experiment,
)
from src.reports.experiments.model_cards import write_experiment_card, write_model_card
from src.reports.training.shap_plots import ValPredictions, compute_shap_plots


class TrainingPipeline:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Orquesta el ciclo completo de entrenamiento para un feature set dado.

    Etapas:
        1. Carga de datos
        2. Evaluacion CV de todos los modelos del catalogo
        3. Tuning con Optuna del modelo ganador
        4. Construccion de Stacking y Mixture of Experts
        5. Seleccion del ganador por OOF accuracy
        6. Analisis de errores y optimizacion de umbral
        7. Re-entrenamiento sobre dataset completo
        8. Guardado de artefactos y promocion a produccion
        9. Registro en MLflow, experimentation log y model card

    Args:
        fs_name: Nombre del feature set (ej. fs-001_baseline).
        model_name: Si se indica, entrena solo ese modelo (sin comparacion multiple).
        tune: Si True, ejecuta tuning de hiperparametros con Optuna.
        build_stack: Si True, construye un StackingClassifier con los top-3 modelos.
        build_moe_flag: Si True, construye un Mixture of Experts.
        compute_shap: Si True, calcula SHAP values y agrega plots al reporte.
        n_iter: Numero de trials de Optuna.
    """

    def __init__(
        self,
        fs_name: str,
        model_name: Optional[str] = None,
        tune: bool = True,
        build_stack: bool = True,
        build_moe_flag: bool = True,
        compute_shap: bool = False,
        n_iter: int = 25,
    ) -> None:
        self.fs_name = fs_name
        self.model_name = model_name
        self.tune = tune
        self.build_stack = build_stack
        self.build_moe_flag = build_moe_flag
        self.compute_shap = compute_shap
        self.n_iter = n_iter

        self.fs = FEATURE_SETS[fs_name]
        self.fold_te_cols: List[str] = list(getattr(self.fs, "fold_te_cols", []))
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Estado interno — se puebla en cada etapa
        self._x_df: pd.DataFrame
        self._y: pd.Series
        self._cv_results: pd.DataFrame
        self._best_name: str
        self._best_cv_score: float
        self._tuned_model: Any
        self._best_params: dict
        self._stacking_model: Any = None
        self._stacking_val: dict = {}
        self._stacking_cv_score: float = 0.0
        self._moe_model: Any = None
        self._moe_val: dict = {}
        self._moe_cv_score: float = 0.0
        self._winner_name: str
        self._winner_model: Any
        self._winner_val: dict
        self._tuned_val: dict
        self._error_tables: dict
        self._best_threshold: float
        self._threshold_acc: float
        self._top_names: List[str]
        self._metadata: dict
        self._promoted: bool = False

    def _wrap_pipeline(self, model: Any) -> Any:
        """Envuelve el modelo en un Pipeline con TE fold-aware si el fs lo requiere."""
        if self.fold_te_cols:
            return make_fold_te_pipeline(model, self.fold_te_cols)
        return model

    def _load_data(self) -> None:
        """Carga el dataset escalado del feature set."""
        train_path = get_train_scaled(self.fs_name)
        df = pd.read_csv(train_path)
        print(f"[...] Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        self._y = df[TARGET]
        self._x_df = df.drop(columns=[TARGET])
        print(f"  Validacion: StratifiedKFold(5) | {self._x_df.shape[0]} filas")

    def _evaluate(self) -> None:
        """Evaluacion CV de todos los modelos del catalogo."""
        _NOT_TUNABLE = {"Baseline"}  # pylint: disable=invalid-name
        models_to_eval = (
            {self.model_name: MODELS[self.model_name]} if self.model_name else MODELS
        )
        if self.fold_te_cols:
            print(f"  Fold-aware TE activo para: {self.fold_te_cols}")
            models_to_eval = {
                name: make_fold_te_pipeline(m, self.fold_te_cols)
                for name, m in models_to_eval.items()
            }

        print(f"\n[CV] Evaluando: {list(models_to_eval.keys())}")
        self._cv_results = evaluate_models(models_to_eval, self._x_df, self._y, self.cv)
        print(self._cv_results.to_string())

        self._best_name = (
            self.model_name
            if self.model_name
            else next(
                n
                for n in self._cv_results.index
                if n not in _NOT_TUNABLE and n in PARAM_SPACES
            )
        )
        self._best_cv_score = float(
            self._cv_results.loc[self._best_name, "cv_accuracy_mean"]
        )
        print(
            f"  Modelo seleccionado: {self._best_name} | CV: {self._best_cv_score:.4f}"
        )

    def _tune(self) -> None:
        """Tuning de hiperparametros con Optuna."""
        self._best_params = {}
        if self.tune and self._best_name in PARAM_SPACES:
            print(f"\n[TUNE] Tuneando {self._best_name} (n_iter={self.n_iter})...")
            base = self._wrap_pipeline(MODELS[self._best_name])
            space_fn = (
                prefix_param_space(PARAM_SPACES[self._best_name])
                if self.fold_te_cols
                else PARAM_SPACES[self._best_name]
            )
            transform = add_model_prefix if self.fold_te_cols else None
            self._tuned_model, raw_params, score = tune_model(
                base,
                self._x_df,
                self._y,
                self.cv,
                TuneConfig(space_fn, n_iter=self.n_iter, param_transform=transform),
            )
            self._best_params = (
                {k.replace("model__", ""): v for k, v in raw_params.items()}
                if self.fold_te_cols
                else raw_params
            )
            print(f"  CV tuneado: {score:.4f} | Params: {self._best_params}")
        else:
            self._tuned_model = self._wrap_pipeline(MODELS[self._best_name])
            print(
                f"\n[SKIP] Tuning omitido -- usando params por defecto de {self._best_name}"
            )

    def _build_ensembles(self) -> None:
        """Construye Stacking y Mixture of Experts."""
        if self.build_stack and not self.model_name:
            print("\n[STACK] Construyendo Stacking...")
            self._top_names = [n for n in self._cv_results.index if n != "Baseline"][:3]
            base_estimators = [
                (name, self._wrap_pipeline(MODELS[name])) for name in self._top_names
            ]
            self._stacking_model, self._stacking_cv_score = build_stacking(
                base_estimators, self._x_df, self._y, self.cv
            )
            self._stacking_val = {
                "val_accuracy": self._stacking_cv_score,
                "val_roc_auc": 0.0,
                "classification_report": "(CV only — OOF no disponible para Stacking)",
                "y_pred": [],
                "y_proba": [],
            }
            print(f"  Stacking -> cv_acc={self._stacking_cv_score:.4f}")
        else:
            self._top_names = [self._best_name]

        if self.build_moe_flag:
            print("\n[MOE] Construyendo Mixture of Experts...")
            self._moe_model, self._moe_cv_score = build_moe(
                self._tuned_model, self._x_df, self._y, self.cv
            )
            sizes = self._moe_model.get_segment_sizes(self._x_df)
            self._moe_val = {
                "val_accuracy": self._moe_cv_score,
                "val_roc_auc": 0.0,
                "classification_report": "(CV only — OOF del MoE)",
                "y_pred": [],
                "y_proba": [],
            }
            print(
                f"  MoE cryo: {sizes['cryo']:,} | activo: {sizes['active']:,} -> cv_acc={self._moe_cv_score:.4f}"
            )

    def _select_and_evaluate_winner(self) -> None:
        """Selecciona el ganador y calcula OOF, errores y umbral optimo."""
        print("\n[OOF] Calculando OOF del modelo tuneado...")
        self._tuned_val = compute_oof_metrics(
            self._tuned_model, self._x_df, self._y, self.cv
        )
        print(
            f"  {self._best_name} -> OOF acc={self._tuned_val['val_accuracy']:.4f} | roc_auc={self._tuned_val['val_roc_auc']:.4f}"
        )

        candidates = [
            (self._best_name, self._tuned_model, self._tuned_val["val_accuracy"])
        ]
        if self._stacking_val:
            candidates.append(
                ("Stacking", self._stacking_model, self._stacking_cv_score)
            )
        if self._moe_val:
            candidates.append(("MoE", self._moe_model, self._moe_cv_score))

        self._winner_name, self._winner_model, _ = max(candidates, key=lambda t: t[2])
        self._winner_val = (
            self._tuned_val
            if self._winner_name == self._best_name
            else compute_oof_metrics(self._winner_model, self._x_df, self._y, self.cv)
        )
        print(
            f"\n[WIN] Ganador: {self._winner_name} | OOF acc={self._winner_val['val_accuracy']:.4f}"
        )

        self._error_tables = analyze_errors(
            self._x_df,
            self._y,
            pd.Series(self._winner_val["y_pred"], index=self._y.index),
        )
        self._best_threshold, self._threshold_acc = optimize_threshold(
            self._y, self._winner_val["y_proba"]
        )
        gain = round(self._threshold_acc - self._winner_val["val_accuracy"], 4)
        print(
            f"  Umbral optimo: {self._best_threshold:.4f} -> acc: {self._threshold_acc:.4f} (ganancia: {gain:+.4f})"
        )

    def _build_metadata(self) -> None:
        """Construye el diccionario de metadata del experimento."""
        log_path = str(DOCS_DIR / "model" / "experimentation_log.md")
        exp_id = get_next_exp_id(log_path)
        feature_names = self._x_df.columns.tolist()

        features_added: list = []
        features_removed: list = []
        if self.fs.parent and self.fs.parent in FEATURE_SETS:
            parent_fs = FEATURE_SETS[self.fs.parent]
            current_cols = set(
                self.fs.numeric_features
                + self.fs.categorical_cols
                + list(self.fs.target_encode_cols)
            )
            parent_cols = set(
                parent_fs.numeric_features
                + parent_fs.categorical_cols
                + list(parent_fs.target_encode_cols)
            )
            features_added = sorted(current_cols - parent_cols)
            features_removed = sorted(parent_cols - current_cols)

        effective_acc = max(self._winner_val["val_accuracy"], self._threshold_acc)
        effective_threshold = (
            self._best_threshold
            if self._threshold_acc > self._winner_val["val_accuracy"]
            else 0.5
        )

        self._metadata = {
            "exp_id": exp_id,
            "model_name": self._winner_name,
            "feature_set_name": self.fs_name,
            "feature_set_description": self.fs.description,
            "feature_set_parent": self.fs.parent,
            "features_added": features_added,
            "features_removed": features_removed,
            "numeric_features": self.fs.numeric_features,
            "val_accuracy": effective_acc,
            "val_accuracy_default_threshold": self._winner_val["val_accuracy"],
            "val_roc_auc": self._winner_val["val_roc_auc"],
            "cv_accuracy": self._best_cv_score,
            "n_features": self._x_df.shape[1],
            "n_train_samples": self._x_df.shape[0],
            "best_params": self._best_params,
            "best_threshold": effective_threshold,
            "feature_names": feature_names,
        }

    def _save_artifacts(self) -> None:
        """Guarda artefacto, promueve a produccion y escribe docs."""
        log_path = str(DOCS_DIR / "model" / "experimentation_log.md")
        exp_id = self._metadata["exp_id"]
        feature_names = self._metadata["feature_names"]
        effective_acc = self._metadata["val_accuracy"]

        if is_duplicate_experiment(self._metadata, log_path) and MODEL_PATH.exists():
            print("\n[SKIP] Experimento identico ya registrado.")
            return

        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = (
            self._winner_name.replace(" ", "_").replace("(", "").replace(")", "")
        )
        exp_artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_{safe_name}.pkl"
        joblib.dump(self._winner_model, exp_artifact)
        print(f"  [SAVE] Artefacto: {exp_artifact}")

        current_best_acc: float | None = None
        if MODEL_METADATA.exists():
            with open(MODEL_METADATA, encoding="utf-8") as f:
                current_best_acc = json.load(f).get("val_accuracy")

        self._promoted = current_best_acc is None or effective_acc > current_best_acc

        if self._promoted:
            PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._winner_model, MODEL_PATH)
            scaler_path = get_scaler_path(self.fs_name)
            if scaler_path.exists():
                shutil.copy2(scaler_path, SCALER_PATH)
            with open(MODEL_METADATA, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2, default=str)
            label = "NUEVO MEJOR MODELO" if current_best_acc else "primer modelo"
            print(f"  [PROD] [{label}] Promovido: {MODEL_PATH}")
            create_git_tag(exp_id, self.fs_name, effective_acc)
        else:
            print(
                f"  [--] No promovido -- {effective_acc:.4f} no supera {current_best_acc:.4f}"
            )

        cards_dir = str(DOCS_DIR / "model" / "cards")
        append_experiment_log(
            metadata=self._metadata,
            path=log_path,
            exp_id=exp_id,
            promoted=self._promoted,
            context=ExperimentContext(
                current_best_acc=current_best_acc,
                cv_results=self._cv_results,
                features_added=self._metadata["features_added"],
                features_removed=self._metadata["features_removed"],
            ),
        )
        write_experiment_card(
            metadata=self._metadata,
            feature_names=feature_names,
            exp_id=exp_id,
            cards_dir=cards_dir,
            promoted=self._promoted,
            current_best_acc=current_best_acc,
        )
        if self._promoted:
            write_model_card(
                metadata=self._metadata,
                feature_names=feature_names,
                path=str(DOCS_DIR / "model" / "model_card.md"),
            )

    def _log_mlflow(self) -> None:
        """Registra parametros y metricas escalares en MLflow."""
        mlflow.set_tag("feature_set", self.fs_name)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.log_param("feature_set", self.fs_name)
        mlflow.log_param("winner_model", str(self._winner_name)[:250])
        mlflow.log_param("exp_id", str(self._metadata.get("exp_id", "")))
        mlflow.log_metric(
            "val_accuracy", float(self._metadata.get("val_accuracy", 0.0))
        )
        mlflow.log_metric("val_roc_auc", float(self._metadata.get("val_roc_auc", 0.0)))
        mlflow.log_metric("cv_accuracy", float(self._metadata.get("cv_accuracy", 0.0)))
        for k, v in list((self._best_params or {}).items())[:40]:
            mlflow.log_param(f"bp_{k}", str(v)[:250])

    def run(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo y retorna el contexto del experimento.

        Returns:
            Diccionario con modelos, metricas y tablas para ReportFactory y tests.
        """
        print(f"\n{'='*60}")
        print(f"TrainingPipeline | {self.fs_name}")
        print(f"{'='*60}")

        with mlrun(f"03_Train_{self.fs_name}"):
            self._load_data()
            self._evaluate()
            self._tune()
            self._build_ensembles()
            self._select_and_evaluate_winner()

            print("\n[FIT] Re-entrenando sobre dataset completo...")
            self._winner_model.fit(self._x_df, self._y)

            self._build_metadata()
            self._save_artifacts()
            self._log_mlflow()

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)

            shap_plots: dict = {}
            if self.compute_shap:
                print("\n[SHAP] Generando analisis SHAP...")
                shap_plots = compute_shap_plots(
                    model=self._winner_model,
                    x_train=self._x_df,
                    val_preds=ValPredictions(
                        X=self._x_df,
                        y_true=self._y,
                        y_proba=self._winner_val["y_proba"],
                    ),
                    feature_names=self._metadata["feature_names"],
                )

        return {
            "fs_name": self.fs_name,
            "cv_results": self._cv_results,
            "best_name": self._best_name,
            "best_params": self._best_params,
            "tuned_val": self._tuned_val,
            "stacking_val": self._stacking_val,
            "moe_val": self._moe_val,
            "winner_name": self._winner_name,
            "winner_val": self._winner_val,
            "winner_model": self._winner_model,
            "feature_names": self._metadata["feature_names"],
            "error_tables": self._error_tables,
            "best_threshold": self._best_threshold,
            "threshold_acc": self._threshold_acc,
            "top_names": self._top_names,
            "metadata": self._metadata,
            "promoted": self._promoted,
            "shap_plots": shap_plots,
        }
