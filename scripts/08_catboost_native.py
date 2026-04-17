"""
08_catboost_native.py - CatBoost con categoricas nativas

En lugar de OHE + TE manual, pasa Deck, HomePlanet, Destination,
AgeCategory y LastName directamente a CatBoost como cat_features.
CatBoost usa Ordered Target Statistics internamente (fold-aware, sin leakage).

18 features: 13 numericas + 5 categoricas nativas.
Sin necesidad de StandardScaler (no beneficia a arboles).

Ejecutar:
    python scripts/08_catboost_native.py
    python scripts/08_catboost_native.py --n-iter 50
"""
import argparse
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config.settings import EXPERIMENTS_DIR, SUBMISSIONS_DIR, TEST_RAW, TRAIN_RAW
from src.features.engineering import encode_cryosleep, encode_side
from src.features.feature_sets.pipelines import _pipeline_fs017
from src.models.training import optimize_threshold
from src.reports.experiments.log import get_next_exp_id

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columnas que CatBoost manejara con Ordered Target Statistics
CAT_FEATURES = ["Deck", "HomePlanet", "Destination", "AgeCategory", "LastName"]

# Columnas numericas (sin TE ni OHE derivadas)
NUMERIC_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNumber", "TotalSpending_Log", "SpendingCategories",
    "HasSpending", "CryoSleep_Encoded", "Side_Encoded",
]

# Columnas a eliminar del DataFrame
COLS_TO_DROP = ["PassengerId", "Name", "Cabin", "TravelGroup", "CryoSleep",
                "VIP", "Side", "TotalSpending", "Transported"]


def _preprocess(df: pd.DataFrame, is_test: bool = False):
    """Aplica transformaciones base y retorna (X, y) o X si is_test.

    Args:
        df: DataFrame crudo (train.csv o test.csv).
        is_test: Si True usa test_pipeline (imputa Age NaN con mediana) y no
            retorna y.

    Returns:
        Si is_test=False: tupla (X DataFrame, y Series).
        Si is_test=True: X DataFrame.
    """
    from src.features.feature_sets import FEATURE_SETS  # pylint: disable=import-outside-toplevel
    from src.features.constants import TARGET  # pylint: disable=import-outside-toplevel
    fs = FEATURE_SETS["fs-017_lastname_te"]

    out = fs.test_pipeline(df) if is_test else _pipeline_fs017(df)

    out["CryoSleep_Encoded"] = out["CryoSleep"].apply(encode_cryosleep)
    out["Side_Encoded"] = out["Side"].apply(encode_side)

    # Convertir categoricas a string (CatBoost necesita str o int para cat_features)
    for col in CAT_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)

    if not is_test:
        y = out[TARGET].astype(int)
        X = out[[c for c in NUMERIC_FEATURES + CAT_FEATURES if c in out.columns]]
        return X, y

    return out[[c for c in NUMERIC_FEATURES + CAT_FEATURES if c in out.columns]]


def _make_pool(X: pd.DataFrame, y: pd.Series | None = None) -> Pool:
    """Crea un CatBoost Pool con las cat_features definidas."""
    return Pool(X, label=y, cat_features=CAT_FEATURES)


def _catboost_cv_score(params: dict, X: pd.DataFrame, y: pd.Series,
                       cv: StratifiedKFold) -> float:
    """Evalua CatBoost con CV manual usando Pool (preserva cat_features)."""
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(_make_pool(X_tr, y_tr), eval_set=_make_pool(X_val, y_val),
                  early_stopping_rounds=50, verbose=False)
        proba = model.predict_proba(_make_pool(X_val))[:, 1]
        scores.append(accuracy_score(y_val, (proba >= 0.5).astype(int)))
    return float(np.mean(scores))


def _tune(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold,
          n_iter: int, study_db: str) -> dict:
    """Optuna TPE para CatBoost con cat_features.

    Persiste el estudio en SQLite para poder reanudar si el script falla.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
        }
        return _catboost_cv_score(params, X, y, cv)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{study_db}",
        study_name="catboost_native",
        load_if_exists=True,
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_iter - completed)
    if remaining == 0:
        print(f"  Estudio ya tiene {completed} trials completos. Saltando tuning.")
    else:
        print(f"  Continuando desde {completed} trials. Ejecutando {remaining} mas...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    print(f"  Mejor CV tuneado: {study.best_value:.4f}")
    return study.best_params


def _oof_proba(params: dict, X: pd.DataFrame, y: pd.Series,
               cv: StratifiedKFold) -> np.ndarray:
    """Genera probabilidades OOF con CV manual."""
    proba = np.zeros(len(X))
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(_make_pool(X_tr, y_tr))
        proba[val_idx] = model.predict_proba(_make_pool(X_val))[:, 1]
    return proba


def main() -> None:
    """Entrena CatBoost con categoricas nativas y genera submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_exp_id("docs/model/experimentation_log.md")

    print("=" * 60)
    print("08_catboost_native.py -- CatBoost con categoricas nativas")
    print(f"  Cat features: {CAT_FEATURES}")
    print(f"  Num features: {len(NUMERIC_FEATURES)}")
    print(f"  Exp ID      : {exp_id}")
    print(f"  n_iter      : {args.n_iter}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    X, y = _preprocess(df_raw)
    print(f"\nDataset: {X.shape[0]:,} filas x {X.shape[1]} features")
    print(f"  Numericas : {NUMERIC_FEATURES}")
    print(f"  Categoricas: {CAT_FEATURES}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Tuning (con persistencia en SQLite para reanudar si falla)
    study_db = str(EXPERIMENTS_DIR / f"exp-{exp_id}_catboost_native_study.db")
    params_cache = EXPERIMENTS_DIR / f"exp-{exp_id}_catboost_native_best_params.json"

    if params_cache.exists():
        print(f"\n[TUNE] Cargando best_params desde cache: {params_cache.name}")
        with open(params_cache, encoding="utf-8") as f:
            best_params = json.load(f)
        print(f"  CV (cache): {best_params.pop('_cv_score', '?')}")
    else:
        print("\n[TUNE] Tuneando CatBoost con Optuna...")
        best_params = _tune(X, y, cv, args.n_iter, study_db)
        best_params["random_seed"] = 42
        # Guardar antes de OOF para no perder en caso de error
        cache_data = {**best_params, "_cv_score": best_params.get("_cv_score", "ver study db")}
        with open(params_cache, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Best params guardados: {params_cache.name}")

    # OOF honesto
    print("\n[OOF] Calculando predicciones out-of-fold...")
    oof_p = _oof_proba(best_params, X, y, cv)
    oof_acc = round(float(accuracy_score(y, (oof_p >= 0.5).astype(int))), 4)
    oof_roc = round(float(roc_auc_score(y, oof_p)), 4)
    threshold, thr_acc = optimize_threshold(y, oof_p)
    print(f"  OOF acc={oof_acc:.4f} | ROC-AUC={oof_roc:.4f} | thr={threshold:.4f} -> {thr_acc:.4f}")

    # Refit final sobre dataset completo
    print("\n[FIT] Entrenando modelo final sobre dataset completo...")
    final_model = CatBoostClassifier(**best_params, verbose=0, allow_writing_files=False)
    final_model.fit(_make_pool(X, y))

    artifact = EXPERIMENTS_DIR / f"exp-{exp_id}_CatBoost_native.pkl"
    joblib.dump(final_model, artifact)
    print(f"  Guardado: {artifact.name}")

    # Metadata
    meta = {
        "exp_id": exp_id,
        "model": "CatBoost_native",
        "cat_features": CAT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_names": NUMERIC_FEATURES + CAT_FEATURES,
        "oof_acc": oof_acc,
        "oof_acc_with_threshold": thr_acc,
        "oof_roc_auc": oof_roc,
        "threshold": threshold,
        "best_params": best_params,
        "n_train_samples": len(X),
    }
    meta_path = EXPERIMENTS_DIR / f"exp-{exp_id}_catboost_native_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Submission
    print("\n[PREDICT] Generando submission...")
    df_test = pd.read_csv(TEST_RAW)
    test_ids = df_test["PassengerId"].copy()
    X_test = _preprocess(df_test, is_test=True)

    y_proba_test = final_model.predict_proba(_make_pool(X_test))[:, 1]
    predictions = (y_proba_test >= threshold).astype(bool)

    submission = pd.DataFrame({
        "PassengerId": test_ids.values,
        "Transported": predictions,
    })
    sub_path = SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"
    submission.to_csv(sub_path, index=False)

    n_true = int(submission["Transported"].sum())
    print(f"  Submission: {sub_path.name}")
    print(f"  Distribucion: {n_true} True ({100*n_true/len(submission):.1f}%) | {len(submission)-n_true} False")
    print("\n[OK] Completado.")


if __name__ == "__main__":
    main()
