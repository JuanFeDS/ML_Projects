"""
Script de Feature Engineering — Spaceship Titanic.

Replica la logica de NB03: aplica el pipeline del feature set seleccionado,
encoding y escalado. Genera los datasets procesados y el scaler serializado.

Ejecutar desde la raiz del proyecto:
    python scripts/02_features.py
    python scripts/02_features.py --feature-set fs-002_cryo_interactions

Genera (con {fs} = nombre del feature set):
    data/processed/train_features_{fs}.csv   (X + target, sin escalar)
    data/processed/train_scaled_{fs}.csv     (X escalado + target)
    models/scaler_{fs}.pkl
    reports/02_features.md
    reports/02_features.html
"""
import argparse
import sys

sys.path.insert(0, ".")  # scripts run from project root
sys.stdout.reconfigure(encoding='utf-8')
# pylint: disable=wrong-import-position

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.settings import (
    MODELS_DIR,
    REPORTS_DIR,
    TRAIN_RAW,
    get_scaler_path,
    get_target_encoder_path,
    get_train_features,
    get_train_scaled,
)
from src.features.constants import TARGET
from src.features.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS
from src.reports.builder import HTMLReport, MarkdownReport
from src.reports.feature_plots import (
    age_scale_compare,
    derived_feature_double_bar,
    total_spending_compare,
)

_ENCODED_COLS = ["CryoSleep", "Side"]


def _encode_cryosleep(val) -> int:
    """Codifica CryoSleep a entero: True->1, False->0, Unknown->-1."""
    if val in (True, "True"):
        return 1
    if val in (False, "False"):
        return 0
    return -1


def _encode_side(val) -> int:
    """Codifica Side a entero: P->0, S->1, Unknown->-1."""
    if val == "P":
        return 0
    if val == "S":
        return 1
    return -1


def _parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Feature Engineering — Spaceship Titanic")
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
    """Ejecuta el pipeline completo de feature engineering."""
    args = _parse_args()
    fs_name = args.feature_set
    fs = FEATURE_SETS[fs_name]

    train_features_path = get_train_features(fs_name)
    train_scaled_path = get_train_scaled(fs_name)
    scaler_pkl_path = get_scaler_path(fs_name)

    print("=" * 60)
    print("02_features.py — Feature Engineering")
    print(f"  Feature set: {fs_name}")
    print("=" * 60)

    df_raw = pd.read_csv(TRAIN_RAW)
    shape_before = df_raw.shape
    print(f"Datos cargados: {shape_before[0]:,} filas x {shape_before[1]} columnas")

    md = MarkdownReport(title="Feature Engineering — Spaceship Titanic")
    html = HTMLReport(title="Feature Engineering — Spaceship Titanic")

    md.add_section("Feature Set")
    md.add_metric("Nombre", fs_name)
    md.add_metric("Descripcion", fs.description)
    if fs.parent:
        md.add_metric("Hereda de", fs.parent)

    md.add_section("Contexto")
    md.add_text(
        f"Pipeline de feature engineering para el feature set `{fs_name}`. "
        "Ejecuta encoding y escalado estandar sobre las features resultantes."
    )
    html.add_intro(
        f"Feature set: <b>{fs_name}</b><br>"
        f"{fs.description}<br><br>"
        "A partir del dataset crudo se aplica el pipeline de transformacion definido. "
        "El resultado es un dataset listo para entrenamiento sin valores nulos."
    )

    # ------------------------------------------------------------------
    # 1. Pipeline del feature set
    # ------------------------------------------------------------------
    print(f"\n[1] Ejecutando pipeline ({fs_name})...")
    df = fs.pipeline(df_raw)
    rows_dropped = shape_before[0] - df.shape[0]
    print(f"  Shape despues del pipeline: {df.shape}")
    print(f"  Filas eliminadas (Age nulo): {rows_dropped}")

    md.add_section("Paso 1 — Pipeline del feature set")
    md.add_metric("Shape entrada", f"{shape_before[0]:,} x {shape_before[1]}")
    md.add_metric("Shape despues", f"{df.shape[0]:,} x {df.shape[1]}")
    md.add_metric("Filas eliminadas (Age nulo)", rows_dropped)
    html.add_section("Paso 1 — Pipeline del feature set")
    html.add_text(
        f"<b>Shape entrada:</b> {shape_before[0]:,} x {shape_before[1]}<br>"
        f"<b>Shape despues:</b> {df.shape[0]:,} x {df.shape[1]}<br>"
        f"<b>Filas eliminadas (Age nulo):</b> {rows_dropped}"
    )

    # ------------------------------------------------------------------
    # 2. Label encoding
    # ------------------------------------------------------------------
    print("\n[2] Label encoding (CryoSleep, Side)...")
    df["CryoSleep_Encoded"] = df["CryoSleep"].apply(_encode_cryosleep)
    df["Side_Encoded"] = df["Side"].apply(_encode_side)

    md.add_section("Paso 2 — Label Encoding (CryoSleep, Side)")
    md.add_metric("CryoSleep_Encoded", "True->1, False->0, Unknown->-1")
    md.add_metric("Side_Encoded", "P->0, S->1, Unknown->-1")
    html.add_section("Paso 2 — Label Encoding (CryoSleep, Side)")
    html.add_text(
        "<b>CryoSleep_Encoded:</b> True->1, False->0, Unknown->-1<br>"
        "<b>Side_Encoded:</b> P->0, S->1, Unknown->-1"
    )

    # ------------------------------------------------------------------
    # 3a. Target Encoding (solo si el feature set lo requiere)
    # ------------------------------------------------------------------
    target_encoder: dict = {}
    if fs.target_encode_cols:
        print(f"\n[3a] Target Encoding para: {fs.target_encode_cols}...")
        y_raw = df[TARGET].astype(int)
        for col in fs.target_encode_cols:
            mapping = df.groupby(col)[TARGET].mean().to_dict()
            target_encoder[col] = mapping
            encoded_col = f"{col}_TE"
            df[encoded_col] = df[col].map(mapping)
            print(f"  {col} → {encoded_col}  ({len(mapping)} categorias)")

        te_path = get_target_encoder_path(fs_name)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(target_encoder, te_path)
        print(f"  Target encoder guardado: {te_path}")

        md.add_section("Paso 3a — Target Encoding")
        for col, mapping in target_encoder.items():
            md.add_metric(f"{col} → {col}_TE", str({k: round(v, 3) for k, v in mapping.items()}))
    else:
        print("\n[3a] Target Encoding: no aplica para este feature set.")

    # ------------------------------------------------------------------
    # 3b. One-Hot Encoding (columnas categoricas restantes)
    # ------------------------------------------------------------------
    print("\n[3b] One-Hot Encoding...")
    if fs.categorical_cols:
        df = pd.get_dummies(df, columns=fs.categorical_cols, drop_first=False)
    ohe_cols = [c for c in df.columns if any(
        c.startswith(cat + "_") for cat in fs.categorical_cols
    )]
    print(f"  Columnas OHE generadas: {len(ohe_cols)}")

    md.add_section("Paso 3b — One-Hot Encoding")
    md.add_metric("Columnas OHE generadas", len(ohe_cols))
    md.add_bullet_list(ohe_cols)
    html.add_section("Paso 3 — Encoding")
    html.add_text(
        f"<b>Target encoded:</b> {fs.target_encode_cols or 'ninguna'}<br>"
        f"<b>Columnas OHE ({len(ohe_cols)}):</b> "
        + ", ".join(f"<code>{c}</code>" for c in ohe_cols)
    )

    # ------------------------------------------------------------------
    # 4. Drop de columnas
    # ------------------------------------------------------------------
    print("\n[4] Dropping columnas...")
    cols_to_drop = fs.features_to_drop + [
        c for c in _ENCODED_COLS if c in df.columns
    ] + list(fs.target_encode_cols)
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)
    print(f"  Columnas eliminadas: {len(cols_to_drop_existing)}")
    print(f"  Shape tras drop: {df.shape}")

    md.add_section("Paso 4 — Drop de columnas")
    md.add_metric("Columnas eliminadas", len(cols_to_drop_existing))
    md.add_bullet_list(cols_to_drop_existing)
    md.add_metric("Shape tras drop", f"{df.shape[0]:,} x {df.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Separar X / y
    # ------------------------------------------------------------------
    print("\n[5] Separando X e y...")
    y = df[TARGET].astype(int)
    x_df = df.drop(columns=[TARGET])

    missing_num = [f for f in fs.numeric_features if f not in x_df.columns]
    if missing_num:
        raise ValueError(
            f"numeric_features faltantes en X: {missing_num}. "
            "Revisa el pipeline del feature set."
        )

    print(f"  X: {x_df.shape}  |  y: {y.shape}")

    md.add_section("Paso 5 — Separacion X / y")
    md.add_metric("Filas", f"{x_df.shape[0]:,}")
    md.add_metric("Features", x_df.shape[1])
    md.add_text("**Lista completa de features resultantes:**")
    md.add_bullet_list(list(x_df.columns))
    html.add_section("Paso 5 — Separacion X / y")
    html.add_metrics_grid([
        (f"{x_df.shape[0]:,}", "filas de entrenamiento"),
        (x_df.shape[1], "features resultantes"),
        (shape_before[1], "variables originales"),
        (rows_dropped, "filas eliminadas"),
        (x_df.isnull().sum().sum(), "nulos residuales"),
    ])
    html.add_text(f"<b>Target:</b> {y.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 6. Guardar train_features (sin escalar)
    # ------------------------------------------------------------------
    print(f"\n[6] Guardando train_features_{fs_name}.csv...")
    train_features_path.parent.mkdir(parents=True, exist_ok=True)
    train_features = x_df.copy()
    train_features[TARGET] = y
    train_features.to_csv(train_features_path, index=False)
    print(f"  Guardado: {train_features_path}  ({train_features.shape})")

    # ------------------------------------------------------------------
    # 7. StandardScaler, guardar y escalar
    # ------------------------------------------------------------------
    print("\n[7] StandardScaler...")
    scaler = StandardScaler()
    x_scaled = x_df.copy()

    bool_cols = x_scaled.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        x_scaled[bool_cols] = x_scaled[bool_cols].astype(int)

    x_scaled[fs.numeric_features] = scaler.fit_transform(x_scaled[fs.numeric_features])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_pkl_path)
    print(f"  Scaler guardado: {scaler_pkl_path}")

    scaler_info = {
        "Tipo": "StandardScaler",
        "Fit sobre": ", ".join(fs.numeric_features),
        "Medias (primeras 5)": ", ".join(f"{m:.3f}" for m in scaler.mean_[:5]),
        "Desv. std (primeras 5)": ", ".join(f"{s:.3f}" for s in scaler.scale_[:5]),
    }
    md.add_section("Paso 6 — StandardScaler")
    for k, v in scaler_info.items():
        md.add_metric(k, v)

    train_scaled = x_scaled.copy()
    train_scaled[TARGET] = y
    train_scaled_path.parent.mkdir(parents=True, exist_ok=True)
    train_scaled.to_csv(train_scaled_path, index=False)
    print(f"  Guardado: {train_scaled_path}  ({train_scaled.shape})")

    # ------------------------------------------------------------------
    # 8. Visualizaciones HTML
    # ------------------------------------------------------------------
    print("\n[8] Generando visualizaciones...")
    html.add_section("Visualizaciones")

    html.add_section("Distribuciones de features derivadas", level=3)

    deck_cols = [c for c in train_features.columns if c.startswith("Deck_")]
    if deck_cols:
        deck_series = (
            train_features[deck_cols]
            .idxmax(axis=1)
            .str.replace("Deck_", "", regex=False)
        )
        deck_counts = deck_series.value_counts().reset_index()
        deck_counts.columns = ["Deck", "count"]
        target_s = pd.Series(y.values, index=train_features.index)
        deck_rate = (
            pd.DataFrame({"Deck": deck_series, TARGET: target_s})
            .groupby("Deck")[TARGET]
            .mean()
            .reset_index()
            .rename(columns={TARGET: "tasa"})
        )
        deck_summary = deck_counts.merge(deck_rate, on="Deck")
        html.add_figure(
            derived_feature_double_bar(deck_summary, "Deck", "Deck (extraido de Cabin)"),
            title="",
        )

    age_cat_cols = [c for c in train_features.columns if c.startswith("AgeCategory_")]
    if age_cat_cols:
        age_series = (
            train_features[age_cat_cols]
            .idxmax(axis=1)
            .str.replace("AgeCategory_", "", regex=False)
        )
        age_counts = age_series.value_counts().reset_index()
        age_counts.columns = ["AgeCategory", "count"]
        age_order = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
        age_counts["order"] = age_counts["AgeCategory"].map(
            {v: i for i, v in enumerate(age_order)}
        )
        age_counts = age_counts.sort_values("order").drop(columns="order")
        target_s = pd.Series(y.values, index=train_features.index)
        age_rate = (
            pd.DataFrame({"AgeCategory": age_series, TARGET: target_s})
            .groupby("AgeCategory")[TARGET]
            .mean()
            .reset_index()
            .rename(columns={TARGET: "tasa"})
        )
        age_summary = age_counts.merge(age_rate, on="AgeCategory")
        html.add_figure(
            derived_feature_double_bar(
                age_summary, "AgeCategory", "AgeCategory (derivada de Age)"
            ),
            title="",
        )

    gs_series = train_features["GroupSize"]
    target_s = pd.Series(y.values, index=train_features.index)
    gs_rate = (
        pd.DataFrame({"GroupSize": gs_series, TARGET: target_s})
        .groupby("GroupSize")[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "tasa"})
    )
    html.add_figure(
        derived_feature_double_bar(
            gs_rate, "GroupSize", "GroupSize (extraido de PassengerId)"
        ),
        title="",
    )

    html.add_section("TotalSpending vs TotalSpending_Log", level=3)
    html.add_figure(total_spending_compare(df_raw, TARGET), title="")

    html.add_section("Distribucion pre/post escalado — Age", level=3)
    age_idx = fs.numeric_features.index("Age")
    html.add_figure(
        age_scale_compare(
            train_features["Age"],
            train_scaled["Age"],
            float(scaler.mean_[age_idx]),
            float(scaler.scale_[age_idx]),
        ),
        title="",
    )

    # ------------------------------------------------------------------
    # 9. Resumen y guardar reportes
    # ------------------------------------------------------------------
    print("\n[9] Guardando reportes...")
    md.add_section("Resumen del Pipeline")
    md.add_metric("Feature set", fs_name)
    md.add_metric("Shape entrada", f"{shape_before[0]:,} x {shape_before[1]}")
    md.add_metric("Shape final (X)", f"{x_df.shape[0]:,} x {x_df.shape[1]}")
    md.add_metric("Filas eliminadas", f"{rows_dropped} (Age nulo, 2.06%)")
    md.add_metric("Nulos residuales en X", x_df.isnull().sum().sum())
    md.add_metric("Features resultantes", x_df.shape[1])

    md.add_section("Archivos Generados")
    md.add_bullet_list([
        f"`{train_features_path}` — X + Transported, sin escalar",
        f"`{train_scaled_path}` — X escalado + Transported",
        f"`{scaler_pkl_path}` — StandardScaler serializado (joblib)",
    ])

    html.add_section("Resumen del Pipeline")
    html.add_text(
        f"<b>Feature set:</b> {fs_name}<br>"
        f"<b>Shape entrada:</b> {shape_before[0]:,} x {shape_before[1]}<br>"
        f"<b>Shape final (X):</b> {x_df.shape[0]:,} x {x_df.shape[1]}<br>"
        f"<b>Filas eliminadas (Age nulo):</b> {rows_dropped}<br>"
        f"<b>Nulos residuales en X:</b> {x_df.isnull().sum().sum()}<br>"
        f"<b>Features resultantes:</b> {x_df.shape[1]}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md.save(str(REPORTS_DIR / "02_features.md"))
    html.save(str(REPORTS_DIR / "02_features.html"))

    print("\nListo.")
    print(f"  Features CSV  -> {train_features_path}")
    print(f"  Scaled CSV    -> {train_scaled_path}")
    print(f"  Scaler PKL    -> {scaler_pkl_path}")
    print(f"  MD Report     -> {REPORTS_DIR / '02_features.md'}")
    print(f"  HTML Report   -> {REPORTS_DIR / '02_features.html'}")


if __name__ == "__main__":
    main()
