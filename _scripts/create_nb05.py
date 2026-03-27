"""Crea NB05 — Predictions para Spaceship Titanic."""
import json
import uuid

output = (
    r"C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects"
    r"\Projects\Classification\prueba_plantilla_de_datos\notebooks\exploratory"
    r"\05.Predictions.ipynb"
)


def cid():
    return str(uuid.uuid4())[:8]


def md(lines):
    return {"cell_type": "markdown", "id": cid(), "metadata": {}, "source": lines}


def code(lines):
    return {
        "cell_type": "code",
        "id": cid(),
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [

        # ── 0: Header
        md([
            "# **Predictions — Spaceship Titanic**\n",
            "\n",
            "**Objetivo:** Generar predicciones sobre `test.csv` usando el mejor modelo entrenado.\n",
            "\n",
            "El pipeline de preprocesamiento se aplica desde cero sobre los datos crudos  \n",
            "para garantizar consistencia exacta con el entrenamiento (mismo scaler fit en train).\n",
            "\n",
            "**Input:**\n",
            "- `data/raw/test.csv` — 4,277 registros a predecir (sin columna `Transported`)\n",
            "- `models/best_advanced_model.pkl` o `models/best_model.pkl` — modelo entrenado\n",
            "\n",
            "**Output:** `data/processed/submission.csv`\n",
            "```\n",
            "PassengerId,Transported\n",
            "0013_01,True\n",
            "0018_01,False\n",
            "...\n",
            "```\n",
        ]),

        # ── 1: Config
        md(["## **1. Configuración**\n"]),
        code([
            "# ── Rutas (ajustar si se usa un modelo diferente)\n",
            "TRAIN_RAW_PATH   = '../../data/raw/train.csv'\n",
            "TEST_RAW_PATH    = '../../data/raw/test.csv'\n",
            "SUBMISSION_PATH  = '../../data/processed/submission.csv'\n",
            "\n",
            "# Selección de modelo: prioriza el avanzado si existe\n",
            "ADVANCED_MODEL_PATH = '../../models/best_advanced_model.pkl'\n",
            "FALLBACK_MODEL_PATH = '../../models/best_model.pkl'\n",
            "\n",
            "# Features numéricas a escalar (mismas que NB03)\n",
            "NUMERIC_FEATURES = [\n",
            "    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',\n",
            "    'GroupSize', 'CabinNumber', 'TotalSpending_Log', 'SpendingCategories',\n",
            "]\n",
            "\n",
            "# Features a eliminar antes del modelado (mismas que NB03)\n",
            "FEATURES_TO_DROP = [\n",
            "    'PassengerId', 'Name', 'Cabin', 'TravelGroup',\n",
            "    'CryoSleep', 'VIP', 'Side', 'TotalSpending',\n",
            "]\n",
            "\n",
            "# Columnas categóricas para OHE (mismo orden que NB03)\n",
            "CATEGORICAL_COLS = ['HomePlanet', 'Destination', 'Deck', 'AgeCategory']\n",
        ]),

        # ── 2: Librerías
        md(["## **2. Librerías**\n"]),
        code([
            "import sys\n",
            "sys.path.insert(0, '../../')\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import joblib\n",
            "import os\n",
            "import warnings\n",
            "\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "from src.features.engineering import build_feature_set\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
        ]),

        # ── 3: Cargar modelo
        md(["## **3. Cargar Modelo**\n"]),
        code([
            "if os.path.exists(ADVANCED_MODEL_PATH):\n",
            "    model = joblib.load(ADVANCED_MODEL_PATH)\n",
            "    model_path_used = ADVANCED_MODEL_PATH\n",
            "elif os.path.exists(FALLBACK_MODEL_PATH):\n",
            "    model = joblib.load(FALLBACK_MODEL_PATH)\n",
            "    model_path_used = FALLBACK_MODEL_PATH\n",
            "else:\n",
            "    raise FileNotFoundError(\n",
            "        f'No se encontró ningún modelo en:\\n'\n",
            "        f'  {ADVANCED_MODEL_PATH}\\n'\n",
            "        f'  {FALLBACK_MODEL_PATH}\\n'\n",
            "        f'Ejecutar NB04 o NB04.1 primero.'\n",
            "    )\n",
            "\n",
            "print(f'Modelo cargado: {model_path_used}')\n",
            "print(f'Tipo: {type(model).__name__}')\n",
        ]),

        # ── 4: Cargar y preprocesar datos
        md([
            "## **4. Preprocesamiento**\n",
            "\n",
            "Se aplica el **mismo pipeline exacto que NB03** sobre ambos datasets.\n",
            "El StandardScaler se ajusta únicamente sobre el set de entrenamiento.\n",
        ]),
        code([
            "# Cargar datos crudos\n",
            "df_train_raw = pd.read_csv(TRAIN_RAW_PATH)\n",
            "df_test_raw  = pd.read_csv(TEST_RAW_PATH)\n",
            "\n",
            "# Guardar PassengerId del test para el output final\n",
            "test_ids = df_test_raw['PassengerId'].copy()\n",
            "\n",
            "print(f'Train raw: {df_train_raw.shape}')\n",
            "print(f'Test  raw: {df_test_raw.shape}')\n",
            "print(f'Test PassengerId sample: {test_ids.head(3).tolist()}')\n",
        ]),
        code([
            "# Feature engineering (funciones de src/features/engineering.py)\n",
            "df_train_fe = build_feature_set(df_train_raw)\n",
            "df_test_fe  = build_feature_set(df_test_raw)\n",
            "\n",
            "print(f'Train tras FE: {df_train_fe.shape}')\n",
            "print(f'Test  tras FE: {df_test_fe.shape}')\n",
        ]),
        code([
            "# Label encoding (mismo que NB03 cell [34])\n",
            "label_map_cryo = {'True': 1, 'False': 0, True: 1, False: 0, 'Unknown': -1}\n",
            "label_map_side = {'P': 0, 'S': 1, 'Unknown': -1}\n",
            "\n",
            "for df in [df_train_fe, df_test_fe]:\n",
            "    df['CryoSleep_Encoded'] = df['CryoSleep'].map(label_map_cryo)\n",
            "    df['Side_Encoded']      = df['Side'].map(label_map_side)\n",
            "\n",
            "print('Label encoding aplicado: CryoSleep_Encoded, Side_Encoded')\n",
        ]),
        code([
            "# One-Hot Encoding (mismo que NB03 cell [36])\n",
            "df_train_enc = pd.get_dummies(\n",
            "    df_train_fe, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, drop_first=False\n",
            ")\n",
            "df_test_enc = pd.get_dummies(\n",
            "    df_test_fe, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, drop_first=False\n",
            ")\n",
            "\n",
            "print(f'Train tras OHE: {df_train_enc.shape}')\n",
            "print(f'Test  tras OHE: {df_test_enc.shape}')\n",
        ]),
        code([
            "# Feature selection (mismo que NB03 cell [38])\n",
            "X_train_raw = df_train_enc.drop(\n",
            "    columns=[c for c in FEATURES_TO_DROP + ['Transported'] if c in df_train_enc.columns]\n",
            ")\n",
            "y_train = df_train_enc['Transported']\n",
            "\n",
            "X_test_raw = df_test_enc.drop(\n",
            "    columns=[c for c in FEATURES_TO_DROP if c in df_test_enc.columns]\n",
            ")\n",
            "\n",
            "# Alinear columnas: test debe tener exactamente las mismas columnas que train\n",
            "# (get_dummies puede generar columnas distintas si hay categorías ausentes en test)\n",
            "X_test_aligned = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)\n",
            "\n",
            "cols_only_train = set(X_train_raw.columns) - set(X_test_raw.columns)\n",
            "cols_only_test  = set(X_test_raw.columns)  - set(X_train_raw.columns)\n",
            "if cols_only_train:\n",
            "    print(f'Columnas en train sin test (rellenadas con 0): {cols_only_train}')\n",
            "if cols_only_test:\n",
            "    print(f'Columnas en test sin train (descartadas): {cols_only_test}')\n",
            "\n",
            "print(f'X_train final: {X_train_raw.shape}')\n",
            "print(f'X_test  final: {X_test_aligned.shape}')\n",
        ]),
        code([
            "# Escalado: fit en train, transform en test\n",
            "numeric_present = [c for c in NUMERIC_FEATURES if c in X_train_raw.columns]\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = X_train_raw.copy()\n",
            "X_train_scaled[numeric_present] = scaler.fit_transform(X_train_raw[numeric_present])\n",
            "\n",
            "X_test_scaled = X_test_aligned.copy()\n",
            "X_test_scaled[numeric_present] = scaler.transform(X_test_aligned[numeric_present])\n",
            "\n",
            "print(f'Escalado aplicado a {len(numeric_present)} features numéricas')\n",
            "print(f'X_test_scaled shape: {X_test_scaled.shape}')\n",
        ]),

        # ── 5: Predicción
        md(["## **5. Predicción**\n"]),
        code([
            "predictions_raw = model.predict(X_test_scaled)\n",
            "\n",
            "# Convertir a boolean (True/False)\n",
            "predictions_bool = predictions_raw.astype(bool)\n",
            "\n",
            "print(f'Predicciones generadas: {len(predictions_bool):,}')\n",
            "unique, counts = np.unique(predictions_bool, return_counts=True)\n",
            "for val, cnt in zip(unique, counts):\n",
            "    print(f'  {val}: {cnt:,} ({cnt/len(predictions_bool):.1%})')\n",
        ]),

        # ── 6: Verificación
        md([
            "## **6. Verificación del Output**\n",
            "\n",
            "Validamos el formato antes de guardar:\n",
            "- Exactamente 4,277 registros\n",
            "- Columnas: PassengerId, Transported\n",
            "- Valores de Transported: True / False (no 0/1, no numérico)\n",
        ]),
        code([
            "submission = pd.DataFrame({\n",
            "    'PassengerId': test_ids.values,\n",
            "    'Transported': predictions_bool,\n",
            "})\n",
            "\n",
            "# Validaciones\n",
            "assert len(submission) == 4277, f'Se esperaban 4,277 registros, se obtuvieron {len(submission)}'\n",
            "assert list(submission.columns) == ['PassengerId', 'Transported'], 'Columnas incorrectas'\n",
            "assert submission['Transported'].dtype == bool, 'Transported debe ser bool'\n",
            "\n",
            "print('Validaciones OK')\n",
            "print(f'Total registros : {len(submission):,}')\n",
            "print(f'Columnas        : {list(submission.columns)}')\n",
            "print(f'Tipo Transported: {submission[\"Transported\"].dtype}')\n",
            "print()\n",
            "print('Muestra del output:')\n",
            "print(submission.head(5).to_string(index=False))\n",
        ]),

        # ── 7: Guardar
        md(["## **7. Guardar Submission**\n"]),
        code([
            "os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)\n",
            "submission.to_csv(SUBMISSION_PATH, index=False)\n",
            "\n",
            "print(f'Submission guardado en: {SUBMISSION_PATH}')\n",
            "print(f'Registros: {len(submission):,} (esperados: 4,277)')\n",
            "print()\n",
            "print('Verificación del archivo:')\n",
            "check = pd.read_csv(SUBMISSION_PATH)\n",
            "print(check.head(3).to_string(index=False))\n",
            "print(f'...  ({len(check):,} registros total)')\n",
        ]),

    ],
}

with open(output, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"NB05 creado: {output}")
print(f"Celdas: {len(nb['cells'])}")
