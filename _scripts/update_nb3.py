"""Actualiza NB03 para usar funciones de engineering.py y referencias cruzadas."""
import json

path = (
    r"C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects"
    r"\Projects\Classification\prueba_plantilla_de_datos\notebooks\exploratory"
    r"\03.feature_engineering.ipynb"
)

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def lines(*args):
    return list(args)


# ── CELL [0]: Header con tabla de decisiones ──────────────────────────────
cells[0]["source"] = lines(
    "# **Feature Engineering — Spaceship Titanic**\n",
    "\n",
    "**Objetivo:** Ejecutar las transformaciones y construcción de features definidas en los notebooks anteriores.\n",
    "\n",
    "**Fundamentos — decisiones tomadas en NB02:**\n",
    "\n",
    "| Feature | Acción | Justificación estadística (NB02) |\n",
    "|---|---|---|\n",
    "| `Cabin` | Extraer `Deck` / `CabinNumber` / `Side` | Deck: chi²=392.3, Side: chi²=91.1 (p<0.001) |\n",
    "| `PassengerId` | Extraer `GroupSize` | chi²=145.3 (p<0.001), patrón no lineal |\n",
    "| Gastos individuales | Crear `TotalSpending_Log` | r=-0.469 vs r=-0.200 original |\n",
    "| `Age` | Categorizar en rangos etarios | Mann-Whitney p<0.001 |\n",
    "| `VIP` | **Descartar** | corr=-0.037, solo 199 positivos de 8,490 |\n",
    "| `CryoSleep` | Conservar (mayor diferenciador) | chi²=1,859.6 — señal más fuerte del dataset |\n",
    "\n",
    "**Cadena de notebooks:**  \n",
    "[NB01](01.Initial_exploration.ipynb) → [NB02](02.Analisis_Target.ipynb) → **NB03 (este)** → [NB04](04.Model_Training.ipynb)\n",
    "\n",
    "**Input:** `data/raw/train.csv`  \n",
    "**Output:** `data/processed/train_features_scaled.csv`\n",
)

# ── CELL [2]: Imports + sys.path ──────────────────────────────────────────
cells[2]["source"] = lines(
    "import sys\n",
    "sys.path.insert(0, '../../')  # Agrega la raíz del proyecto al path\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import warnings\n",
    "\n",
    "from src.features.engineering import (\n",
    "    extract_cabin_features,\n",
    "    extract_group_features,\n",
    "    create_spending_features,\n",
    "    create_age_features,\n",
    ")\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)\n",
)

# ── CELL [5]: Missing values intro → referencia NB01 ──────────────────────
cells[5]["source"] = lines(
    "## **3. Verificación de Datos Faltantes**\n",
    "\n",
    "> **NB01** identificó 12 columnas con ~2% de nulos cada una (patrón uniforme, no sistemático).\n",
    "> Verificamos el estado antes de proceder y aplicamos la estrategia acordada:\n",
    ">\n",
    "> - Variables categóricas (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`) → `'Unknown'`\n",
    "> - Variables de gasto (`RoomService`, etc.) → `0` (nulo = sin gasto registrado)\n",
    "> - `Age` → eliminar filas (179 registros, 2.06%) para evitar sesgo en categorización\n",
)

# ── CELL [7]: Section 4 header ────────────────────────────────────────────
cells[7]["source"] = lines(
    "## **4. Ingeniería de Features**\n",
    "\n",
    "Las transformaciones de esta sección ejecutan las decisiones de NB02.  \n",
    "La lógica de cada función vive en `src/features/engineering.py`.\n",
    "\n",
    "### **4.1 Extraer Features de Cabin**\n",
)

# ── CELL [9]: Extract Cabin → use function ────────────────────────────────
cells[9]["source"] = lines(
    "# NB02: Deck (chi²=392.3) y Side (chi²=91.1) → discriminadores significativos\n",
    "df_clean = extract_cabin_features(df_clean)\n",
    "\n",
    "print('Features extraídas de Cabin: Deck, CabinNumber, Side')\n",
    "print('Distribución de Deck:')\n",
    "print(df_clean['Deck'].value_counts())\n",
)

# ── CELL [12]: Extract Group → use function ───────────────────────────────
cells[12]["source"] = lines(
    "# NB02: GroupSize (chi²=145.3) — grupos de 3-6 con mayor tasa de transporte\n",
    "df_clean = extract_group_features(df_clean)\n",
    "\n",
    "print('Features de grupo creadas: TravelGroup, GroupSize')\n",
    "print('Distribución de GroupSize:')\n",
    "print(df_clean['GroupSize'].value_counts().sort_index())\n",
)

# ── CELL [14]: Spending features → use function ───────────────────────────
cells[14]["source"] = lines(
    "# NB02: TotalSpending_Log es el feature más correlacionado: r=-0.469 vs r=-0.200 original\n",
    "df_clean = create_spending_features(df_clean)\n",
    "\n",
    "print('Features de gasto: TotalSpending, HasSpending, SpendingCategories, TotalSpending_Log')\n",
    "print(df_clean[['TotalSpending', 'TotalSpending_Log']].describe())\n",
)

# ── CELL [17]: Age categorization → use function ──────────────────────────
cells[17]["source"] = lines(
    "# NB02: Age significativa (Mann-Whitney p<0.001); se categoriza para capturar no-linealidad\n",
    "# Rangos: Child(<13), Teen(13-18), YoungAdult(18-30), Adult(30-60), Senior(60+)\n",
    "df_clean = create_age_features(df_clean)\n",
    "\n",
    "print('Categorías de edad:')\n",
    "print(df_clean['AgeCategory'].value_counts())\n",
)

# ── CELL [20]: Handle missing → add NB01/NB02 reference ──────────────────
cells[20]["source"] = lines(
    "# Estrategia de nulos — NB01: distribución uniforme ~2%; NB02: nulos en CryoSleep/VIP son informativos\n",
    "# Categóricas → 'Unknown' (preservar como categoría en el modelo)\n",
    "df_clean['HomePlanet'].fillna('Unknown', inplace=True)\n",
    "df_clean['CryoSleep'].fillna('Unknown', inplace=True)\n",
    "df_clean['Destination'].fillna('Unknown', inplace=True)\n",
    "\n",
    "# Variables de gasto → 0 (nulo = no utilizó el servicio)\n",
    "spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']\n",
    "for col in spending_cols:\n",
    "    df_clean[col].fillna(0, inplace=True)\n",
    "\n",
    "print('Imputaciones aplicadas:')\n",
    "for col in ['HomePlanet', 'CryoSleep', 'Destination']:\n",
    "    n = df_clean[col].value_counts().get('Unknown', 0)\n",
    "    print(f'  {col}: {n} registros con Unknown')\n",
    "print(f'  Spending ({len(spending_cols)} cols): nulos → 0')\n",
)

# ── CELL [30]: Outlier strategy → reference NB02 ─────────────────────────
cells[30]["source"] = lines(
    "# NB02: outliers en gasto son señal real — Mann-Whitney p<0.001 para todas las variables de gasto\n",
    "# No se eliminan; la transformación log1p (ya aplicada en sección 4.3) reduce la asimetría\n",
    "print('Estrategia de outliers:')\n",
    "print('  Variables de gasto: MANTENER (valores extremos discriminan, confirmado en NB02)')\n",
    "print('  Age: MANTENER (edades extremas son válidas)')\n",
    "print('  Transformación log1p ya aplicada a TotalSpending en sección 4.3')\n",
    "print()\n",
    "print('  Mejora de correlación con target (NB02):')\n",
    "print('    TotalSpending     → r = -0.200')\n",
    "print('    TotalSpending_Log → r = -0.469')\n",
    "print(f'\\n  Distribución post-transformación:')\n",
    "print(f\"    TotalSpending     → media: {df_clean['TotalSpending'].mean():.2f}, std: {df_clean['TotalSpending'].std():.2f}\")\n",
    "print(f\"    TotalSpending_Log → media: {df_clean['TotalSpending_Log'].mean():.2f}, std: {df_clean['TotalSpending_Log'].std():.2f}\")\n",
)

# ── CELL [38]: Feature selection → add NB02 references ───────────────────
cells[38]["source"] = lines(
    "# Feature selection basada en el análisis estadístico de NB02\n",
    "features_to_drop = [\n",
    "    'PassengerId', 'Name', 'Cabin', 'TravelGroup',  # Originales ya procesadas/extraídas\n",
    "    'CryoSleep',        # Reemplazada por CryoSleep_Encoded\n",
    "    'VIP',              # NB02: corr=-0.037, solo 199 positivos → descartada\n",
    "    'Side',             # NB02: Deck (chi²=392) > Side (chi²=91); se reduce dimensionalidad\n",
    "    'TotalSpending',    # Reemplazada por TotalSpending_Log (r: -0.200 → -0.469)\n",
    "]\n",
    "\n",
    "df_final = df_encoded.drop(columns=[col for col in features_to_drop if col in df_encoded.columns])\n",
    "\n",
    "target = 'Transported'\n",
    "X = df_final.drop(columns=[target])\n",
    "y = df_final[target]\n",
    "\n",
    "print('Features finales para modelado:')\n",
    "print(f'  Total features: {X.shape[1]}')\n",
    "print(f'  Muestras: {X.shape[0]:,}')\n",
    "print(f'\\nPrimeras 10 features: {X.columns.tolist()[:10]}')\n",
)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("NB3 actualizado exitosamente")
print(f"Total celdas: {len(cells)}")
