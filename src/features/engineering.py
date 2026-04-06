"""
Modulo para ingenieria de features.

Funciones para crear, transformar y seleccionar features para modelado.
"""
from typing import List, Optional

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def create_date_features(
    df: pd.DataFrame,
    date_column: str,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Crea features derivadas de una columna de fecha.

    Args:
        df: DataFrame a procesar
        date_column: Columna de fecha
        features: Features a crear ('year', 'month', 'day', 'dayofweek', 'quarter')
                 Si None, crea todas

    Returns:
        DataFrame con nuevas columnas de features temporales
    """
    df_copy = df.copy()
    date_col = pd.to_datetime(df_copy[date_column])

    all_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear']
    features_to_create = features if features else all_features

    for feature in features_to_create:
        if feature == 'year':
            df_copy[f'{date_column}_year'] = date_col.dt.year
        elif feature == 'month':
            df_copy[f'{date_column}_month'] = date_col.dt.month
        elif feature == 'day':
            df_copy[f'{date_column}_day'] = date_col.dt.day
        elif feature == 'dayofweek':
            df_copy[f'{date_column}_dayofweek'] = date_col.dt.dayofweek
        elif feature == 'quarter':
            df_copy[f'{date_column}_quarter'] = date_col.dt.quarter
        elif feature == 'dayofyear':
            df_copy[f'{date_column}_dayofyear'] = date_col.dt.dayofyear

    return df_copy


def create_binned_features(
    df: pd.DataFrame,
    column: str,
    bins: List[float],
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Crea feature categorica binneando una variable numerica.

    Args:
        df: DataFrame a procesar
        column: Columna a binnear
        bins: Limites de los bins
        labels: Etiquetas para cada bin (opcional)

    Returns:
        DataFrame con nueva columna binneada
    """
    df_copy = df.copy()
    df_copy[f'{column}_bin'] = pd.cut(df_copy[column], bins=bins, labels=labels)
    return df_copy


def create_interaction_features(
    df: pd.DataFrame,
    columns: List[str],
    operation: str = 'multiply'
) -> pd.DataFrame:
    """
    Crea features de interaccion entre columnas.

    Args:
        df: DataFrame a procesar
        columns: Lista de columnas para interaccion
        operation: Operacion ('multiply', 'add', 'divide', 'subtract')

    Returns:
        DataFrame con nueva columna de interaccion
    """
    df_copy = df.copy()

    if len(columns) != 2:
        raise ValueError("Se requieren exactamente 2 columnas para interaccion")

    col1, col2 = columns
    feature_name = f'{col1}_x_{col2}'

    if operation == 'multiply':
        df_copy[feature_name] = df_copy[col1] * df_copy[col2]
    elif operation == 'add':
        df_copy[feature_name] = df_copy[col1] + df_copy[col2]
    elif operation == 'divide':
        df_copy[feature_name] = df_copy[col1] / df_copy[col2].replace(0, np.nan)
    elif operation == 'subtract':
        df_copy[feature_name] = df_copy[col1] - df_copy[col2]

    return df_copy


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot',
    drop_first: bool = False
) -> pd.DataFrame:
    """
    Codifica variables categoricas.

    Args:
        df: DataFrame a procesar
        columns: Columnas categoricas a codificar
        method: Metodo de encoding ('onehot', 'label', 'ordinal')
        drop_first: Si True, elimina primera categoria (evita multicolinealidad)

    Returns:
        DataFrame con columnas codificadas
    """
    df_copy = df.copy()

    if method == 'onehot':
        df_copy = pd.get_dummies(
            df_copy,
            columns=columns,
            drop_first=drop_first,
            prefix=columns
        )
    elif method == 'label':
        for col in columns:
            label_encoder = LabelEncoder()
            df_copy[col] = label_encoder.fit_transform(df_copy[col].astype(str))

    return df_copy


def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'standard'
) -> pd.DataFrame:
    """
    Escala features numericas.

    Args:
        df: DataFrame a procesar
        columns: Columnas a escalar
        method: Metodo de escalado ('standard', 'minmax', 'robust')

    Returns:
        DataFrame con columnas escaladas
    """
    df_copy = df.copy()

    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy


# ---------------------------------------------------------------------------
# Spaceship Titanic — Feature Engineering especifico del proyecto
# Funciones derivadas del analisis estadistico documentado en NB02.
# ---------------------------------------------------------------------------

_SPENDING_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_CATEGORICAL_FILL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']


def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae Deck, CabinNumber y Side desde la columna Cabin.

    NB02: Deck (chi²=392.3) y Side (chi²=91.1) son discriminadores
    estadisticamente significativos (p < 0.001).

    Args:
        df: DataFrame con columna 'Cabin' en formato 'Deck/Num/Side'

    Returns:
        DataFrame con columnas 'Deck', 'CabinNumber', 'Side' anadidas
    """
    df_copy = df.copy()
    df_copy['Deck'] = df_copy['Cabin'].apply(
        lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown'
    )
    df_copy['CabinNumber'] = df_copy['Cabin'].apply(
        lambda x: int(x.split('/')[1]) if pd.notna(x) else 0
    )
    df_copy['Side'] = df_copy['Cabin'].apply(
        lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown'
    )
    return df_copy


def extract_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae TravelGroup y GroupSize desde PassengerId.

    NB02: GroupSize (chi²=145.3, p < 0.001) muestra patron no lineal
    con el target — viajeros en grupos de 3-6 tienen mayor tasa de transporte.

    Args:
        df: DataFrame con columna 'PassengerId' en formato 'GGGG_NN'

    Returns:
        DataFrame con columnas 'TravelGroup' y 'GroupSize' anadidas
    """
    df_copy = df.copy()
    df_copy['TravelGroup'] = df_copy['PassengerId'].str.split('_').str[0]
    df_copy['GroupSize'] = df_copy.groupby('TravelGroup')['TravelGroup'].transform('count')
    return df_copy


def create_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea TotalSpending, HasSpending, SpendingCategories y TotalSpending_Log.

    NB02: TotalSpending_Log es el feature mas correlacionado con el target
    (r=-0.469 vs r=-0.200 del original) gracias a la transformacion log.

    Args:
        df: DataFrame con columnas de gasto individuales

    Returns:
        DataFrame con features de gasto agregadas anadidas
    """
    df_copy = df.copy()
    df_copy['TotalSpending'] = df_copy[_SPENDING_COLS].fillna(0).sum(axis=1)
    df_copy['HasSpending'] = (df_copy['TotalSpending'] > 0).astype(int)
    df_copy['SpendingCategories'] = (df_copy[_SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    df_copy['TotalSpending_Log'] = np.log1p(df_copy['TotalSpending'])
    return df_copy


def _categorize_age(age) -> str:
    """Clasifica edad en rangos etarios."""
    if pd.isna(age):
        return 'Unknown'
    elif age < 13:
        return 'Child'
    elif age < 18:
        return 'Teen'
    elif age < 30:
        return 'YoungAdult'
    elif age < 60:
        return 'Adult'
    return 'Senior'


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categoriza Age en rangos etarios (Child, Teen, YoungAdult, Adult, Senior).

    NB02: Age es significativa (Mann-Whitney p < 0.001). Se categoriza para
    capturar la no-linealidad de su relacion con el target.

    Args:
        df: DataFrame con columna 'Age'

    Returns:
        DataFrame con columna 'AgeCategory' anadida
    """
    df_copy = df.copy()
    df_copy['AgeCategory'] = df_copy['Age'].apply(_categorize_age)
    return df_copy


def handle_missing_values_spaceship(
    df: pd.DataFrame,
    impute_age: bool = False,
) -> pd.DataFrame:
    """
    Aplica la estrategia de nulos definida en NB02.

    Estrategia:
    - Categoricas (HomePlanet, CryoSleep, Destination, VIP) → 'Unknown'
    - Variables de gasto → 0 (ausencia = sin gasto registrado)
    - Age (train): eliminar filas (179 registros, 2.06%) para evitar sesgo en AgeCategory
    - Age (test):  imputar con mediana para preservar todos los registros

    Args:
        df: DataFrame con valores faltantes
        impute_age: Si True, imputa Age con la mediana en lugar de eliminar filas.
                    Usar True para datos de test donde no se pueden perder registros.

    Returns:
        DataFrame sin nulos en las columnas tratadas
    """
    df_copy = df.copy()
    for col in _CATEGORICAL_FILL:
        df_copy[col] = df_copy[col].fillna('Unknown')
    for col in _SPENDING_COLS:
        df_copy[col] = df_copy[col].fillna(0)
    if impute_age:
        df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].median())
    else:
        df_copy = df_copy.dropna(subset=['Age'])
    return df_copy


_DECK_TO_HOMEPLANET = {
    "A": "Europa", "B": "Europa", "C": "Europa",
    "G": "Earth",
}


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas fisicas del dataset para corregir e imputar valores nulos.

    Debe llamarse despues de extract_cabin_features y extract_group_features
    (necesita Deck, Side y TravelGroup disponibles).

    Reglas aplicadas en orden:
    1. HomePlanet por grupo: propaga HomePlanet conocido a companeros sin dato.
       Justificacion: los grupos viajan desde el mismo planeta de origen.
    2. Deck → HomePlanet: A/B/C son exclusivos de Europa; G es exclusivo de Earth.
    3. Deck y Side por grupo: propaga Deck/Side conocido a companeros con 'Unknown'.
    4. CryoSleep=True → spending = 0 (regla fisica: congelados no consumen).
    5. Spending > 0 → CryoSleep = False (inversa: si hay gasto, no estaba en cryo).
    6. Age <= 12 → spending = 0 (menores de 12 no acceden a amenidades de pago).

    Args:
        df: DataFrame con columnas Deck, Side, TravelGroup, HomePlanet, CryoSleep,
            Age y _SPENDING_COLS.

    Returns:
        DataFrame con valores imputados por reglas de dominio.
    """
    df_copy = df.copy()

    # --- Regla 1: HomePlanet por grupo ---
    known_hp = (
        df_copy.dropna(subset=["HomePlanet"])
        .groupby("TravelGroup")["HomePlanet"]
        .first()
    )
    hp_null = df_copy["HomePlanet"].isna()
    df_copy.loc[hp_null, "HomePlanet"] = (
        df_copy.loc[hp_null, "TravelGroup"].map(known_hp)
    )

    # --- Regla 2: Deck → HomePlanet ---
    for deck, planet in _DECK_TO_HOMEPLANET.items():
        mask = (df_copy["Deck"] == deck) & df_copy["HomePlanet"].isna()
        df_copy.loc[mask, "HomePlanet"] = planet

    # --- Regla 3: Deck y Side por grupo ---
    for col in ["Deck", "Side"]:
        if col not in df_copy.columns:
            continue
        known_val = (
            df_copy[df_copy[col] != "Unknown"]
            .groupby("TravelGroup")[col]
            .first()
        )
        unknown_mask = df_copy[col] == "Unknown"
        filled = df_copy.loc[unknown_mask, "TravelGroup"].map(known_val)
        df_copy.loc[unknown_mask, col] = filled.fillna("Unknown")

    # --- Regla 4: CryoSleep=True → spending = 0 ---
    cryo_true = df_copy["CryoSleep"].isin([True, "True"])
    for col in _SPENDING_COLS:
        df_copy.loc[cryo_true & df_copy[col].isna(), col] = 0.0

    # --- Regla 5: Spending > 0 → CryoSleep = False ---
    spending_positive = (df_copy[_SPENDING_COLS].fillna(0) > 0).any(axis=1)
    cryo_null = df_copy["CryoSleep"].isna()
    df_copy.loc[spending_positive & cryo_null, "CryoSleep"] = False

    # --- Regla 6: Age <= 12 → spending = 0 ---
    age_child = df_copy["Age"].notna() & (df_copy["Age"] <= 12)
    for col in _SPENDING_COLS:
        df_copy.loc[age_child & df_copy[col].isna(), col] = 0.0

    return df_copy


def impute_spending_group_aware(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa columnas de gasto con conciencia de grupo y CryoSleep.

    Estrategia diferencial por estado de CryoSleep:
    - CryoSleep=True: gasto → 0 (regla del mundo: congelados no gastan)
    - CryoSleep=False/Unknown: NaN → mediana del TravelGroup → mediana global → 0

    Prerequisito: TravelGroup y CryoSleep ya disponibles en el DataFrame
    (llamar despues de extract_group_features y fill categoricals).

    Args:
        df: DataFrame con columnas TravelGroup, CryoSleep y _SPENDING_COLS.

    Returns:
        DataFrame con columnas de gasto imputadas.
    """
    df_copy = df.copy()
    cryo_mask = df_copy["CryoSleep"].isin([True, "True"])

    for col in _SPENDING_COLS:
        null_mask = df_copy[col].isna()
        if null_mask.sum() == 0:
            continue
        global_med = df_copy[col].median()
        group_med = (
            df_copy.groupby("TravelGroup")[col]
            .transform("median")
            .fillna(global_med)
            .fillna(0)
        )
        df_copy.loc[null_mask & cryo_mask, col] = 0.0
        df_copy.loc[null_mask & ~cryo_mask, col] = group_med[null_mask & ~cryo_mask]

    return df_copy


def create_exp007_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 6 features del Exp-007 para Spaceship Titanic.

    Debe llamarse despues de extract_cabin_features, extract_group_features,
    create_spending_features, handle_missing_values_spaceship y el recomputo
    de GroupSize — es decir, sobre el DataFrame ya limpio.

    Features creadas:
        Route: Combinacion HomePlanet + Destination. Captura la trayectoria
            completa del pasajero, que determina su vector de vuelo en el
            momento del incidente.
        GroupCryoSleepRate: Fraccion de miembros del grupo en CryoSleep.
            Captura si el pasajero tomo una decision colectiva o individual.
        CryoSleepViolation: 1 si CryoSleep=True pero TotalSpending > 0.
            Identifica casos que violan la regla del mundo (congelado = sin gasto).
        LuxurySpendingRatio: (Spa + VRDeck) / (TotalSpending + 1).
            Perfil de gasto: ocio individual vs servicios sociales/basicos.
        CabinNumber_DeckPercentile: Posicion relativa de CabinNumber dentro
            de su Deck. Normaliza la coordenada espacial entre decks.
        GroupSpendingMean: Media de TotalSpending del grupo. Aporta contexto
            economico a pasajeros en CryoSleep (cuyo gasto individual es 0).

    Args:
        df: DataFrame con features base ya construidas y nulos tratados.

    Returns:
        DataFrame con las 6 nuevas features anadidas.
    """
    df_copy = df.copy()

    # Feature 1: Route
    hp = df_copy['HomePlanet'].fillna('Unknown').astype(str)
    dest = df_copy['Destination'].fillna('Unknown').astype(str)
    df_copy['Route'] = hp + '_to_' + dest

    # Feature 2: GroupCryoSleepRate
    cryo_num = df_copy['CryoSleep'].map(
        {True: 1, 'True': 1, False: 0, 'False': 0, 'Unknown': 0}
    ).fillna(0)
    df_copy['_cryo_num'] = cryo_num
    df_copy['GroupCryoSleepRate'] = (
        df_copy.groupby('TravelGroup')['_cryo_num'].transform('mean')
    )
    df_copy = df_copy.drop(columns=['_cryo_num'])

    # Feature 3: CryoSleepViolation
    cryo_true = df_copy['CryoSleep'].isin([True, 'True'])
    total_spend = df_copy[_SPENDING_COLS].fillna(0).sum(axis=1)
    df_copy['CryoSleepViolation'] = (cryo_true & (total_spend > 0)).astype(int)

    # Feature 4: LuxurySpendingRatio
    luxury = df_copy['Spa'].fillna(0) + df_copy['VRDeck'].fillna(0)
    df_copy['LuxurySpendingRatio'] = luxury / (df_copy['TotalSpending'] + 1)

    # Feature 5: CabinNumber_DeckPercentile
    deck_min = df_copy.groupby('Deck')['CabinNumber'].transform('min')
    deck_max = df_copy.groupby('Deck')['CabinNumber'].transform('max')
    span = (deck_max - deck_min).replace(0, 1)
    df_copy['CabinNumber_DeckPercentile'] = (
        (df_copy['CabinNumber'] - deck_min) / span
    ).fillna(0.5)

    # Feature 6: GroupSpendingMean
    df_copy['GroupSpendingMean'] = (
        df_copy.groupby('TravelGroup')['TotalSpending'].transform('mean')
    )

    return df_copy


def create_fs005_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 7 features de contexto estructural del fs-005.

    Requiere que el DataFrame ya tenga aplicado _pipeline_fs001 (TotalSpending,
    CabinNumber, Deck, TravelGroup, CryoSleep, Age y Name disponibles).

    Features creadas:
        SpendingEntropy: Entropia de Shannon sobre la distribucion de gasto entre
            las 5 categorias de servicio. Captura la FORMA del gasto, no su magnitud.
            H=0 si no gasta o concentra todo en una categoria. H maxima si distribuye
            uniformemente. Los arboles no pueden reconstruir esto con splits individuales.
        GroupSpendingZScore: Desviacion del gasto individual respecto al promedio del grupo,
            normalizada por la desviacion estandar del grupo. Captura si el pasajero es
            un outlier economico dentro de su propio contexto social.
        CabinNeighborhoodDensity: Numero de pasajeros en el mismo Deck con numero de
            cabina a distancia <= 50. Proxie de la densidad de ocupacion de la zona
            fisica del barco donde estaba el pasajero durante la anomalia.
        FamilySizeFromName: Numero de pasajeros con el mismo apellido en el dataset.
            Captura la unidad familiar independientemente del grupo de viaje (TravelGroup).
            Usa la columna Name que actualmente no aporta ninguna feature.
        GroupCryoAlignment: Nivel de consenso del grupo en la decision de CryoSleep.
            max(GroupCryoSleepRate, 1 - GroupCryoSleepRate). Distingue grupos que
            tomaron la decision colectivamente (alignment=1) de los mixtos (alignment=0.5).
        SpendingCategoryProfile: Huella binaria de CUALES servicios usa el pasajero
            (no cuanto). String de 5 bits, ej. '10110' = RoomService+ShoppingMall+Spa.
            Candidato para target encoding. SpendingCategories solo cuenta cuantos.
        GroupAgeDispersion: Desviacion estandar de Age dentro del TravelGroup.
            Grupos homogeneos en edad (amigos/colegas) vs heterogeneos (familias con
            ninos) tienen perfiles de transporte distintos.

    Args:
        df: DataFrame con pipeline fs-001 ya aplicado.

    Returns:
        DataFrame con las 7 nuevas features anadidas.
    """
    df_copy = df.copy()

    # 1. SpendingEntropy
    total = df_copy['TotalSpending'].values
    entropy_vals = np.zeros(len(df_copy))
    for col in _SPENDING_COLS:
        p = df_copy[col].fillna(0).values / (total + 1e-10)
        entropy_vals += np.where(p > 0, -p * np.log(p), 0.0)
    df_copy['SpendingEntropy'] = np.where(total > 0, entropy_vals, 0.0)

    # 2. GroupSpendingZScore
    g_mean = df_copy.groupby('TravelGroup')['TotalSpending'].transform('mean')
    g_std = df_copy.groupby('TravelGroup')['TotalSpending'].transform('std').fillna(0)
    df_copy['GroupSpendingZScore'] = (df_copy['TotalSpending'] - g_mean) / (g_std + 1)

    # 3. CabinNeighborhoodDensity (vectorizado por Deck)
    density = np.zeros(len(df_copy))
    cabin_vals = df_copy['CabinNumber'].values
    deck_vals = df_copy['Deck'].values
    for deck in np.unique(deck_vals):
        mask = deck_vals == deck
        idx = np.where(mask)[0]
        cn = cabin_vals[idx]
        diff = np.abs(cn[:, np.newaxis] - cn[np.newaxis, :])
        counts = (diff <= 50).sum(axis=1) - 1
        density[idx] = counts
    df_copy['CabinNeighborhoodDensity'] = density

    # 4. FamilySizeFromName
    last_names = df_copy['Name'].apply(
        lambda x: x.split()[-1] if pd.notna(x) and str(x).strip() else None
    )
    family_counts = last_names.value_counts().to_dict()
    df_copy['FamilySizeFromName'] = last_names.map(family_counts).fillna(1).astype(int)

    # 5. GroupCryoAlignment
    cryo_num = df_copy['CryoSleep'].map(
        {True: 1, 'True': 1, False: 0, 'False': 0, 'Unknown': 0}
    ).fillna(0)
    df_copy['_cryo_num'] = cryo_num
    g_cryo_rate = df_copy.groupby('TravelGroup')['_cryo_num'].transform('mean')
    df_copy['GroupCryoAlignment'] = np.maximum(g_cryo_rate, 1 - g_cryo_rate)
    df_copy = df_copy.drop(columns=['_cryo_num'])

    # 6. SpendingCategoryProfile (string binario → target encoding en 02_features)
    profile = pd.Series([''] * len(df_copy), index=df_copy.index)
    for col in _SPENDING_COLS:
        profile = profile + (df_copy[col].fillna(0) > 0).astype(int).astype(str)
    df_copy['SpendingCategoryProfile'] = profile

    # 7. GroupAgeDispersion
    df_copy['GroupAgeDispersion'] = (
        df_copy.groupby('TravelGroup')['Age'].transform('std').fillna(0)
    )

    return df_copy


def create_fs003_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 3 features de fs-003 para Spaceship Titanic.

    Debe llamarse despues de _pipeline_fs001 (GroupSize, Age y
    TotalSpending/SpendingCategories ya disponibles).

    Features creadas:
        IsAlone: 1 si GroupSize == 1. Viajeros solos tienen perfil
            de transporte distinto al de grupos (analogia Titanic original).
        IsChild: 1 si Age < 13. Refuerzo explicito de la senal de ninos
            ya capturada en AgeCategory, pero como binario directo.
        SpendingIntensity: TotalSpending / (SpendingCategories + 1).
            Gasto promedio por categoria activa. Diferencia a quien gasta
            concentrado (spa/VR) de quien usa todos los servicios moderadamente.

    Args:
        df: DataFrame con features base ya construidas (post _pipeline_fs001).

    Returns:
        DataFrame con las 3 nuevas features anadidas.
    """
    df_copy = df.copy()
    df_copy['IsAlone'] = (df_copy['GroupSize'] == 1).astype(int)
    df_copy['IsChild'] = (df_copy['Age'] < 13).astype(int)
    df_copy['SpendingIntensity'] = (
        df_copy['TotalSpending'] / (df_copy['SpendingCategories'] + 1)
    )
    return df_copy


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering para Spaceship Titanic.

    Ejecuta en orden:
    1. Extraccion de Cabin → Deck, CabinNumber, Side
    2. Extraccion de PassengerId → TravelGroup, GroupSize
    3. Creacion de features de gasto → TotalSpending, TotalSpending_Log, etc.
    4. Categorizacion de Age → AgeCategory
    5. Tratamiento de nulos (estrategia definida en NB02)
    6. Recomputa GroupSize tras la eliminacion de filas por Age nulo
    7. Features avanzadas del Exp-007 (Route, GroupCryoSleepRate, etc.)

    Args:
        df: DataFrame crudo de train.csv

    Returns:
        DataFrame con todas las features construidas, listo para encoding y escalado
    """
    df_out = extract_cabin_features(df)
    df_out = extract_group_features(df_out)
    df_out = create_spending_features(df_out)
    df_out = create_age_features(df_out)
    df_out = handle_missing_values_spaceship(df_out)
    df_out['GroupSize'] = df_out.groupby('TravelGroup')['TravelGroup'].transform('count')
    df_out = create_exp007_features(df_out)
    return df_out


# ---------------------------------------------------------------------------
# Encoding helpers (punto único de verdad — usados por scripts/ y src/models/)
# ---------------------------------------------------------------------------

def encode_cryosleep(val) -> int:
    """Codifica CryoSleep a entero.

    Args:
        val: Valor original (True, False, "True", "False" o cualquier otro).

    Returns:
        1 para True, 0 para False, -1 para desconocido.
    """
    if val in (True, "True"):
        return 1
    if val in (False, "False"):
        return 0
    return -1


def encode_side(val) -> int:
    """Codifica Side a entero.

    Args:
        val: Valor original ("P", "S" o cualquier otro).

    Returns:
        0 para P (port), 1 para S (starboard), -1 para desconocido.
    """
    if val == "P":
        return 0
    if val == "S":
        return 1
    return -1
