"""
Orquestador del pipeline de Feature Engineering.

Centraliza la lógica de transformación, encoding y escalado para
garantizar consistencia entre scripts, notebooks y producción.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.config.settings import MODELS_DIR, get_target_encoder_path, get_scaler_path
from src.features.constants import TARGET
from src.features.engineering import encode_cryosleep, encode_side

def run_feature_pipeline(df_raw: pd.DataFrame, fs: Any, fs_name: str) -> Dict[str, Any]:
    """Ejecuta el flujo completo de ingeniería de características.
    
    Args:
        df_raw: DataFrame original.
        fs: Objeto FeatureSet de src/features/feature_sets.py.
        fs_name: Nombre del feature set.
        
    Returns:
        Diccionario con X_train, y, scaler, target_encoder y metadatos.
    """
    df = df_raw.copy()
    
    # 1. Pipeline base
    df = fs.pipeline(df)
    
    # 2. Encoding manual (CryoSleep, Side)
    df["CryoSleep_Encoded"] = df["CryoSleep"].apply(encode_cryosleep)
    df["Side_Encoded"] = df["Side"].apply(encode_side)
    
    # 3. Target Encoding
    target_encoder = {}
    if fs.target_encode_cols:
        for col in fs.target_encode_cols:
            mapping = df.groupby(col)[TARGET].mean().to_dict()
            target_encoder[col] = mapping
            df[f"{col}_TE"] = df[col].map(mapping)
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(target_encoder, get_target_encoder_path(fs_name))
        
    # 4. One-Hot Encoding
    if fs.categorical_cols:
        df = pd.get_dummies(df, columns=fs.categorical_cols, drop_first=False)
        
    # 5. Drop de columnas innecesarias
    cols_to_drop = fs.features_to_drop + ["CryoSleep", "Side"] + list(fs.target_encode_cols)
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)
    
    # 6. Separación X/y
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    
    # 7. Escalado
    scaler = StandardScaler()
    X_scaled = X.copy()
    # Asegurar que booleanos sean enteros para el escalado
    bool_cols = X_scaled.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        X_scaled[bool_cols] = X_scaled[bool_cols].astype(int)
        
    X_scaled[fs.numeric_features] = scaler.fit_transform(X_scaled[fs.numeric_features])
    joblib.dump(scaler, get_scaler_path(fs_name))
    
    return {
        "X_raw": X,
        "X_scaled": X_scaled,
        "y": y,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "metadata": {
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "cols": list(X.columns)
        }
    }
