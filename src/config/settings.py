"""
Configuracion de rutas del proyecto Spaceship Titanic.

Carga las variables de entorno desde el archivo .env ubicado en la raiz
del proyecto y expone objetos Path tipados listos para importar.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Directorio base del proyecto
BASE_DIR: Path = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# Directorios de datos
DATA_RAW_DIR: Path = Path(
    os.getenv(
        "DATA_RAW_DIR", 
        str(BASE_DIR / "data" / "raw"))
)

# Directorio de datos procesados
DATA_PROCESSED_DIR: Path = Path(
    os.getenv(
        "DATA_PROCESSED_DIR", 
        str(BASE_DIR / "data" / "processed"))
)

# Directorio de features
DATA_FEATURES_DIR: Path = BASE_DIR / "data" / "features"

# Directorio de submissions
SUBMISSIONS_DIR: Path = BASE_DIR / "data" / "submissions"

# Directorio de modelos
MODELS_DIR: Path = Path(
    os.getenv(
        "MODELS_DIR", 
        str(BASE_DIR / "models"))
)

# Directorio de reports
REPORTS_DIR: Path = Path(
    os.getenv(
        "REPORTS_DIR", 
        str(BASE_DIR / "reports"))
)

# Directorio de docs
DOCS_DIR: Path = BASE_DIR / "docs"

# Rutas derivadas — archivos concretos
TRAIN_RAW: Path = DATA_RAW_DIR / "train.csv"
TEST_RAW: Path = DATA_RAW_DIR / "test.csv"

PRODUCTION_DIR: Path = MODELS_DIR / "production"
EXPERIMENTS_DIR: Path = MODELS_DIR / "experiments"
SCALER_PATH: Path = PRODUCTION_DIR / "scaler.pkl"
MODEL_PATH: Path = PRODUCTION_DIR / "best_model.pkl"
MODEL_METADATA: Path = PRODUCTION_DIR / "model_metadata.json"

# MLflow Tracking
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "spacechip_titanic")

# Prefijo del run padre al ejecutar el pipeline completo (run.py / run_pipeline.py)
MLFLOW_PIPELINE_RUN_PREFIX: str = os.getenv(
    "MLFLOW_PIPELINE_RUN_PREFIX", 
    "Spaceship_Titanic_Full_Pipeline"
)


# ---------------------------------------------------------------------------
# Rutas dinamicas por feature set
# ---------------------------------------------------------------------------

def get_train_scaled(fs_name: str) -> Path:
    """Ruta del dataset escalado para un feature set especifico.

    Args:
        fs_name: Nombre del feature set (e.g. 'fs-001_baseline').

    Returns:
        Path a data/features/train_scaled_{fs_name}.csv
    """
    return DATA_FEATURES_DIR / f"train_scaled_{fs_name}.csv"


def get_train_features(fs_name: str) -> Path:
    """Ruta del dataset de features sin escalar para un feature set especifico.

    Args:
        fs_name: Nombre del feature set (e.g. 'fs-001_baseline').

    Returns:
        Path a data/features/train_features_{fs_name}.csv
    """
    return DATA_FEATURES_DIR / f"train_features_{fs_name}.csv"


def get_submission_path(exp_id: str) -> Path:
    """Ruta del archivo submission para un experimento especifico.

    Args:
        exp_id: ID del experimento (e.g. '014').

    Returns:
        Path a data/submissions/exp-{exp_id}_submission.csv
    """
    return SUBMISSIONS_DIR / f"exp-{exp_id}_submission.csv"


def get_target_encoder_path(fs_name: str) -> Path:
    """Ruta del target encoder serializado para un feature set especifico.

    Solo existe si el feature set tiene target_encode_cols no vacias.

    Args:
        fs_name: Nombre del feature set (e.g. 'fs-004_target_encoding').

    Returns:
        Path a models/experiments/target_encoder_{fs_name}.pkl
    """
    return EXPERIMENTS_DIR / f"target_encoder_{fs_name}.pkl"


def get_scaler_path(fs_name: str) -> Path:
    """Ruta del scaler serializado para un feature set especifico.

    Args:
        fs_name: Nombre del feature set (e.g. 'fs-001_baseline').

    Returns:
        Path a models/experiments/scaler_{fs_name}.pkl
    """
    return EXPERIMENTS_DIR / f"scaler_{fs_name}.pkl"
