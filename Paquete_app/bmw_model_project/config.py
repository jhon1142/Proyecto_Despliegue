"""
config.py
Configuraciones globales del proyecto BMW Model Project
"""

from pathlib import Path

# ==============================
# Rutas principales
# ==============================
ROOT_DIR = Path(__file__).resolve().parent
TRAINED_DIR = ROOT_DIR / "trained"
TRAINED_DIR.mkdir(exist_ok=True)

# ==============================
# Fuentes de datos
# ==============================
DATA_URL = ROOT_DIR /"data" / "data.csv"

# ==============================
# Configuración de Clustering
# ==============================
N_CLUSTERS = 8

# ==============================
# Configuración MLflow
# ==============================
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "modelo_bmw_ventas"
MODEL_REGISTER_NAME = "modelo_bmw_produccion"

# ==============================
# Configuración de modelo
# ==============================
TEST_SIZE = 0.2
RANDOM_STATE = 42
