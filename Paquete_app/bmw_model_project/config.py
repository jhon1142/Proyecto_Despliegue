"""
config.py
Configuraciones globales del proyecto BMW Model Project
"""

from pathlib import Path

# ==============================
# Rutas principales
# ==============================

PACKAGE_ROOT = Path(__file__).resolve().parent
TRAINED_DIR = PACKAGE_ROOT / "trained"
DATA_URL = PACKAGE_ROOT / "data" / "data.csv"

# ==============================
# Fuentes de datos
# ==============================
DATA_URL = PACKAGE_ROOT /"data" / "data.csv"

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
