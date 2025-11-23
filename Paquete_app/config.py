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
DATA_URL = "https://raw.githubusercontent.com/jhon1142/Proyecto_Despliegue/main/BMW%20sales%20data%20(2010-2024)%20(1).csv"

# ==============================
# Configuración de Clustering
# ==============================
N_CLUSTERS = 8

# ==============================
# Configuración MLflow
# ==============================
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "modelo_bmw_ventas"
MODEL_REGISTER_NAME = "modelo_bmw_produccion"

# ==============================
# Configuración de modelo
# ==============================
TEST_SIZE = 0.2
RANDOM_STATE = 42
