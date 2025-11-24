import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
import joblib

from bmw_model_project import pipeline, config
from app import __version__, schemas  # __version__ de la API, schemas de tu proyecto

api_router = APIRouter()

# -------------------------
# Cargar modelo entrenado
# -------------------------
MODEL_PATH = config.TRAINED_DIR / "best_model.joblib"
model = joblib.load(MODEL_PATH)
logger.info(f"✅ Modelo cargado desde: {MODEL_PATH}")

# -------------------------
# Ruta de health check
# -------------------------
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get: Verifica que la API esté corriendo
    """
    health = schemas.Health(
        name="BMW Sales Prediction API",  # o settings.PROJECT_NAME si tienes
        api_version=__version__,
        model_version="0.1.0"
    )
    return health.dict()

# -------------------------
# Ruta de predicciones
# -------------------------
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Predicción usando el modelo BMW
    """
    try:
        # Convertir inputs a DataFrame
        input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
        logger.info(f"Datos recibidos para predicción: {input_df.head()}")

        # Aplicar funciones del pipeline
        input_df = pipeline.create_features(input_df)
        input_df = pipeline.perform_clustering(input_df)

        # Columnas que usa el modelo
        numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
        categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]
        X = input_df[numeric_features + categorical_features]

        # Generar predicciones
        preds = model.predict(X).tolist()
        logger.info(f"Predicciones generadas: {preds}")

        return schemas.PredictionResults(errors=None, version="0.1.0", predictions=preds)

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))
