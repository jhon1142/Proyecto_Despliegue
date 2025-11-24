from fastapi import FastAPI
from typing import List, Optional, Any
import joblib
import pandas as pd
from arbol import asection, aprint
from pydantic import BaseModel

# Importar tu paquete BMW ya instalado
from bmw_model_project import pipeline, config

app = FastAPI(title="BMW Sales Prediction API")

# -------------------------
# Versiones
# -------------------------
API_VERSION = "1.0.0"
MODEL_VERSION = "0.1.0"

# -------------------------
# Cargar modelo entrenado
# -------------------------
MODEL_PATH = config.TRAINED_DIR / "best_model.joblib"
model = joblib.load(MODEL_PATH)
aprint(f"✅ Modelo cargado desde: {MODEL_PATH}")

# -------------------------
# Pydantic Models
# -------------------------
class Health(BaseModel):
    name: str
    api_version: str
    model_version: str

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

class DataInputSchema(BaseModel):
    Year: int
    Engine_Size_L: float
    Mileage_KM: float
    Price_USD: float
    Region: str
    Color: str
    Fuel_Type: str
    Transmission: str
    Model: str

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Year": 2020,
                        "Engine_Size_L": 2.0,
                        "Mileage_KM": 15000,
                        "Price_USD": 45000,
                        "Region": "Europe",
                        "Color": "Black",
                        "Fuel_Type": "Petrol",
                        "Transmission": "Automatic",
                        "Model": "X5"
                    }
                ]
            }
        }

# -------------------------
# Endpoints
# -------------------------
@app.get("/health", response_model=Health)
def health_check():
    """Endpoint de health check de la API."""
    return Health(
        name="BMW Sales Prediction API",
        api_version=API_VERSION,
        model_version=MODEL_VERSION
    )

@app.post("/predict", response_model=PredictionResults)
def predict(data: MultipleDataInputs):
    """
    Endpoint para predicciones múltiples usando el modelo empaquetado.
    """
    try:
        with asection("Inicio predicción"):
            # Convertir lista de inputs a DataFrame
            df = pd.DataFrame([d.dict() for d in data.inputs])
            aprint(f"Datos recibidos:\n{df.head()}")

            # Generar features y clusters usando funciones de tu paquete
            df = pipeline.create_features(df)
            df = pipeline.perform_clustering(df)

            # Columnas que usa el modelo
            numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
            categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]
            X = df[numeric_features + categorical_features]

            preds = model.predict(X).tolist()
            aprint(f"Predicciones generadas: {preds}")

        return PredictionResults(errors=None, version=MODEL_VERSION, predictions=preds)

    except Exception as e:
        aprint(f"❌ Error en predicción: {e}")
        return PredictionResults(errors=str(e), version=MODEL_VERSION, predictions=None)
