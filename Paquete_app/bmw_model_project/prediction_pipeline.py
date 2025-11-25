"""
prediction_pipeline.py
Contiene la lógica para procesar datos para predicciones.
"""

import joblib
import pandas as pd
import json
from . import config

def create_features_for_prediction(df):
    """Genera nuevas variables relevantes para el modelo usando valores de entrenamiento."""
    df["age_model"] = 2024 - df["Year"]
    
    # Cargar el cuantil de entrenamiento
    with open(config.TRAINED_DIR / "is_luxury_quantile.json", "r") as f:
        is_luxury_quantile = json.load(f)["is_luxury_quantile"]
        
    df["is_luxury"] = (df["Price_USD"] > is_luxury_quantile).astype(int)

    def clasificar_segmento(ref):
        if pd.isna(ref):
            return "Otro"
        ref = str(ref).upper()
        if ("I3" in ref) or ("ELECTRIC" in ref) or ("HYBRID" in ref):
            return "Electrico_Hibrido"
        elif ("I8" in ref) or (ref.startswith("M")) or ("Z" in ref):
            return "Deportivo"
        elif ref.startswith("X"):
            return "SUV"
        elif "5" in ref or "7" in ref:
            return "Ejecutivo"
        elif "3" in ref:
            return "Sedan"
        else:
            return "Otro"

    df["Segmento"] = df["Model"].apply(clasificar_segmento)
    return df

def cluster_for_prediction(df):
    """Aplica clustering a nuevos datos usando el modelo y scaler entrenados."""
    cluster_features = ["Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]
    X = df[cluster_features]

    # Cargar scaler y kmeans entrenados
    scaler = joblib.load(config.TRAINED_DIR / "scaler_cluster.joblib")
    kmeans = joblib.load(config.TRAINED_DIR / "kmeans_model.joblib")

    X_scaled = scaler.transform(X)
    df["Cluster"] = kmeans.predict(X_scaled).astype(str)

    return df
