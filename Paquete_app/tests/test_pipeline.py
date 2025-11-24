# tests/test_pipeline.py
import joblib
import json
from pathlib import Path
import pandas as pd
from bmw_model_project import pipeline, config

TRAINED_DIR = config.TRAINED_DIR

def test_artifacts_exist():
    """Verifica que todos los archivos de entrenamiento existan"""
    artifacts = [
        "best_model.joblib",
        "best_model_metadata.json",
        "scaler_cluster.joblib",
        "kmeans_model.joblib",
    ]
    missing = [f for f in artifacts if not (TRAINED_DIR / f).exists()]
    assert not missing, f"Faltan artefactos: {missing}"

def test_model_prediction():
    """Verifica que el modelo pueda hacer predicciones"""
    model_path = TRAINED_DIR / "best_model.joblib"
    best_model = joblib.load(model_path)

    # Cargar datos y generar features
    df = pd.read_csv(config.DATA_URL, encoding="utf-8")
    df = pipeline.create_features(df)
    df = pipeline.perform_clustering(df)

    numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
    categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]

    preds = best_model.predict(X)
    assert len(preds) == len(df), "La cantidad de predicciones no coincide con los datos"
