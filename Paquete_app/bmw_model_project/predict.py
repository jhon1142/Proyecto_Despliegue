"""
predict.py
Usa el modelo entrenado para hacer predicciones sobre nuevos datos.
"""

import joblib
import pandas as pd
from . import config, prediction_pipeline


def predict_new(sample_dict):
    """
    Realiza predicciones con el modelo entrenado.
    sample_dict: dict con los campos esperados (Year, Engine_Size_L, Mileage_KM, Price_USD, ...)
    """
    model = joblib.load(config.TRAINED_DIR / "best_model.joblib")

    df_sample = pd.DataFrame([sample_dict])
    df_sample = prediction_pipeline.create_features_for_prediction(df_sample)
    df_sample = prediction_pipeline.cluster_for_prediction(df_sample)

    y_pred = model.predict(df_sample)
    return y_pred


if __name__ == "__main__":
    ejemplo = {
        "Year": 2023,
        "Engine_Size_L": 2.0,
        "Mileage_KM": 15000,
        "Price_USD": 65000,
        "Region": "Europe",
        "Color": "Black",
        "Fuel_Type": "Petrol",
        "Transmission": "Automatic",
        "Model": "M3"
    }
    pred = predict_new(ejemplo)
    print(f"Predicción estimada de volumen de ventas: {pred[0]:.2f}")
