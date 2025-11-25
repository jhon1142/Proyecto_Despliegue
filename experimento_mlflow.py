"""
experimento_mlflow.py
Ejecuta un experimento de MLflow con un registro detallado de cada combinacion de hiperparametros.
"""

import itertools
import random
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

# --- Copied and adapted from Paquete_app/bmw_model_project/ ---
# This avoids having to install the package to run the experiment script.
PACKAGE_ROOT = Path(__file__).resolve().parent / "Paquete_app" / "bmw_model_project"
TRAINED_DIR = PACKAGE_ROOT / "trained"
DATA_URL = PACKAGE_ROOT / "data" / "data.csv"
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "modelo_bmw_ventas_detallado"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CLUSTERS = 8
# --- End of copied config ---


def load_data():
    """Carga los datos desde la fuente definida."""
    df = pd.read_csv(str(DATA_URL), encoding="utf-8")
    return df


def create_features(df):
    """Genera nuevas variables relevantes para el modelo."""
    df["age_model"] = 2024 - df["Year"]
    
    is_luxury_quantile = df["Price_USD"].quantile(0.75)
    with open(TRAINED_DIR / "is_luxury_quantile.json", "w") as f:
        json.dump({"is_luxury_quantile": is_luxury_quantile}, f)
        
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


def perform_clustering(df):
    """Entrena KMeans y agrega columna 'Cluster'."""
    cluster_features = ["Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]
    X = df[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=N_CLUSTERS,
            init="k-means++",
            max_iter=300,
            random_state=RANDOM_STATE,
            n_init=10
        ))
    ])

    df["Cluster"] = kmeans.fit_predict(X[cluster_features]).astype(str)

    joblib.dump(kmeans, TRAINED_DIR / "kmeans_model.joblib")
    return df


def build_preprocessor(numeric_features, categorical_features):
    """Crea un preprocesador para columnas numéricas y categóricas."""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )
    return preprocessor


def run_experiment():
    """
    Ejecuta el experimento de ML, registrando cada prueba en MLflow.
    """
    print("=== Iniciando experimento detallado de MLflow ===")
    
    # 1. Cargar y preparar datos
    df = load_data()
    df = create_features(df)
    df = perform_clustering(df)

    # 2. Definir features y target
    target_col = "Sales_Volume"
    numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
    categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]

    df[categorical_features] = df[categorical_features].fillna("UNKNOWN").astype(str)
    X = df[numeric_features + categorical_features]
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 3. Definir modelos y grillas de hiperparámetros
    models = {
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [6, 8, 12, 20, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [4, 6, 8, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 8, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        },
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    best_overall = None
    
    # 4. Iniciar bucle de experimentos
    for model_name, base_model in models.items():
        print(f"--- Entrenando modelo: {model_name} ---")
        
        # Generar todas las combinaciones de hiperparámetros
        keys, values = zip(*param_grids[model_name].items())
        all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Seleccionar 15 combinaciones al azar
        num_experiments = min(15, len(all_param_combinations))
        param_sample = random.sample(all_param_combinations, num_experiments)

        for i, params in enumerate(param_sample):
            with mlflow.start_run(run_name=f"{model_name}_trial_{i+1}"):
                print(f"  Trial {i+1}/{num_experiments} con params: {params}")

                # Crear y entrenar el pipeline
                regressor = base_model.set_params(**params)
                model_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("regressor", regressor)
                ])

                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)

                # Calcular y registrar métricas
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_params(params)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(model_pipeline, artifact_path="model")

                # Mantener el mejor modelo
                if not best_overall or r2 > best_overall["r2"]:
                    best_overall = {
                        "name": f"{model_name}_trial_{i+1}",
                        "model": model_pipeline,
                        "r2": r2,
                        "mse": mse,
                        "params": params,
                    }

    # 5. Guardar el mejor modelo de todos los experimentos
    if best_overall:
        joblib.dump(best_overall["model"], TRAINED_DIR / "best_model_experimento.joblib")
        
        meta = {
            "best_model_run": best_overall["name"],
            "r2": best_overall["r2"],
            "mse": best_overall["mse"],
            "params": best_overall["params"],
        }
        with open(TRAINED_DIR / "best_model_experimento_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
            
        print(f"\n✅ Experimento completado. Mejor modelo guardado: {best_overall['name']} (R2={best_overall['r2']:.4f})")
    else:
        print("\n❌ No se pudo entrenar ningún modelo.")


if __name__ == "__main__":
    run_experiment()