"""
pipeline.py
Contiene la lógica completa de entrenamiento para el proyecto BMW.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

from . import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

def load_data():
    """Carga los datos desde la fuente definida en config.py"""
    df = pd.read_csv(str(config.DATA_URL), encoding="utf-8")
    return df


def create_features(df):
    """Genera nuevas variables relevantes para el modelo"""
    df["age_model"] = 2024 - df["Year"]
    
    # Guardar el cuantil para usar en predicción
    is_luxury_quantile = df["Price_USD"].quantile(0.75)
    with open(config.TRAINED_DIR / "is_luxury_quantile.json", "w") as f:
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
    """Entrena KMeans y agrega columna 'Cluster'"""
    cluster_features = ["Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume"]
    X = df[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=config.N_CLUSTERS,
        init="k-means++",
        max_iter=300,
        random_state=config.RANDOM_STATE,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    # Guardar artefactos
    joblib.dump(scaler, config.TRAINED_DIR / "scaler_cluster.joblib")
    joblib.dump(kmeans, config.TRAINED_DIR / "kmeans_model.joblib")

    return df


def build_preprocessor(numeric_features, categorical_features):
    """Crea un preprocesador para columnas numéricas y categóricas"""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )
    return preprocessor


def train_and_register_model(df):
    """Entrena modelos (RandomForest, XGBoost, GradientBoosting) y registra en MLflow"""

    df = df.copy()
    target_col = "Sales_Volume"

    numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
    categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]

    df[categorical_features] = df[categorical_features].fillna("UNKNOWN").astype(str)
    X = df[numeric_features + categorical_features]
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    rf = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))])
    xgb = Pipeline([("preprocessor", preprocessor), ("regressor", XGBRegressor(random_state=42, n_jobs=-1))])
    gb = Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(random_state=42))])

    param_grids = {
        "RandomForest": {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__max_depth": [6, 8, 12, 20, None],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [4, 6, 8, 10],
            "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
            "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        "GradientBoosting": {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [3, 5, 8, 10],
            "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        },
    }

    models = {
        "RandomForest": rf,
        "XGBoost": xgb,
        "GradientBoosting": gb,
    }

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    best_overall = None

    for name, model in models.items():
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=name):
            search = RandomizedSearchCV(model, param_grids[name], n_iter=15, cv=3, scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(best_model, artifact_path=f"{name}-model")

            if not best_overall or r2 > best_overall["r2"]:
                best_overall = {"name": name, "model": best_model, "r2": r2, "mse": mse}

    # Guardar modelo final
    joblib.dump(best_overall["model"], config.TRAINED_DIR / "best_model.joblib")

    meta = {
        "best_model": best_overall["name"],
        "r2": best_overall["r2"],
        "mse": best_overall["mse"],
    }
    with open(config.TRAINED_DIR / "best_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Entrenamiento completado. Mejor modelo: {best_overall['name']} (R2={best_overall['r2']:.4f})")

