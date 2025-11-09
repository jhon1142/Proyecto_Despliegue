#!/usr/bin/env python3
"""
modelo_bmw.py

Pipeline reproducible para:
- carga y preprocesamiento de datos BMW (2010-2024)
- segmentación con KMeans
- entrenamiento y optimización de modelos (XGBoost, RandomForest, GradientBoosting)
- validación y registro en MLflow
- guardado de artefactos para producción

Requisitos:
- MLflow server accesible en mlflow.set_tracking_uri(...)
- Python 3.8+
"""

import os
import logging
from pathlib import Path
import argparse
import joblib
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# ---------------------------
# Configuración mínima
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Rutas de salida
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents) else Path(".")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Funciones auxiliares
# ---------------------------
def cargar_datos(url: str) -> pd.DataFrame:
    logger.info("Cargando datos desde: %s", url)
    df = pd.read_csv(url, encoding='utf-8')
    logger.info("Datos cargados. Shape: %s", df.shape)
    return df

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creando features adicionales")
    df = df.copy()
    # Edad del modelo
    df['age_model'] = 2024 - df['Year']
    # Indicador lujo (percentil 75)
    df['is_luxury'] = (df['Price_USD'] > df['Price_USD'].quantile(0.75)).astype(int)

    # Clasificación de segmento basada en el nombre del modelo (mejorable con regex)
    def clasificar_segmento(ref):
        if pd.isna(ref):
            return "Otro"
        r = str(ref).upper()
        if ("I3" in r) or ("ELECTRIC" in r) or ("HYBRID" in r):
            return "Electrico_Hibrido"
        elif ("I8" in r) or (r.startswith("M") and len(r) <= 3) or ("Z" in r):
            return "Deportivo"
        elif r.startswith("X"):
            return "SUV"
        elif "7" in r or "SERIE 7" in r:
            return "Ejecutivo"
        elif "5" in r or "SERIE 5" in r:
            return "Ejecutivo"
        elif "3" in r or "SERIE 3" in r:
            return "Sedan"
        else:
            return "Otro"

    df['Segmento'] = df['Model'].apply(clasificar_segmento)
    logger.info("Features creados: age_model, is_luxury, Segmento")
    return df

def entrenar_kmeans_and_save(df: pd.DataFrame, cluster_features, n_clusters=8):
    """
    Entrena KMeans sobre cluster_features, guarda scaler y modelo KMeans,
    y retorna df con columna 'Cluster' (como string).
    """
    logger.info("Entrenando KMeans con n_clusters=%s", n_clusters)
    Xc = df[cluster_features].copy()
    scaler_cluster = StandardScaler()
    Xc_scaled = scaler_cluster.fit_transform(Xc)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(Xc_scaled)
    df['Cluster'] = clusters.astype(str)  # tratar como categórica
    # Guardar artefactos
    joblib.dump(scaler_cluster, MODELS_DIR / "scaler_cluster.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.joblib")
    logger.info("KMeans entrenado y guardado en %s", MODELS_DIR)
    return df

def build_preprocessor(numeric_features, categorical_features):
    """
    Construye ColumnTransformer con StandardScaler para num y OneHotEncoder para cat.
    Retorna el transformer (listo para usar dentro de Pipeline).
    """
    logger.info("Construyendo preprocesador")
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
        # PCA se omite por decisión (entorpecía modelo); si se desea, añadir aquí
    ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='drop')
    return preprocessor

def train_and_evaluate(models_search_space, X_train, y_train, X_test, y_test, mlflow_experiment):
    """
    models_search_space: list of tuples (estimator_object, param_distributions, name)
    Ejecuta RandomizedSearchCV para cada uno, registra en MLflow y devuelve resultados list.
    """
    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for estimator, param_dist, name in models_search_space:
        logger.info("Procesando modelo: %s", name)
        with mlflow.start_run(experiment_id=mlflow_experiment.experiment_id, run_name=name):
            # Randomized search (rápido). Si se desea exhaustivo, cambiar a GridSearchCV.
            n_iter_search = min(25, max(10, int(np.prod([len(v) for v in param_dist.values()])**0.5)))
            logger.info("RandomizedSearchCV n_iter=%s", n_iter_search)
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            search.fit(X_train, y_train)

            best = search.best_estimator_
            y_pred = best.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-val scores (MSE)
            cv_mse = -cross_val_score(best, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1).mean()

            logger.info("Modelo %s - R2: %.4f - MSE: %.4f (CV_MSE: %.4f)", name, r2, mse, cv_mse)
            logger.info("Mejores params: %s", search.best_params_)

            # Registrar en MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("mse", float(mse))
            mlflow.log_metric("r2", float(r2))
            mlflow.log_metric("cv_mse", float(cv_mse))
            # Registrar el modelo (como artefacto y opcionalmente en registry)
            artifact_path = f"{name}-model"
            try:
                mlflow.sklearn.log_model(sk_model=best, artifact_path=artifact_path)
            except Exception as e:
                logger.warning("No se pudo loggear modelo como sklearn artifact: %s", e)

            results.append({
                'name': name,
                'best_estimator': best,
                'best_params': search.best_params_,
                'r2': r2,
                'mse': mse
            })
    return results

# ---------------------------
# Main
# ---------------------------
def main(args):
    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    experiment = mlflow.set_experiment(args.experiment_name)

    # Cargar datos
    df = cargar_datos(args.data_url)

    # Preprocesamiento básico y features
    df = crear_features(df)

    # Verificar columnas necesarias
    expected_cols = ['Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume', 'Model', 'Year']
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no encontrada en dataset: {c}")

    # Entrenar KMeans y guardar artefactos
    cluster_features = ['Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume']
    df = entrenar_kmeans_and_save(df, cluster_features, n_clusters=args.n_clusters)

    # Preparar variables para el modelo predictivo
    target_col = 'Sales_Volume'
    # Seleccionar variables base (ajustar según disponibilidad)
    numeric_features = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'age_model']
    categorical_features = ['Region', 'Color', 'Fuel_Type', 'Transmission', 'Segmento', 'Cluster', 'is_luxury']

    # Asegurarse de que categóricas no contengan nan
    df[categorical_features] = df[categorical_features].fillna('UNKNOWN').astype(str)

    # Construir preprocesador y pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Pipeline final: preprocessor -> model (model será usado por RandomizedSearchCV)
    # Para search, pasaremos estimadores sin pipeline puesto que queremos preprocesar dentro de Pipeline
    # Construir X, y
    X = df[numeric_features + categorical_features]
    y = df[target_col].values

    # División train/test (no stratify para regresión)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, shuffle=True
    )

    # Construir Pipeline para cada estimator: preprocessor + estimator
    # Definir estimadores base
    rf_base = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    xgb_base = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'))
    ])
    gb_base = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # Definir espacios de búsqueda (Randomized)
    rf_param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [6, 8, 12, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    xgb_param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0]
    }
    gb_param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.6, 0.8, 1.0]
    }

    models_search_space = [
        (rf_base, rf_param_dist, "RandomForest"),
        (xgb_base, xgb_param_dist, "XGBoost"),
        (gb_base, gb_param_dist, "GradientBoosting")
    ]

    # Entrenamiento y registro
    results = train_and_evaluate(models_search_space, X_train, y_train, X_test, y_test, experiment)

    # Seleccionar mejor modelo por R2
    results_sorted = sorted(results, key=lambda r: r['r2'], reverse=True)
    best = results_sorted[0]
    logger.info("Mejor modelo: %s (R2=%.4f)", best['name'], best['r2'])

    # Guardar mejor modelo localmente
    best_model_path = MODELS_DIR / f"best_model_{best['name']}.joblib"
    joblib.dump(best['best_estimator'], best_model_path)
    logger.info("Mejor modelo guardado en: %s", best_model_path)

    # Registrar el mejor modelo en MLflow Model Registry (intentar)
    model_name = args.register_name
    try:
        # MLflow register
        mlflow.sklearn.log_model(
            sk_model=best['best_estimator'],
            artifact_path="best-model",
            registered_model_name=model_name
        )
        logger.info("Modelo registrado en MLflow Registry con nombre: %s", model_name)
    except Exception as e:
        logger.warning("No se pudo registrar modelo en Registry (ver MLflow server). Error: %s", e)

    # Guardar metadata (parámetros y métricas) localmente
    meta = {
        'best_model': best['name'],
        'r2': float(best['r2']),
        'mse': float(best['mse']),
        'best_params': best['best_params']
    }
    with open(MODELS_DIR / "best_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Pipeline finalizado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y registra modelos de ventas BMW.")
    parser.add_argument("--data_url", type=str,
                        default="https://raw.githubusercontent.com/jhon1142/Proyecto_Despliegue/main/BMW%20sales%20data%20(2010-2024)%20(1).csv",
                        help="URL o path local del CSV de datos")
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000",
                        help="URI del servidor MLflow (http://host:port)")
    parser.add_argument("--experiment_name", type=str, default="modelo_bmw_ventas",
                        help="Nombre del experimento MLflow")
    parser.add_argument("--n_clusters", type=int, default=8, help="Número de clusters KMeans")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tamaño del test set")
    parser.add_argument("--register_name", type=str, default="modelo_bmw_produccion",
                        help="Nombre para registrar el modelo en MLflow Registry")
    args = parser.parse_args()
    main(args)
