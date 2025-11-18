
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
import warnings

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

def clasificar_segmento(ref):
    """Clasifica el segmento del vehículo basado en el modelo."""
    ref = ref.upper()
    if "I3" in ref or "ELECTRIC" in ref or "HYBRID" in ref:
        return "Eléctrico / Híbrido"
    elif "I8" in ref or "M" in ref or "Z" in ref:
        return "Deportivo"
    elif "X" in ref:
        return "Camioneta / SUV"
    elif "7" in ref or "5" in ref:
        return "Ejecutivo"
    elif "3" in ref:
        return "Sedán"
    else:
        return "Otro"

def run_experiment(clf_params, run_name):
    """
    Ejecuta un experimento completo: carga de datos, preprocesamiento,
    entrenamiento de un pipeline y registro en MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        print(f"--- Iniciando ejecución: {run_name} ---")

        # --- 1. Carga y Preparación de Datos ---
        df = pd.read_csv("BMW sales data.csv")
        df['is_luxury'] = (df['Price_USD'] > df['Price_USD'].quantile(0.75)).astype(int)
        df['Segmento'] = df['Model'].apply(clasificar_segmento)
        
        # Mapear target a 0 y 1
        df['Sales_Classification'] = df['Sales_Classification'].map({'Low': 0, 'High': 1})

        # --- 2. División de Datos ---
        X = df.drop('Sales_Classification', axis=1)
        y = df['Sales_Classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- 3. Definición del Pipeline de Preprocesamiento y Modelo ---
        numeric_features = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume', 'is_luxury']
        categorical_features = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission', 'Segmento']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Crear el pipeline completo
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(**clf_params))
        ])

        # --- 4. Entrenamiento del Pipeline ---
        print(f"Entrenando pipeline con parámetros: {clf_params}")
        full_pipeline.fit(X_train, y_train)

        # --- 5. Evaluación ---
        y_pred = full_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")

        # --- 6. Registro en MLflow ---
        print("Registrando en MLflow...")
        mlflow.log_params(clf_params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        
        # Mapear y_test a etiquetas para el reporte
        y_test_labels = y_test.map({0: 'Low', 1: 'High'})
        y_pred_labels = pd.Series(y_pred).map({0: 'Low', 1: 'High'})
        
        report_labels = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        for class_label, metrics in report_labels.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_label}_{metric_name}", value)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        plt.xlabel('Predicha')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusión - {run_name}')
        
        confusion_matrix_path = f"cm_{run_name}.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path, "plots")
        
        # Registrar el PIPELINE completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="bmw_sales_pipeline",
            registered_model_name="bmw_sales_classifier_pipeline"
        )
        
        print(f"--- Ejecución {run_name} completada. ---")
        print() # Salto de línea

if __name__ == "__main__":
    mlflow.set_experiment("Clasificacion_Ventas_BMW_Pipeline")

    # Usaremos solo el mejor conjunto de parámetros encontrado anteriormente
    best_params = {
        "max_iter": 1500,
        "solver": 'liblinear',
        "C": 10,
        "penalty": 'l1',
        "class_weight": 'balanced',
        "random_state": 42
    }
    
    run_name = f"Pipeline_C_{best_params['C']}_solver_{best_params['solver']}_penalty_{best_params['penalty']}"
    
    print("Ejecutando el mejor experimento con un pipeline completo.")
    run_experiment(best_params, run_name)
    
    print("Experimento con pipeline completado.")
