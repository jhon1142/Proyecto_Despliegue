import pandas as pd
import joblib
import numpy as np

# --- Carga de Datos y Modelo ---

print(">>> Iniciando la generación de predicciones <<<")

# 1. Cargar el pipeline completo (preprocesador + modelo)
model_filename = 'entrega_2_modelo_clasificacion_bmw.joblib'
try:
    model_pipeline = joblib.load(model_filename)
    print(f"[Paso 1/3] Modelo '{model_filename}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del modelo '{model_filename}'.")
    print("Por favor, ejecuta primero el script 'entrega_2.py' para entrenar y guardar el modelo.")
    exit()

# 2. Cargar los datos originales para la predicción
try:
    # El nombre del archivo en el script de entrenamiento es diferente. Usamos ese.
    data = pd.read_csv('BMW sales data.csv')
    print("[Paso 2/3] Datos de entrada ('BMW sales data.csv') cargados.")
except FileNotFoundError:
    print("Error: El archivo 'BMW sales data.csv' no fue encontrado.")
    exit()

# --- Preparación de Datos para Predicción ---
# Se deben aplicar las mismas transformaciones iniciales que en el script de entrenamiento
# ANTES de pasarlo al pipeline.

# Crear la característica 'age_model'
if 'Year' in data.columns:
    data['age_model'] = 2024 - data['Year']
    # Eliminar 'Year' y 'Sales_Volume' como en el script de entrenamiento
    data_to_predict = data.drop(columns=['Year', 'Sales_Volume'], errors='ignore')
else:
    print("Advertencia: La columna 'Year' no se encontró para crear 'age_model'.")
    data_to_predict = data.copy()

# El pipeline espera que la columna 'Sales_Classification' no esté, así que la quitamos si existe.
if 'Sales_Classification' in data_to_predict.columns:
    data_to_predict = data_to_predict.drop('Sales_Classification', axis=1)

# --- Realizar y Guardar Predicciones ---

print("[Paso 3/3] Realizando predicciones y guardando el archivo...")

# 3. Usar el pipeline para predecir sobre los nuevos datos
# El pipeline se encarga de la imputación, el escalado y el one-hot encoding.
predictions_numeric = model_pipeline.predict(data_to_predict)

# Mapear las predicciones numéricas (0, 1) de vuelta a etiquetas ('Low', 'High')
predictions_labels = pd.Series(predictions_numeric).map({0: 'Low', 1: 'High'}).values

# 4. Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame(predictions_labels, columns=['Sales_Classification_Predicted'])

# 5. Guardar las predicciones en el archivo predic.csv
output_filename = 'predic.csv'
predictions_df.to_csv(output_filename, index=False)

print(f"\n>>> ¡Proceso completado! Predicciones guardadas en '{output_filename}' <<<")