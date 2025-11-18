import pandas as pd
import mlflow.pyfunc
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

def generar_predicciones():
    """
    Carga el modelo en 'Staging' desde el registro de MLflow,
    prepara los datos y genera un archivo de predicciones.
    """
    print(">>> Iniciando la generación de predicciones <<<")

    # --- 1. Cargar Modelo desde el Registro de MLflow ---
    model_name = "bmw_sales_classifier_pipeline"
    stage = "Staging" # Asumimos que el modelo bueno fue promovido a Staging
    model_uri = f"models:/{model_name}/{stage}"
    
    try:
        print(f"[Paso 1/3] Cargando modelo '{model_name}' (Versión en '{stage}')...")
        # Para que esto funcione, la UI de MLflow debe estar corriendo
        model_pipeline = mlflow.pyfunc.load_model(model_uri)
        print("Modelo cargado exitosamente desde el registro de MLflow.")
    except Exception as e:
        print(f"Error al cargar el modelo desde MLflow: {e}")
        print("Asegúrate de que la UI de MLflow esté corriendo y que un modelo haya sido transicionado a 'Staging'.")
        exit()

    # --- 2. Cargar y Preparar Datos de Inferencia ---
    try:
        df = pd.read_csv("BMW sales data.csv")
        print("[Paso 2/3] Datos de entrada ('BMW sales data.csv') cargados.")
    except FileNotFoundError:
        print("Error: El archivo 'BMW sales data.csv' no fue encontrado.")
        exit()

    # El pipeline fue entrenado con estas características, así que las creamos
    df['is_luxury'] = (df['Price_USD'] > df['Price_USD'].quantile(0.75)).astype(int)
    df['Segmento'] = df['Model'].apply(clasificar_segmento)
    
    # --- 3. Realizar y Guardar Predicciones ---
    print("[Paso 3/3] Realizando predicciones y guardando el archivo...")
    
    predictions_numeric = model_pipeline.predict(df)

    # Mapear las predicciones numéricas (0, 1) de vuelta a etiquetas ('Low', 'High')
    predictions_labels = pd.Series(predictions_numeric).map({0: 'Low', 1: 'High'}).values

    # Crear un DataFrame con las predicciones
    output_df = pd.DataFrame({
        'Sales_Classification_Predicted': predictions_labels
    })

    # Guardar las predicciones en el archivo predic.csv
    output_filename = 'predic.csv'
    output_df.to_csv(output_filename, index=False)

    print(f"\n>>> ¡Proceso completado! Predicciones guardadas en '{output_filename}' <<<")
    print(f"Se generaron {len(output_df)} predicciones.")

if __name__ == "__main__":
    generar_predicciones()
