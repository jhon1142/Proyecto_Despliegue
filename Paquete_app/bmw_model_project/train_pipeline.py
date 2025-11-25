"""
train_pipeline.py
Ejecuta todo el flujo de entrenamiento de extremo a extremo.
"""

from . import pipeline

if __name__ == "__main__":
    print("=== Iniciando pipeline de entrenamiento BMW ===")
    df = pipeline.load_data()
    df = pipeline.create_features(df)
    df = pipeline.perform_clustering(df)
    pipeline.train_and_register_model(df)
    print("=== Proceso completado exitosamente ===")
