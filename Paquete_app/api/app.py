import json
import joblib
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

import plotly.express as px

from bmw_model_project import config

# Cargar modelo y métricas
best_model_path = config.TRAINED_DIR / "best_model.joblib"
best_model = joblib.load(best_model_path)

with open(config.TRAINED_DIR / "best_model_metadata.json") as f:
    metrics = json.load(f)

# Cargar dataset
df = pd.read_csv(config.DATA_URL, encoding="utf-8")

# Feature engineering
from bmw_model_project.pipeline import create_features, perform_clustering
df = create_features(df)
df = perform_clustering(df)

# Definir sets de features
numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]

all_features = numeric_features + categorical_features

# Inicializar app
app = dash.Dash(__name__)
app.title = "Dashboard BMW Model"

app.layout = html.Div([
    html.H1("Dashboard Resultados Modelo BMW"),

    # --- Mejor modelo y métricas ---
    html.Div([
        html.H3("Mejor modelo:"),
        html.P(f"{metrics['best_model']}"),
        html.H4("Métricas:"),
        html.Ul([
            html.Li(f"R2: {metrics['r2']:.4f}"),
            html.Li(f"MSE: {metrics['mse']:.2f}")
        ])
    ], style={"margin-bottom": "30px"}),

    # --- Selector de características ---
    html.H3("Selecciona características para predecir"),
    dcc.Dropdown(
        id="feature-selector",
        options=[{"label": f, "value": f} for f in all_features],
        value=all_features,   # por defecto usa todas
        multi=True
    ),

    # --- Gráfica ---
    dcc.Graph(id="scatter-pred-vs-true"),

    # --- Tabla ---
    dash_table.DataTable(
        id="table-predictions",
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"}
    )
])

# ---------------------- CALLBACKS ----------------------

@app.callback(
    [Output("scatter-pred-vs-true", "figure"),
     Output("table-predictions", "data")],
    [Input("feature-selector", "value")]
)
def update_predictions(selected_features):

    # Evitar que el modelo falle si no hay features
    if not selected_features:
        return px.scatter(title="Selecciona al menos una característica"), df.to_dict("records")

    # Construir matriz X con las características seleccionadas
    X = df[selected_features]

    # Generar predicciones
    try:
        df["Predicted_Sales"] = best_model.predict(X)
    except Exception as e:
        # Manejar errores por incompatibilidad de features
        print(f"Error en predicción: {e}")
        return px.scatter(title="Error con las características seleccionadas"), df.to_dict("records")

    # Crear figura
    fig = px.scatter(df,
                     x="Sales_Volume",
                     y="Predicted_Sales",
                     labels={"Sales_Volume": "Ventas Reales",
                             "Predicted_Sales": "Predicciones"},
                     title="Predicciones vs Ventas Reales")

    return fig, df.to_dict("records")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

