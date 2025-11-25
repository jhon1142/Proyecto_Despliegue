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

# Si tienes funciones para generar features y clustering:
from bmw_model_project.pipeline import create_features, perform_clustering
df = create_features(df)
df = perform_clustering(df)

# Predicciones con el mejor modelo
numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]
X = df[numeric_features + categorical_features]
y_true = df["Sales_Volume"].values
y_pred = best_model.predict(X)
df["Predicted_Sales"] = y_pred

# Inicializar app
app = dash.Dash(__name__)
app.title = "Dashboard BMW Model"

app.layout = html.Div([
    html.H1("Dashboard Resultados Modelo BMW"),
    
    html.Div([
        html.H3("Mejor modelo:"),
        html.P(f"{metrics['best_model']}"),
        html.H4("Métricas:"),
        html.Ul([
            html.Li(f"R2: {metrics['r2']:.4f}"),
            html.Li(f"MSE: {metrics['mse']:.2f}")
        ])
    ], style={"margin-bottom": "30px"}),

    html.H3("Predicciones vs Reales"),
    dcc.Graph(
        id="scatter-pred-vs-true",
        figure=px.scatter(df, x="Sales_Volume", y="Predicted_Sales", 
                          labels={"Sales_Volume": "Ventas Reales", "Predicted_Sales": "Predicciones"},
                          title="Predicciones vs Ventas Reales")
    ),

    html.H3("Tabla de predicciones"),
    dash_table.DataTable(
        id="table-predictions",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"}
    )
])

if __name__ == "__main__":
    app.run(debug=True)


