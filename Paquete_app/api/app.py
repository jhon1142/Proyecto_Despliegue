import json
import joblib
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

from bmw_model_project import config
from bmw_model_project.pipeline import create_features, perform_clustering

# ================================
# 1. Cargar modelo y métricas
# ================================
best_model_path = config.TRAINED_DIR / "best_model.joblib"
best_model = joblib.load(best_model_path)

with open(config.TRAINED_DIR / "best_model_metadata.json") as f:
    metrics = json.load(f)

# ================================
# 2. Cargar y preparar datos
# ================================
df = pd.read_csv(config.DATA_URL, encoding="utf-8")
df = create_features(df)
df = perform_clustering(df)

numeric_features = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD", "age_model"]
categorical_features = ["Region", "Color", "Fuel_Type", "Transmission", "Segmento", "Cluster", "is_luxury"]

# Convertir categóricas a string
df[categorical_features] = df[categorical_features].astype(str)

X = df[numeric_features + categorical_features]
df["Predicted_Sales"] = best_model.predict(X)

# ================================
# 2.1 Calcular percentiles para clasificación
# ================================
P30 = df["Sales_Volume"].quantile(0.30)
P70 = df["Sales_Volume"].quantile(0.70)

# ================================
# 3. Crear app Dash
# ================================
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
    ),

    html.Hr(),

    # ================================
    # 4. Predicción personalizada
    # ================================
    html.H2("Predicción personalizada"),
    html.Div([
        html.Label("Año"),
        dcc.Input(id="inp-year", type="number", value=2020),

        html.Label("Tamaño del motor (L)"),
        dcc.Input(id="inp-engine", type="number", value=2.0, step=0.1),

        html.Label("Kilometraje (KM)"),
        dcc.Input(id="inp-mileage", type="number", value=50000),

        html.Label("Precio USD"),
        dcc.Input(id="inp-price", type="number", value=20000),

        html.Label("Edad del modelo"),
        dcc.Input(id="inp-age", type="number", value=5),

        html.Label("Región"),
        dcc.Dropdown(
            id="inp-region",
            options=[{"label": x, "value": x} for x in sorted(df["Region"].unique())],
            value=sorted(df["Region"].unique())[0]
        ),

        html.Label("Color"),
        dcc.Dropdown(
            id="inp-color",
            options=[{"label": x, "value": x} for x in sorted(df["Color"].unique())],
            value=sorted(df["Color"].unique())[0]
        ),

        html.Label("Combustible"),
        dcc.Dropdown(
            id="inp-fuel",
            options=[{"label": x, "value": x} for x in sorted(df["Fuel_Type"].unique())],
            value=sorted(df["Fuel_Type"].unique())[0]
        ),

        html.Label("Transmisión"),
        dcc.Dropdown(
            id="inp-trans",
            options=[{"label": x, "value": x} for x in sorted(df["Transmission"].unique())],
            value=sorted(df["Transmission"].unique())[0]
        ),

        html.Label("Segmento"),
        dcc.Dropdown(
            id="inp-seg",
            options=[{"label": x, "value": x} for x in sorted(df["Segmento"].unique())],
            value=sorted(df["Segmento"].unique())[0]
        ),

        html.Label("Cluster"),
        dcc.Dropdown(
            id="inp-cluster",
            options=[{"label": x, "value": x} for x in sorted(df["Cluster"].unique())],
            value=sorted(df["Cluster"].unique())[0]
        ),

        html.Label("¿Es lujo?"),
        dcc.Dropdown(
            id="inp-lux",
            options=[{"label": "Sí", "value": 1}, {"label": "No", "value": 0}],
            value=0
        ),

        html.Br(),
        html.Button("Calcular predicción", id="btn-predict", n_clicks=0),
        html.H3(id="prediction-output", style={"margin-top": "20px", "color": "blue"})
    ])
])

# ================================
# 5. Callback para predicción personalizada
# ================================
@app.callback(
    Output("prediction-output", "children"),
    Input("btn-predict", "n_clicks"),
    [
        Input("inp-year", "value"),
        Input("inp-engine", "value"),
        Input("inp-mileage", "value"),
        Input("inp-price", "value"),
        Input("inp-age", "value"),
        Input("inp-region", "value"),
        Input("inp-color", "value"),
        Input("inp-fuel", "value"),
        Input("inp-trans", "value"),
        Input("inp-seg", "value"),
        Input("inp-cluster", "value"),
        Input("inp-lux", "value"),
    ]
)
def make_prediction(n_clicks, year, engine, mileage, price, age,
                    region, color, fuel, trans, seg, cluster, lux):
    if n_clicks == 0:
        return ""

    input_df = pd.DataFrame([{
        "Year": year,
        "Engine_Size_L": engine,
        "Mileage_KM": mileage,
        "Price_USD": price,
        "age_model": age,
        "Region": region,
        "Color": color,
        "Fuel_Type": fuel,
        "Transmission": trans,
        "Segmento": seg,
        "Cluster": cluster,
        "is_luxury": lux
    }])

    # convertir categóricas
    cat_cols = ["Region","Color","Fuel_Type","Transmission","Segmento","Cluster","is_luxury"]
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    try:
        pred = float(best_model.predict(input_df)[0])

        # ======== Clasificación LOW / MEDIUM / HIGH ========
        if pred < P30:
            classification = "LOW"
        elif pred < P70:
            classification = "MEDIUM"
        else:
            classification = "HIGH"
        # ===================================================

        return f"Predicción estimada de ventas: {pred:.2f} unidades — Categoría: {classification}"

    except Exception as e:
        return f"Error en predicción: {str(e)}"

# ================================
# 6. Ejecutar servidor
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)