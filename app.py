import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import random
import pandas as pd

df = pd.read_csv("BMW sales data.csv", parse_dates=False)

modelos = df.Model.unique()
regiones = df.Region.unique()
combustibles = df.Fuel_Type.unique()
transmisiones = df.Transmission.unique()
colores = df.Color.unique()

# Crear la app Dash
app = dash.Dash(__name__)

# Layout del tablero
app.layout = html.Div([
    html.H1("Tablero de Predicción de Ventas BMW (Demo)"),

    html.Div([
        html.Label("Modelo"),
        dcc.Dropdown(id='modelo', options=[{'label': m, 'value': m} for m in modelos], value='X3'),
    ], style={"margin": "10px"}),

    html.Div([
        html.Label("Región"),
        dcc.Dropdown(id='region', options=[{'label': r, 'value': r} for r in regiones], value='Europe'),
    ], style={"margin": "10px"}),

    html.Div([
        html.Label("Tipo de combustible"),
        dcc.Dropdown(id='combustible', options=[{'label': f, 'value': f} for f in combustibles], value='Gasoline'),
    ], style={"margin": "10px"}),

    html.Div([
        html.Label("Transmisión"),
        dcc.Dropdown(id='transmision', options=[{'label': t, 'value': t} for t in transmisiones], value='Automatic'),
    ], style={"margin": "10px"}),

    html.Div([
        html.Label("Color"),
        dcc.Dropdown(id='color', options=[{'label': c, 'value': c} for c in colores], value='Black'),
    ], style={"margin": "10px"}),

    html.Button("Predecir", id='boton-predecir', n_clicks=0),
    
    html.H2("Predicción:"),
    html.Div(id='resultado'),
    html.Div(id='probabilidades')
])

# Callback para predicción inventada 
# Esta parte del código simula una predicción y probabilidades
@app.callback(
    Output("resultado", "children"),
    Output("probabilidades", "children"),
    Input("boton-predecir", "n_clicks"),
    State("modelo", "value"),
    State("region", "value"),
    State("combustible", "value"),
    State("transmision", "value"),
    State("color", "value")
)
def predecir(n_clicks, modelo, region, combustible, transmision, color):
    if n_clicks > 0:
        # Valores inventados de predicción
        clases = ['Alta venta', 'Baja venta']
        prediccion = random.choice(clases)
        # Probabilidades simuladas (sumen 1)
        prob = [random.uniform(0,1) for _ in clases]
        total = sum(prob)
        prob = [round(p/total,2) for p in prob]
        prob_text = ", ".join([f"{c}: {p:.2f}" for c,p in zip(clases, prob)])
        return f"Predicción simulada: {prediccion}", f"Probabilidades simuladas: {prob_text}"
    return "", ""

if __name__ == "__main__":
    app.run(debug=True)
