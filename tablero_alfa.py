import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import random

import pandas as pd

df = pd.read_csv("BMW sales data.csv", parse_dates=False)

# Supongamos que tenemos algunas opciones de la base BMW
modelos = ['X1', 'X3', 'X5', '3 Series', '5 Series']
regiones = ['Europe', 'Asia', 'North America']
combustibles = ['Gasoline', 'Diesel', 'Electric', 'Hybrid']
transmisiones = ['Manual', 'Automatic']
colores = ['Black', 'White', 'Blue', 'Red']

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
