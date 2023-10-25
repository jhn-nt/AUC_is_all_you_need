import dash
import numpy as np
import pandas as pd
import polars as pl
import os
import pyarrow.parquet as pq
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
from dataclasses import dataclass
from typing import Callable

# import functions
from beta_gen import ModelConfig
from model_run import ExperimentConfig, run_experiments, METRICS

# app functions
def load_results(filename: str = 'results/result_model_10.parquet') -> pl.DataFrame:
    """Load experiment results from a parquet file."""
    a= pl.from_arrow(pq.read_table(filename))
    return a.to_pandas().round(2)
    

def get_metric_names():
    return list(METRICS.keys())

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Dash app setup
app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df= load_results()

taskbar_markdown_text = '''
### Beta distributions

Select model configurations to generate distributions.
'''

def create_default_subgroup(index: int) -> html.Div:
    return html.Div([

        html.Div([
            html.Label(f"Subgroup {index + 1}:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
        ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '10px'}),

        html.Div([
            html.Label("PPEV:"),  # Positive Predicted Expected Value abbreviated
            dcc.Slider(
                id={"type": "pos-pred-EV-input", "index": index},
                min=0.1,
                max=0.9,
                value=0.6,
                marks={i/10: str(i/10) for i in range(1, 10) if i % 2 == 0}  # Only every 0.2
            ),
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("PPC:"),  # Positive Predicted Confidence abbreviated
            dcc.Slider(
                id={"type": "pos-pred-confidence-input", "index": index},
                min=1,
                max=5,
                value=2,
                marks={i: str(i) for i in range(1, 6)}
            ),
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("NPEV:"),  # Negative Predicted Expected Value abbreviated
            dcc.Slider(
                id={"type": "neg-pred-EV-input", "index": index},
                min=0.1,
                max=0.9,
                value=0.6,
                marks={i/10: str(i/10) for i in range(1, 10) if i % 2 == 0}  # Only every 0.2
            ),
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Label("NPC:"),  # Negative Predicted Confidence abbreviated
            dcc.Slider(
                id={"type": "neg-pred-confidence-input", "index": index},
                min=1,
                max=5,
                value=2,
                marks={i: str(i) for i in range(1, 6)}
            ),
        ], style={'marginBottom': '10px'}),

    ], style={'display': 'inline-block', 'marginRight': '10px', 'width': '45%', 'verticalAlign': 'top'})

# layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1("Model Results Dashboard", style={"textAlign": "center", 'color': colors['text']}),
    html.Div(
        [
            html.Div(
                [
                    dcc.Markdown(children=taskbar_markdown_text),
                    html.Br(),
                    html.Div(id='subgroups-div', children=[
                        # Set two default subgroups
                        create_default_subgroup(0),
                        create_default_subgroup(1),
                    ]),
                ],
                style={
                    'width': '25%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'borderRight': 'thin lightgrey solid',
                    'paddingRight': '20px',
                    'color': colors['text']
                }
            ),
            html.Div(
                [
                    html.Br(),
                    dcc.Graph(id="subgroup-viz-graph"),
                    html.Br(),
                    html.Div("Model Performance Comparison:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
                    generate_table(df)
                ],
                style={'width': '73%', 'display': 'inline-block', 'color': colors['text']}
            )
        ],
        style={'padding': '20px'}
    )
])

           
@app.callback(
    Output("subgroup-viz-graph", "figure"),
    [
    Input({'type': 'pos-pred-EV-input', 'index': ALL}, "value"),
    Input({'type': 'pos-pred-confidence-input', 'index': ALL}, "value"),
    Input({'type': 'neg-pred-EV-input', 'index': ALL}, "value"),
    Input({'type': 'neg-pred-confidence-input', 'index': ALL}, "value"),
    ]
)
def update_figure(pos_ev_values, pos_conf_values, neg_ev_values, neg_conf_values):
    # Assuming that `ModelConfig` is a class that has a method `subgroup_viz` that 
    # returns a plotly figure based on its properties.
    model_config = ModelConfig(
        pos_pred_EV=pos_ev_values,
        pos_pred_confidence=pos_conf_values,
        neg_pred_EV=neg_ev_values,
        neg_pred_confidence=neg_conf_values
    )
    
    fig = model_config.subgroup_viz()
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
