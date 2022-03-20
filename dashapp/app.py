import pandas as pd
import plotly.express as px  
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

def get_data():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv")
    return data

def clean_data(data):
    data.columns = [i.lower().replace(" ","_") for i in data.columns]
    data["target"] = data.apply(lambda df: True if df["class"]=="Positive" else False, axis=1)
    data = data.drop(columns=["class"])
    feature_cols = [i for i in data.columns if "target" not in i]
    bool_cols = [i for i in feature_cols if "age" not in i and "gender" not in i]
    data[bool_cols] = data[bool_cols] == "Yes"
    data["is_male"] = data.gender == "Male"
    data = data.drop(columns=["gender"])
    return data

def get_model(data):
    y = data.target
    X = data[[i for i in data.columns if "target" not in i]]
    model = RandomForestClassifier(n_estimators=50,
                class_weight='balanced_subsample',
                max_depth=8)
    model = model.fit(X, y)
    return model



data = get_data()
training_data = clean_data(data)

external_stylesheets = ["https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css"]

app = Dash(name="Dash-demo", external_stylesheets=external_stylesheets,)

# ------------------------------------------------------------------------------
# App layout
def build_navbar():
    nav = html.Div(className="navbar",
                        children=[
                            html.Div(className="navbar-brand",
                                    children=[
                                        html.A(className="navbar-item",
                                               children=[
                                                         html.P(className="subtitle", 
                                                         children=['Forecast']

                                                         ),
                                                        ]
                                                )
                                            ]
                                    )
                                ]
                  )
    return nav
    

def get_tab2():
    return html.Div(children=[
         html.Div(className="container",
                children=[html.Div(className="columns",
                                  children=[
                                    html.Div(className="column", 
                                            children = [
                                    html.Div([
                                            "Symptoms",
                                            dcc.Checklist(
                                                options=[
                                                {'label': 'Polyuria  ', 'value': 'polyuria'},
                                                {'label': 'Polydipsia  ', 'value': 'polydipsia'},
                                                {'label': 'Sudden Weight Loss  ', 'value': 'sudden_weight_loss'},
                                                {'label': 'Weakness  ', 'value': 'weakness'},
                                                {'label': 'Polyphagia  ', 'value': 'polyphagia'},
                                                {'label': 'Genital Thrush  ', 'value': 'genital_thrush'},
                                                {'label': 'Visual Blurring  ', 'value': 'visual_blurring'},
                                                {'label': 'Itching  ', 'value': 'itching'},
                                                {'label': 'Irritability  ', 'value': 'irritability'},
                                                {'label': 'Delayed Healing  ', 'value': 'delayed_healing'},
                                                {'label': 'Partial Paresis  ', 'value': 'partial_paresis'},
                                                {'label': 'Muscle Stiffness  ', 'value': 'muscle_stiffness'},
                                                {'label': 'Alopecia  ', 'value': 'alopecia'},
                                                {'label': 'Obesity  ', 'value': 'obesity'},
                                                {'label': 'Is Male  ', 'value': 'is_male'},
                                            ],
                                            id='input-on-Symptoms',
                                            className="checkbox"
                                            ),

                                      ]),
                                    html.Br(),
                                    html.Div([
                                            "Age",
                                            dcc.Input(className="input",id='input-on-age', type='number'),
                                            ]),
                                    html.Br(),

                                    html.Button('Predict', className="button is-info",
                                                                            id='submit-val', n_clicks=0),
                                                            html.Br(),
                                                            html.Br(),

                                                            dcc.Loading(
                                                                id="ls-loading-2",
                                                                children=[html.Div([html.Div(id="ls-loading-output-2")])],
                                                                type="circle",
                                                            )
                                    ]),
                                html.Div(className="column", 
                                            children = [

                     
                                            ])]),
                        ])
                        
                        ])


def build_layout():

    layout =html.Div(
                [
                dcc.Store(id='tab1-datatable-value'),
                dcc.Store(id='tab1-datatable-data-raw'),
                dcc.Store(id='tab2-status-datatable'),

                build_navbar(),

                dcc.Tabs(
                        id="tabs-with-classes",
                        parent_className='custom-tabs',
                        className='tabs is-centered is-boxed',
                        children=[
                                dcc.Tab(label='Forecast Summary', 
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                       ),
                                dcc.Tab(label='Prediction', 
                                        className='custom-tab',
                                        selected_className='custom-tab--selected',
                                        children=get_tab2()),
                            ]),
                html.Footer(className="footer",
                            children=[
                                html.Div(className="content has-text-centered", 
                                children=["Copyright © 2022 UAlbany Demo"])
                            ]
                            )
                ])
    return layout


@app.callback(Output("ls-loading-output-2", "children"), 
    [Input(component_id='input-on-Symptoms', component_property='value'),
     Input(component_id='input-on-age', component_property='value'),
     Input('submit-val', 'n_clicks'),]
     )
def input_triggers_nested(selected_symptoms,age,n_clicks):
    predicted = "NA"

    if n_clicks > 0:
        try:
            model = get_model(training_data)

            cols = ['polydipsia', 'obesity', 'polyuria', 'sudden_weight_loss', 
                                'irritability', 'weakness', 'polyphagia', 'genital_thrush', 
                                'visual_blurring', 'muscle_stiffness', 'partial_paresis', 
                                'delayed_healing', 'itching', 'alopecia', 'is_male']

            true_dict = {c:c==s for  c in cols for s in selected_symptoms if c==s}
            false_dict = {i:False for i in cols if i not in true_dict.keys()}
            final_dict = {"age": age}
            final_dict.update(true_dict)
            final_dict.update(false_dict)
            predicted = model.predict(pd.DataFrame.from_dict(final_dict,orient='index').T)[0]
            print(predicted)
            predict_prob = model.predict_proba(pd.DataFrame.from_dict(final_dict,orient='index').T)
            print(predict_prob)
        except:
            predicted = "Error: age is required."
    return f"Predicted Value: {predicted}"




app.layout = build_layout()



# ------------------------------------------------------------------------------
if __name__ == '__main__':

    app.run_server(debug=True)
