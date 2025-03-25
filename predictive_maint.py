import pandas as pd

file_path = 'C:\\Users\\WINDOWS\\Desktop\\predictive\\predictive_maintenance.csv'
data = pd.read_csv(file_path)
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = data.drop(['UDI', 'Product ID', 'Type', 'Failure Type'], axis=1) 

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

X = data.drop('Target', axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train[:5], y_train[:5]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
import pandas as pd

failure_probabilities = model.predict_proba(X_test)[:, 1]  

failure_risk_df = pd.DataFrame({
    'Product ID': range(len(X_test)),  
    'Actual Failure': y_test.values, 
    'Failure Probability': failure_probabilities
})

failure_risk_df = failure_risk_df.set_index('Product ID')
failure_risk_df['Actual Failure'] = y_test.reset_index(drop=True)

failure_risk_df = failure_risk_df.sort_values(by='Failure Probability', ascending=False)

top_machines = failure_risk_df.head(10)
print(top_machines)
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd

# app = dash.Dash(__name__)

# failure_probabilities = model.predict_proba(X_test)[:, 1] 

# failure_risk_df = pd.DataFrame({
#     'Product ID': range(len(X_test)),
#     'Actual Failure': y_test.values,
#     'Failure Probability': failure_probabilities
# })


# app.layout = html.Div([
#     html.H1("Machine Failure Prediction Dashboard"),
#     html.Label("Select Number of Machines to Display:"),

#     dcc.Slider(
#         id='num-machines-slider',
#         min=1,
#         max=20,
#         step=1,
#         value=10,  
#         marks={i: str(i) for i in range(1, 21)}
#     ),

#     dcc.Graph(id='failure-risk-graph')
# ])

# @app.callback(
#     Output('failure-risk-graph', 'figure'),
#     [Input('num-machines-slider', 'value')]
# )
# def update_graph(num_machines):
  
#     top_machines = failure_risk_df.head(num_machines)

#     fig = px.scatter(top_machines,
#                      x='Product ID',
#                      y='Failure Probability',
#                      size='Failure Probability',
#                      color='Actual Failure',     
#                      hover_data=['Failure Probability', 'Actual Failure'],
#                      labels={'Failure Probability': 'Predicted Failure Probability',
#                              'Actual Failure': 'Actual Failure Status'},
#                      title=f'Top {num_machines} Machines with Highest Failure Risk')


#     fig.update_layout(
#         xaxis_title="Machine Index",
#         yaxis_title="Predicted Failure Probability",
#         coloraxis_colorbar=dict(
#             title="Actual Failure",
#             tickvals=[0, 1],
#             ticktext=['No Failure', 'Failure']
#         ),
#         template='plotly_dark'  
#     )

#     return fig


# if __name__ == '__main__':
#     app.run_server(debug=True)










# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd
# import numpy as np

# app = dash.Dash(__name__)

# failure_probabilities = model.predict_proba(X_test)[:, 1]
# failure_risk_df = pd.DataFrame({
#     'Product ID': range(len(X_test)),
#     'Actual Failure': y_test.values,
#     'Failure Probability': failure_probabilities
# })

# feature_importances = model.feature_importances_
# importance_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': feature_importances
# }).sort_values(by='Importance', ascending=False)
# app.layout = html.Div([
#     html.H1("Machine Failure Prediction Dashboard"),
    
#     html.Label("Filter by Actual Failure Status:"),
#     dcc.Dropdown(
#         id='failure-status-filter',
#         options=[
#             {'label': 'All', 'value': 'All'},
#             {'label': 'Failure', 'value': 1},
#             {'label': 'No Failure', 'value': 0}
#         ],
#         value='All',
#         clearable=False
#     ),
    
#     html.Label("Select Number of Machines to Display:"),
#     dcc.Slider(
#         id='num-machines-slider',
#         min=1,
#         max=20,
#         step=1,
#         value=10,
#         marks={i: str(i) for i in range(1, 21)}
#     ),
    
#     dcc.Graph(id='failure-risk-graph'),
    
#     html.H3("Feature Importance:"),
#     dcc.Graph(
#         id='feature-importance-graph',
#         figure=px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
#     ),
    
#     html.H3("Detailed Insights for Selected Machine:"),
#     dcc.Dropdown(
#         id='machine-selector',
#         options=[{'label': f'Machine {i}', 'value': i} for i in failure_risk_df['Product ID']],
#         placeholder="Select a Machine",
#         clearable=True
#     ),
#     dcc.Graph(id='machine-insights-graph')
# ])

# @app.callback(
#     Output('failure-risk-graph', 'figure'),
#     [Input('num-machines-slider', 'value'),
#      Input('failure-status-filter', 'value')]
# )
# def update_graph(num_machines, status_filter):
#     filtered_df = failure_risk_df.copy()
    
#     if status_filter != 'All':
#         filtered_df = filtered_df[filtered_df['Actual Failure'] == int(status_filter)]
    
#     top_machines = filtered_df.head(num_machines)
#     fig = px.scatter(
#         top_machines,
#         x='Product ID',
#         y='Failure Probability',
#         size='Failure Probability',
#         color='Actual Failure',
#         hover_data=['Failure Probability', 'Actual Failure'],
#         labels={'Failure Probability': 'Predicted Failure Probability',
#                 'Actual Failure': 'Actual Failure Status'},
#         title=f'Top {num_machines} Machines with Highest Failure Risk'
#     )
#     fig.update_layout(
#         xaxis_title="Machine Index",
#         yaxis_title="Predicted Failure Probability",
#         template='plotly_dark'
#     )
#     return fig
# @app.callback(
#     Output('machine-insights-graph', 'figure'),
#     [Input('machine-selector', 'value')]
# )
# def display_machine_insights(selected_machine):
#     if selected_machine is None:
#         return px.line(title="Select a machine to view detailed insights.")
#     np.random.seed(selected_machine)
#     time_series = pd.DataFrame({
#         'Time': np.arange(1, 11),
#         'Temperature': np.random.uniform(280, 320, 10),
#         'Torque': np.random.uniform(50, 100, 10),
#         'Wear': np.random.uniform(0, 100, 10)
#     })
    
#     fig = px.line(
#         time_series, 
#         x='Time', 
#         y=['Temperature', 'Torque', 'Wear'], 
#         title=f"Trends for Machine {selected_machine}",
#         labels={'value': 'Measurement', 'variable': 'Feature'}
#     )
#     fig.update_layout(
#         xaxis_title="Time",
#         yaxis_title="Measurement",
#         template='plotly_dark'
#     )
#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=True)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import io
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

failure_probabilities = model.predict_proba(X_test)[:, 1]
failure_risk_df = pd.DataFrame({
    'Product ID': range(len(X_test)),
    'Actual Failure': y_test.values,
    'Failure Probability': failure_probabilities
})

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


app.layout = html.Div([
    html.H1("Machine Failure Prediction Dashboard", style={'textAlign': 'center'}),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filter by Failure Status:"),
            dcc.Dropdown(
                id='failure-status-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Failure', 'value': 1},
                    {'label': 'No Failure', 'value': 0}
                ],
                value='All',
                clearable=False
            )
        ], width=4),
        dbc.Col([
            html.Label("Number of Machines to Display:"),
            dcc.Slider(
                id='num-machines-slider',
                min=1,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(1, 21)}
            )
        ], width=8)
    ]),
    
    dcc.Graph(id='failure-risk-graph'),
    
    dbc.Row([
        dbc.Col([
            html.H3("Summary Statistics"),
            html.Div(id='summary-statistics')
        ], width=4),
        dbc.Col([
            html.H3("Feature Importance"),
            dcc.Graph(
                id='feature-importance-graph',
                figure=px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
            )
        ], width=8)
    ]),
    
    html.H3("Detailed Insights for Selected Machine:"),
    dcc.Dropdown(
        id='machine-selector',
        options=[{'label': f'Machine {i}', 'value': i} for i in failure_risk_df['Product ID']],
        placeholder="Select a Machine",
        clearable=True
    ),
    dcc.Graph(id='machine-insights-graph'),
    
    html.H3("Provide Feedback on Machine Predictions:"),
    dcc.Textarea(
        id='feedback-box',
        placeholder="Enter feedback...",
        style={'width': '100%', 'height': '100px'}
    ),
    html.Button("Submit Feedback", id='submit-feedback', n_clicks=0),
    html.Div(id='feedback-display', style={'marginTop': '20px'}),
    
    html.H3("Download Report:"),
    html.A(
        "Download Failure Risk Report",
        id="download-link",
        href="",
        download="failure_risk_report.csv",
        target="_blank",
        style={'color': '#17a2b8', 'fontSize': '16px'}
    )
])

# Callbacks
@app.callback(
    Output('failure-risk-graph', 'figure'),
    [Input('num-machines-slider', 'value'),
     Input('failure-status-filter', 'value')]
)
def update_graph(num_machines, status_filter):
    filtered_df = failure_risk_df.copy()
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Actual Failure'] == int(status_filter)]
    
    top_machines = filtered_df.head(num_machines)
    fig = px.scatter(
        top_machines,
        x='Product ID',
        y='Failure Probability',
        size='Failure Probability',
        color='Actual Failure',
        hover_data=['Failure Probability', 'Actual Failure'],
        title=f'Top {num_machines} Machines with Highest Failure Risk'
    )
    fig.update_layout(
        xaxis_title="Machine Index",
        yaxis_title="Predicted Failure Probability",
        template='plotly_dark'
    )
    return fig

@app.callback(
    Output('summary-statistics', 'children'),
    [Input('failure-status-filter', 'value')]
)
def update_summary(status_filter):
    filtered_df = failure_risk_df.copy()
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Actual Failure'] == int(status_filter)]
    
    avg_probability = filtered_df['Failure Probability'].mean()
    total_failures = filtered_df['Actual Failure'].sum()
    
    return html.Div([
        html.P(f"Average Failure Probability: {avg_probability:.2f}"),
        html.P(f"Total Failures: {total_failures}")
    ])

@app.callback(
    Output('machine-insights-graph', 'figure'),
    [Input('machine-selector', 'value')]
)
def display_machine_insights(selected_machine):
    if selected_machine is None:
        raise PreventUpdate
    
    np.random.seed(selected_machine)
    time_series = pd.DataFrame({
        'Time': np.arange(1, 11),
        'Temperature': np.random.uniform(280, 320, 10),
        'Torque': np.random.uniform(50, 100, 10),
        'Wear': np.random.uniform(0, 100, 10)
    })
    
    fig = px.line(
        time_series, 
        x='Time', 
        y=['Temperature', 'Torque', 'Wear'], 
        title=f"Trends for Machine {selected_machine}",
        labels={'value': 'Measurement', 'variable': 'Feature'}
    )
    fig.update_layout(template='plotly_dark')
    return fig

@app.callback(
    Output('feedback-display', 'children'),
    [Input('submit-feedback', 'n_clicks')],
    [State('feedback-box', 'value')]
)
def display_feedback(n_clicks, feedback):
    if n_clicks > 0 and feedback:
        return html.Div([
            html.H5("Your Feedback:"),
            html.P(feedback)
        ])
    return ""

@app.callback(
    Output('download-link', 'href'),
    [Input('failure-status-filter', 'value')]
)
def generate_csv_download_link(status_filter):
    filtered_df = failure_risk_df.copy()
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Actual Failure'] == int(status_filter)]
    
    csv_string = filtered_df.to_csv(index=False, encoding='utf-8')
    return "data:text/csv;charset=utf-8," + csv_string

if __name__ == '__main__':
    app.run_server(debug=True)
