import pandas as pd
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State
import dash.dash_table
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import sys


print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")


# SUBTASK-4
# --------------------------------
# Data loading and preprocessing function
def load_and_preprocess_data():
    try:
        df_top = pd.read_excel("Audi/TOP_consumers_engine_2024.xlsx", header=None)
        df_water = pd.read_excel("Audi/Water_consumption_AH_2024.xlsx", header=None)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        return None, None, None

    measurement_points = []
    for i in range(4, len(df_water)):
        row = df_water.iloc[i]
        if pd.notna(row[0]):
            point = {
                'ID': row[0],
                'Production_Hall': row[1],
                'Short_Name': row[2],
                'Unit': row[3],
                'Water_Type': row[4],
                'Diameter': row[5]
            }
            measurement_points.append(point)
    
    consumption_data = []
    months = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i in range(4, len(df_water)):
        row = df_water.iloc[i]
        if pd.notna(row[1]):  
            for j, month in enumerate(months):
                col_idx = 6 + j
                if col_idx < len(row) and pd.notna(row[col_idx]):
                    try:
                        consumption = {
                            'Production_Hall': row[1],
                            'Short_Name': str(row[2]) if pd.notna(row[2]) else 'Unknown',
                            'Month': month,
                            'Year': 2023 if j == 0 else 2024,
                            'Consumption_m3': float(row[col_idx])
                        }
                        consumption_data.append(consumption)
                    except Exception as e:
                        print(f"Skipping row {i} col {col_idx} due to error: {e}")
                        continue
    
    df_consumption = pd.DataFrame(consumption_data)
    
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df_consumption['Month_Num'] = df_consumption['Month'].map(month_map)
    df_consumption['Date'] = df_consumption.apply(
        lambda x: datetime(x['Year'], x['Month_Num'], 1), axis=1)
    
    return df_top, df_water, df_consumption

# --------------------------------
# Model class
class WaterConsumptionModel:
    def __init__(self, data):
        self.data = data.copy()
        self.normalize_data()
        
    def normalize_data(self):
        self.data['Season'] = self.data['Month_Num'].apply(self.get_season)
        
    def get_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def calculate_yearly_consumption(self):
        return self.data.groupby('Year')['Consumption_m3'].sum().reset_index()
    
    def calculate_monthly_consumption(self):
        monthly = self.data.groupby(['Year', 'Month_Num', 'Month'])['Consumption_m3'].sum().reset_index()
        monthly = monthly.sort_values(['Year', 'Month_Num'])
        return monthly
    
    def top_consumers(self, n=5):
        return self.data.groupby('Short_Name')['Consumption_m3'].sum().nlargest(n).reset_index()
    
    def consumption_by_hall(self):
        return self.data.groupby('Production_Hall')['Consumption_m3'].sum().reset_index()
    
    def consumption_by_season(self):
        return self.data.groupby(['Production_Hall', 'Season'])['Consumption_m3'].sum().reset_index()
    
    def detect_anomalies(self, threshold=2):
        grouped = self.data.groupby(['Production_Hall', 'Short_Name'])
        mean = grouped['Consumption_m3'].transform('mean')
        std = grouped['Consumption_m3'].transform('std').replace(0, np.nan)
        self.data['Z_Score'] = (self.data['Consumption_m3'] - mean) / std
        
        anomalies = self.data[self.data['Z_Score'].abs() > threshold].copy()
        anomalies.dropna(subset=['Z_Score'], inplace=True)
        return anomalies

# --------------------------------
# Digital Twin class
class WaterSystemDigitalTwin:
    def __init__(self, historical_data):
        self.historical_data = historical_data.copy()
        self.current_state = self.initialize_state()
        self.simulation_results = None
        self.weather = {'temperature': 20}  # Default weather
        
    def initialize_state(self):
        state = {
            'production_halls': {},
            'water_sources': {
                'Brunn 1': {'capacity': 50000, 'current_level': 40000},
                'Brunn 2': {'capacity': 50000, 'current_level': 45000},
                'Municipal': {'capacity': float('inf'), 'current_level': float('inf')}
            },
            'distribution_network': {
                'pipes': {},
                'valves': {}
            }
        }
        
        for hall in self.historical_data['Production_Hall'].unique():
            if pd.isna(hall):
                continue
                
            hall_data = self.historical_data[self.historical_data['Production_Hall'] == hall]
            avg_consumption = hall_data['Consumption_m3'].mean()
            
            state['production_halls'][hall] = {
                'average_consumption': avg_consumption,
                'current_consumption': avg_consumption,
                'status': 'normal',
                'water_source': 'Brunn 1'  # Default source
            }
        
        return state
    
    def simulate_day(self, production_plan, weather=None):
        results = []
        
        if weather:
            self.weather = weather
        
        for hall, plan in production_plan.items():
            if hall not in self.current_state['production_halls']:
                print(f"Warning: Production hall '{hall}' not found in historical data.")
                continue
            
            base_consumption = self.current_state['production_halls'][hall]['average_consumption']
            
            consumption_factor = plan.get('production_level', 1)
            # Weather-based consumption adjustment
            if self.weather.get('temperature', 0) > 25:
                consumption_factor *= 1.2
            
            daily_consumption = base_consumption * consumption_factor
            
            source = plan.get('water_source', 'Brunn 1')
            self.current_state['production_halls'][hall]['water_source'] = source
            
            if source in self.current_state['water_sources']:
                prev_level = self.current_state['water_sources'][source]['current_level']
                new_level = prev_level - daily_consumption
                
                if new_level < 0:
                    daily_consumption = prev_level  # Use remaining water
                    new_level = 0
                    self.current_state['production_halls'][hall]['status'] = 'critical'
                else:
                    self.current_state['water_sources'][source]['current_level'] = new_level
                
                # Update status based on new level
                capacity = self.current_state['water_sources'][source]['capacity']
                if new_level < 0.1 * capacity:
                    self.current_state['production_halls'][hall]['status'] = 'critical'
                elif new_level < 0.2 * capacity:
                    self.current_state['production_halls'][hall]['status'] = 'warning'
                else:
                    self.current_state['production_halls'][hall]['status'] = 'normal'
            else:
                self.current_state['production_halls'][hall]['status'] = 'warning'
            
            results.append({
                'hall': hall,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'consumption': daily_consumption,
                'status': self.current_state['production_halls'][hall]['status'],
                'source': source,
                'source_level': self.current_state['water_sources'].get(source, {}).get('current_level', None),
                'production_level': consumption_factor
            })
        
        self.simulation_results = pd.DataFrame(results)
        return self.simulation_results

    def optimize_water_distribution(self):
        critical_halls = [k for k, v in self.current_state['production_halls'].items() 
                         if v['status'] == 'critical']
        
        for hall in critical_halls:
            current_source = self.current_state['production_halls'][hall]['water_source']
            alternative_sources = [k for k, v in self.current_state['water_sources'].items() 
                                 if v['current_level'] > 10000 and k != current_source]
            
            if alternative_sources:
                new_source = alternative_sources[0]
                self.current_state['production_halls'][hall]['water_source'] = new_source
                self.current_state['production_halls'][hall]['status'] = 'normal'
                print(f"Optimized: {hall} switched to {new_source}")

# --------------------------------
# Load data
df_top, df_water, df_consumption = load_and_preprocess_data()
if df_consumption is None:
    raise Exception("Data loading failed. Please check the files.")

# Clean data - ensure Short_Name is string type
df_consumption['Short_Name'] = df_consumption['Short_Name'].astype(str)

# Create model
water_model = WaterConsumptionModel(df_consumption)

# Create Digital Twin
digital_twin = WaterSystemDigitalTwin(df_consumption)

# --------------------------------
# Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("Audi Hungaria Water Consumption Monitoring Panel", 
            style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#003366', 'padding': '20px'}),
    
    dcc.Tabs(id="main-tabs", value='general-tab', children=[
        dcc.Tab(label='General Overview', value='general-tab', children=[
            html.Div([
                dcc.Graph(id='yearly-consumption'),
                dcc.Graph(id='monthly-trend'),
                dcc.Graph(id='top-consumers')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Production Halls', value='halls-tab', children=[
            html.Div([
                dcc.Dropdown(
                    id='hall-selector',
                    options=[{'label': hall, 'value': hall} 
                            for hall in sorted(df_consumption['Production_Hall'].dropna().unique())],
                    value=sorted(df_consumption['Production_Hall'].dropna().unique())[0],
                    style={'width': '50%', 'margin': '20px auto'}
                ),
                dcc.Graph(id='hall-consumption'),
                dcc.Graph(id='seasonal-consumption'),
                dcc.Graph(id='hall-anomalies')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Anomaly Detection', value='anomaly-tab', children=[
            html.Div([
                html.H3("Anomaly Detection Settings", style={'textAlign': 'center'}),
                html.Div([
                    html.Label("Select Z-Score Threshold:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='anomaly-threshold',
                        min=1,
                        max=5,
                        step=0.1,
                        value=2.5,
                        marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'margin': '30px auto', 'width': '80%'}),
                
                dcc.Graph(id='anomaly-detection'),
                html.H4("Detected Anomalies", style={'textAlign': 'center', 'marginTop': '30px'}),
                html.Div(id='anomaly-table', style={'margin': '20px auto', 'width': '90%'})
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Comparisons', value='compare-tab', children=[
            html.Div([
                html.H3("Compare Measurement Points", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='compare-selector',
                    options=[{'label': name, 'value': name} 
                            for name in sorted(df_consumption['Short_Name'].dropna().unique())],
                    multi=True,
                    value=['Brunn 1', 'Brunn 2'],
                    style={'width': '80%', 'margin': '20px auto'}
                ),
                dcc.Graph(id='comparison-chart'),
                dcc.Graph(id='comparison-trend')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Digital Twin', value='digital-twin-tab', children=[
            html.Div([
                html.H2("Water System Digital Twin Simulation", style={'textAlign': 'center'}),
                
                # Simulation controls
                html.Div([
                    html.Div([
                        html.Label("Production Hall:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dt-hall-dropdown',
                            options=[{'label': hall, 'value': hall} 
                                    for hall in sorted(df_consumption['Production_Hall'].dropna().unique())],
                            value=sorted(df_consumption['Production_Hall'].dropna().unique())[0],
                            style={'width': '100%'}
                        ),
                    ], style={'width': '30%', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("Production Level Multiplier:", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='dt-production-level',
                            min=0,
                            max=2,
                            step=0.1,
                            value=1,
                            marks={0: '0', 1: '1', 2: '2'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={'width': '30%', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("Water Source:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dt-water-source',
                            options=[
                                {'label': 'Brunn 1', 'value': 'Brunn 1'},
                                {'label': 'Brunn 2', 'value': 'Brunn 2'},
                                {'label': 'Municipal', 'value': 'Municipal'}
                            ],
                            value='Brunn 1',
                            style={'width': '100%'}
                        ),
                    ], style={'width': '30%', 'padding': '10px'}),
                ], style={'display': 'flex', 'justifyContent': 'center', 'margin': '20px 0'}),
                
                # Weather controls
                html.Div([
                    html.Label("Simulated Weather Conditions", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Slider(
                        id='dt-weather-temp',
                        min=0,
                        max=40,
                        step=1,
                        value=20,
                        marks={0: '0°C', 20: '20°C', 40: '40°C'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '80%', 'margin': '20px auto'}),
                
                html.Button('Run Simulation', id='simulate-button', n_clicks=0, 
                          style={'margin': '20px auto', 'display': 'block', 'padding': '10px 20px'}),
                
                # Results display
                html.Div(id='dt-status-message', style={
                    'margin': '20px', 
                    'fontSize': '18px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }),
                
                html.Div([
                    dcc.Graph(id='dt-simulation-results'),
                    dcc.Graph(id='dt-source-levels'),
                ], style={'display': 'flex'}),
                
                html.Div([
                    dcc.Graph(id='dt-consumption-history'),
                    dcc.Graph(id='dt-source-history'),
                ], style={'display': 'flex'})
            ], style={'padding': '20px'})
        ])
    ]),
    
    # Hidden div to store simulation history
    html.Div(id='simulation-history-store', style={'display': 'none'})
], style={'backgroundColor': '#f0f2f5'})

# --------------------------------
# Callbacks

@app.callback(
    [Output('yearly-consumption', 'figure'),
     Output('monthly-trend', 'figure'),
     Output('top-consumers', 'figure')],
    Input('main-tabs', 'value'),
    prevent_initial_call=True
)
def update_general_overview(tab):
    if tab != 'general-tab':
        raise PreventUpdate
    
    # Yearly consumption
    yearly = water_model.calculate_yearly_consumption()
    fig_yearly = px.bar(yearly, x='Year', y='Consumption_m3', 
                       title='Annual Total Water Consumption',
                       labels={'Consumption_m3': 'Consumption (m³)'},
                       color='Year',
                       template='plotly_white')
    
    # Monthly trend
    monthly = water_model.calculate_monthly_consumption()
    fig_monthly = px.line(monthly, x='Month', y='Consumption_m3', color='Year',
                         title='Monthly Consumption Trends',
                         labels={'Consumption_m3': 'Consumption (m³)'},
                         template='plotly_white')
    
    # Top consumers
    top = water_model.top_consumers(10)
    fig_top = px.bar(top, x='Short_Name', y='Consumption_m3',
                    title='Top 10 Water Consumers',
                    labels={'Consumption_m3': 'Total Consumption (m³)', 'Short_Name': 'Measurement Point'},
                    color='Consumption_m3',
                    template='plotly_white')
    
    return fig_yearly, fig_monthly, fig_top

@app.callback(
    [Output('hall-consumption', 'figure'),
     Output('seasonal-consumption', 'figure'),
     Output('hall-anomalies', 'figure')],
    [Input('hall-selector', 'value')]
)
def update_hall_visualizations(hall):
    # Monthly consumption for the hall
    hall_data = df_consumption[df_consumption['Production_Hall'] == hall]
    monthly = hall_data.groupby(['Month', 'Month_Num'])['Consumption_m3'].sum().reset_index().sort_values('Month_Num')
    
    fig_hall = px.bar(monthly, x='Month', y='Consumption_m3',
                     title=f'{hall} - Monthly Water Consumption',
                     labels={'Consumption_m3': 'Consumption (m³)'},
                     template='plotly_white')
    
    # Seasonal consumption
    seasonal = water_model.consumption_by_season()
    seasonal = seasonal[seasonal['Production_Hall'] == hall]
    
    fig_seasonal = px.pie(seasonal, values='Consumption_m3', names='Season',
                         title=f'{hall} - Seasonal Consumption Distribution',
                         template='plotly_white')
    
    # Anomalies for the hall
    anomalies = water_model.detect_anomalies(2.5)
    hall_anomalies = anomalies[anomalies['Production_Hall'] == hall]
    
    fig_anomalies = px.scatter(hall_anomalies, x='Date', y='Consumption_m3',
                              color='Z_Score', hover_data=['Short_Name'],
                              title=f'{hall} - Detected Anomalies',
                              labels={'Consumption_m3': 'Consumption (m³)'},
                              template='plotly_white')
    fig_anomalies.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    
    return fig_hall, fig_seasonal, fig_anomalies

@app.callback(
    [Output('anomaly-detection', 'figure'),
     Output('anomaly-table', 'children')],
    [Input('anomaly-threshold', 'value'),
     Input('main-tabs', 'value')]
)
def update_anomaly_detection(threshold, tab):
    if tab != 'anomaly-tab':
        raise PreventUpdate
    
    anomalies = water_model.detect_anomalies(threshold)
    
    # Create scatter plot
    fig = px.scatter(
        water_model.data, 
        x='Date', 
        y='Consumption_m3',
        color='Z_Score',
        color_continuous_scale=['green', 'yellow', 'red'],
        range_color=[0, 5],
        hover_data=['Short_Name', 'Production_Hall'],
        title=f'Anomaly Detection (Z-Score Threshold: {threshold})',
        labels={'Consumption_m3': 'Consumption (m³)'},
        template='plotly_white'
    )
    
    # Add threshold reference line
    fig.update_layout(
        shapes=[
            dict(
                type='line',
                yref='paper',
                y0=0,
                y1=1,
                x0=water_model.data['Date'].min(),
                x1=water_model.data['Date'].max(),
                line=dict(color='red', dash='dash'))
        ]
    )
    
    # Create table with anomalies
    if not anomalies.empty:
        table = dash.dash_table.DataTable(
            columns=[
                {"name": "Date", "id": "Date"},
                {"name": "Production Hall", "id": "Production_Hall"},
                {"name": "Measurement Point", "id": "Short_Name"},
                {"name": "Consumption (m³)", "id": "Consumption_m3"},
                {"name": "Z-Score", "id": "Z_Score", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            data=anomalies.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Z_Score} > 3',
                        'column_id': 'Z_Score'
                    },
                    'backgroundColor': 'red',
                    'color': 'white'
                },
                {
                    'if': {
                        'filter_query': '{Z_Score} < -3',
                        'column_id': 'Z_Score'
                    },
                    'backgroundColor': 'blue',
                    'color': 'white'
                }
            ],
            filter_action="native",
            sort_action="native",
            page_size=10
        )
    else:
        table = html.Div("No anomalies detected with current threshold", 
                        style={'color': 'green', 'fontSize': '18px', 'textAlign': 'center'})
    
    return fig, table

@app.callback(
    [Output('comparison-chart', 'figure'),
     Output('comparison-trend', 'figure')],
    [Input('compare-selector', 'value'),
     Input('main-tabs', 'value')]
)
def update_comparison_charts(selected_names, tab):
    if tab != 'compare-tab':
        raise PreventUpdate
    
    if not selected_names:
        empty_fig = px.line(title="Please select at least one measurement point")
        return empty_fig, empty_fig
    
    filtered = df_consumption[df_consumption['Short_Name'].isin(selected_names)]
    
    # Total consumption comparison
    total_consumption = filtered.groupby('Short_Name')['Consumption_m3'].sum().reset_index()
    fig_total = px.bar(total_consumption, x='Short_Name', y='Consumption_m3',
                      color='Short_Name',
                      title='Total Water Consumption Comparison',
                      labels={'Consumption_m3': 'Total Consumption (m³)', 'Short_Name': 'Measurement Point'},
                      template='plotly_white')
    
    # Monthly trend comparison
    monthly = filtered.groupby(['Short_Name', 'Month', 'Month_Num'])['Consumption_m3'].mean().reset_index()
    monthly = monthly.sort_values(['Short_Name', 'Month_Num'])
    
    fig_trend = px.line(monthly, x='Month', y='Consumption_m3', color='Short_Name',
                       title='Monthly Consumption Trends',
                       labels={'Consumption_m3': 'Average Consumption (m³)'},
                       template='plotly_white')
    
    return fig_total, fig_trend

# Digital Twin callbacks
simulation_history = []

@app.callback(
    [Output('dt-simulation-results', 'figure'),
     Output('dt-source-levels', 'figure'),
     Output('dt-consumption-history', 'figure'),
     Output('dt-source-history', 'figure'),
     Output('dt-status-message', 'children'),
     Output('simulation-history-store', 'children')],
    [Input('simulate-button', 'n_clicks')],
    [State('dt-hall-dropdown', 'value'),
     State('dt-production-level', 'value'),
     State('dt-water-source', 'value'),
     State('dt-weather-temp', 'value'),
     State('simulation-history-store', 'children')]
)
def run_digital_twin_simulation(n_clicks, hall, prod_level, water_source, temp, history_json):
    if n_clicks == 0:
        # Initial state - empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Run simulation to see results",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No simulation data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, "Configure and run the simulation", dash.no_update
    
    # Prepare production plan
    production_plan = {
        hall: {
            'production_level': prod_level,
            'water_source': water_source
        }
    }
    
    # Run simulation with weather data
    weather = {'temperature': temp}
    sim_results = digital_twin.simulate_day(production_plan, weather)
    digital_twin.optimize_water_distribution()
    
    # Store simulation results in history
    sim_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hall': hall,
        'consumption': sim_results['consumption'].iloc[0],
        'status': sim_results['status'].iloc[0],
        'source': water_source,
        'source_level': sim_results['source_level'].iloc[0],
        'production_level': prod_level,
        'weather_temp': temp
    }
    
    if history_json:
        simulation_history = pd.read_json(history_json, orient='split')
    else:
        simulation_history = pd.DataFrame(columns=sim_data.keys())
    
    simulation_history = pd.concat([simulation_history, pd.DataFrame([sim_data])], ignore_index=True)
    
    # Create consumption figure
    consumption_fig = px.bar(
        sim_results, 
        x='hall', 
        y='consumption', 
        color='status',
        title=f'Daily Water Consumption - {hall}',
        labels={'hall': 'Production Hall', 'consumption': 'Consumption (m³)'},
        color_discrete_map={
            'normal': 'green', 
            'warning': 'orange', 
            'critical': 'red'
        },
        template='plotly_white'
    )
    
    # Create water source levels figure
    sources = pd.DataFrame.from_dict(digital_twin.current_state['water_sources'], orient='index')
    sources.reset_index(inplace=True)
    sources.rename(columns={'index': 'source'}, inplace=True)
    
    source_fig = px.bar(
        sources,
        x='source',
        y='current_level',
        title='Current Water Source Levels',
        labels={'current_level': 'Remaining Volume (m³)', 'source': 'Water Source'},
        text='current_level',
        color='source',
        template='plotly_white'
    )
    source_fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    source_fig.update_layout(showlegend=False)
    
    # Create consumption history figure
    if len(simulation_history) > 0:
        history_fig = px.line(
            simulation_history,
            x='timestamp',
            y='consumption',
            color='hall',
            title='Consumption History',
            labels={'consumption': 'Consumption (m³)', 'timestamp': 'Time'},
            template='plotly_white'
        )
    else:
        history_fig = go.Figure()
        history_fig.update_layout(
            title="Consumption History",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No history data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
    
    # Create source history figure
    if len(simulation_history) > 0:
        source_history_fig = px.line(
            simulation_history,
            x='timestamp',
            y='source_level',
            color='source',
            title='Water Source Levels History',
            labels={'source_level': 'Remaining Volume (m³)', 'timestamp': 'Time'},
            template='plotly_white'
        )
    else:
        source_history_fig = go.Figure()
        source_history_fig.update_layout(
            title="Water Source Levels History",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No history data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
    
    # Status message
    status = sim_results['status'].iloc[0]
    consumption = sim_results['consumption'].iloc[0]
    source_level = sim_results['source_level'].iloc[0]
    
    if status == 'normal':
        message = f"✅ Normal operation: {hall} consumed {consumption:.2f} m³ (Source: {water_source}, Remaining: {source_level:.0f} m³)"
        message_style = {'color': 'green'}
    elif status == 'warning':
        message = f"⚠️ Warning: {hall} consumed {consumption:.2f} m³ (Source: {water_source}, Remaining: {source_level:.0f} m³ - Low level!)"
        message_style = {'color': 'orange'}
    else:
        message = f"🚨 Critical: {hall} consumed {consumption:.2f} m³ (Source: {water_source}, Remaining: {source_level:.0f} m³ - Critical level!)"
        message_style = {'color': 'red'}
    
    return (
        consumption_fig, 
        source_fig, 
        history_fig, 
        source_history_fig,
        html.Div(message, style=message_style),
        simulation_history.to_json(orient='split')
    )

# --------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

