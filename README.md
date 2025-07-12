# Audi Hungaria Water Management Project – TASK 2

## Subtask 1: Creating a Data Model for the Water Cycle

### What the task asks for:
Structure the raw Excel water data into a usable model that allows consistent tracking, aggregation, and analysis.
### Code Implementation:
The raw Excel files (TOP_consumers_engine_2024.xlsx, Water_consumption_AH_2024.xlsx) contain unstructured headers and fragmented data.

The script:

Cleans and extracts the water consumption data from each metering point.

Constructs a structured DataFrame called df_detailed with these key fields:

```python
Production_Hall, Short_Name, Unit, Year, Month, Consumption_m3
```
Adds a Season field based on the month for further pattern analysis.
### Outcome:
A well-normalized time-series database of monthly water consumption per unit, which forms the **core data model** of the water cycle.

This model is suitable for all kinds of advanced analytics (seasonality, forecasting, anomaly detection).

## Subtask 2: Developing the Digital Model (Task 1 Logic)

### What the task asks for:
Implement Task 1’s analytical methods (e.g., specific water consumption, seasonal trends, anomaly detection) in a reusable digital model.

### Code Implementation:
The WaterConsumptionModel class encapsulates the logic from Task 1:

calculate_yearly_consumption()

top_consumers()

consumption_by_season()

detect_anomalies() (z-score based)

Forecasting logic is built using **XGBoost**:

Data is preprocessed with lag features and categorical encoding.

Model is validated with **TimeSeriesSplit**, ensuring temporal integrity.

Predictive modeling for **2025 consumption** by month and by production hall.

Average MAE is computed (~158,875 m³), quantifying model accuracy.

### Outcome:

This digital model transforms the static data into a **simulation-ready analytical engine**, as required by Task 2.

It incorporates the **calculation methodology** described in Task 1, including causal factors and time-based logic.


## Subtask 3: Creating Visualization Interpretable at Management Level

### What the task asks for:
Management-level, actionable insights from the water data—clear, high-level visuals.

### Code Implementation:
Bar plots, line graphs, and heatmaps created using **Seaborn** and **Matplotlib**:

**Annual consumption trends**

**Top consumers**

**Monthly usage patterns**

**Seasonal impacts**

Forecast visualizations:

**Predicted 2025 monthly water consumption**

**Top predicted consumers in 2025**

These provide **executive-level dashboards** for strategic decisions (e.g., peak load planning, identifying critical water-consuming assets).

### Outcome:
These visuals help communicate complex water data to non-technical stakeholders and decision-makers.

## Subtask 4: Digital Twin Implementation for Water System Simulation

### Core Components

1. **Data Loading and Preprocessing**
```python
def load_and_preprocess_data():
    # Loads raw Excel files and transforms into structured format
    # Key transformations:
    # - Extracts measurement points metadata
    # - Converts wide format to long format
    # - Adds temporal features (Month_Num, Date)
    # Returns three DataFrames: df_top, df_water, df_consumption
```

2. **Water Consumption Model**
```python
class WaterConsumptionModel:
    # Analytical engine providing:
    # - Yearly/monthly consumption calculations
    # - Top consumers identification
    # - Seasonal analysis
    # - Anomaly detection using Z-scores
    def detect_anomalies(self, threshold=2):
        # Uses statistical methods to flag unusual consumption patterns
        # Returns anomalies with Z-scores
```

3. **Digital Twin Core**
```python
class WaterSystemDigitalTwin:
    # Simulation engine featuring:
    # - System state tracking (water sources, production halls)
    # - Daily simulation capability
    # - Optimization algorithms
    # - Historical data storage
    
    def simulate_day(self, production_plan, weather=None):
        # Simulates water consumption based on:
        # - Production levels
        # - Weather conditions
        # - Water source availability
        # Updates system state and returns results
```
<img width="1454" height="1458" alt="audi1" src="https://github.com/user-attachments/assets/75da3a28-2da7-456a-9088-9c8e8e05ef19" />

## Key Features

1. **Interactive Simulation Controls**
```python
# Production parameters
dcc.Dropdown(id='dt-hall-dropdown', ...)  # Select production hall
dcc.Slider(id='dt-production-level', ...)  # Set production intensity

# Water source selection
dcc.Dropdown(id='dt-water-source', ...)  # Choose water source

# Environmental factors
dcc.Slider(id='dt-weather-temp', ...)  # Set temperature impact
```
2. **Real-time Visualization**
```python
# Main simulation outputs
dcc.Graph(id='dt-simulation-results')  # Shows current consumption
dcc.Graph(id='dt-source-levels')       # Displays water source status

# Historical trends
dcc.Graph(id='dt-consumption-history') # Tracks consumption over time
dcc.Graph(id='dt-source-history')      # Shows source level changes
```
3. **Anomaly Detection System**
```python
# Configurable detection threshold
dcc.Slider(
    id='anomaly-threshold',
    min=1, max=5, step=0.1,
    value=2.5,
    marks={i: str(i) for i in range(1, 6)}
)

# Visual output
px.scatter(
    color_continuous_scale=['green', 'yellow', 'red'],
    range_color=[0, 5],
    title='Anomaly Detection'
)
```

## Simulation Logic
1. **Consumption Calculation**
```python
# Base consumption adjusted by:
daily_consumption = (base_consumption 
                    * production_level 
                    * weather_factor)
# Where weather_factor = 1.2 if temperature > 25°C
```
2. **Water Source Management**
```python
# Source level updates
new_level = current_level - daily_consumption

# Status determination:
if new_level < 0.1 * capacity: status = 'critical'
elif new_level < 0.2 * capacity: status = 'warning'
else: status = 'normal'
```
3. **Optimization Engine**
```python
def optimize_water_distribution(self):
    # Automatically switches critical halls to alternative sources
    # Prioritizes sources with sufficient capacity (>10,000 m³)
```
<img width="4314" height="1971" alt="audi2" src="https://github.com/user-attachments/assets/d470ab7b-13ec-42f1-856e-d71e79db84f6" />

## Advanced Features
1. **Temporal Analysis**
```python
# Cyclical feature encoding for seasonality
df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)
```
2. **Predictive Capabilities**
```python
# XGBoost forecasting model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)  # Uses lag features + temporal encoding
```
3. **Scenario Testing**
```python
# Example production plan
production_plan = {
    'Hall_A': {
        'production_level': 1.5,  # 150% capacity
        'water_source': 'Brunn_2'
    }
}
# With weather impact
weather = {'temperature': 28}  # Hot summer day
```
## Management Visualizations
1. **Status Overview**
```python
px.bar(
    color='status',
    color_discrete_map={
        'normal': 'green',
        'warning': 'orange',
        'critical': 'red'
    }
)
```
2. **Source Monitoring**
```python
px.bar(
    text='current_level',
    title='Water Source Levels',
    labels={'current_level': 'Remaining Volume (m³)'}
)
```
3. **Historical Trends**
```python
px.line(
    title='Consumption History',
    labels={'consumption': 'Consumption (m³)'}
)
```
