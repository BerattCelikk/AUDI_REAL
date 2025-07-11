# Audi Hungaria Water Management Project – Final Study

## Task 1: Water Metering System Concept (Foundation for Digitalization)

### Measurement System Concept

#### Placement of Measurement Points
To enable full digital tracking of water usage, meters should be placed at:
- Main intake points (e.g., wells, treatment outputs)
- Each production hall (e.g., G19, G95)
- Selected production lines within each hall (especially for the two selected products)
- Key technology zones (e.g., sprinklers, high-pressure washers)

#### Measurement Principles and Instruments
- Non-intrusive ultrasonic or electromagnetic meters to ensure minimal disruption
- Smart meters supporting real-time telemetry and digital integration
- Example instruments: 
  - Siemens SITRANS
  - Krohne OPTIFLUX

#### Data and Collection Points
To support digital modeling and visualization:
- Monthly water consumption per measurement point
- Production hall mapping
- Water types (industrial, drinking)
- Production volume per line (for specific consumption)
- Environmental variables (e.g., temperature, shift data)

All collected data is stored in structured tables and time-indexed for visualization.

### Calculation Method

#### Specific Water Consumption per Product
Using the `df_detailed` model, water consumption can be allocated per production line and per product batch.

#### Identification of Background Variables
Variables such as:
- Seasonal variation (Winter, Spring, etc.)
- Production volume and schedule
- Water type and use case (cleaning, cooling, etc.)

are incorporated into a causal logic model.

#### Modeling Logic
- Monthly time-series breakdown with seasonal tags
- Product-line correlation to consumption (`WaterConsumptionModel` class)
- Lagged variables for forecasting (XGBoost-based)
- Abnormality detection via z-score for anomaly monitoring

#### Direct and Indirect Consumption
- **Direct**: Measured usage directly tied to production lines
- **Indirect**: Auxiliary consumption (cleaning, HVAC, etc.) inferred via residual modeling

---

## Task 2: Digitalization of Water Consumption

### Data Model Construction from Source Files
Raw Excel datasets (`TOP_consumers_engine_2024.xlsx`, `Water_consumption_AH_2024.xlsx`) were parsed and cleaned. Due to unstructured headers and multiple header rows, preprocessing was applied to standardize data for modeling.

The cleaned data was transformed into a monthly time-series format using:

```python
records = []
for i in range(4, df_water.shape[0]):
    row = df_water.iloc[i]
    for idx, date in enumerate(dates):
        value = row[start_col + idx]
        if pd.notna(value):
            records.append({
                "Production_Hall": row[1],
                "Short_Name": row[2],
                "Unit": row[3],
                "Year": date.year,
                "Month": date.month,
                "Consumption_m3": float(value)
            })
