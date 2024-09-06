import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Path to the saved models
model_zone1_path = r'C:\Users\BRIGHT WORLD\Documents\AWS\Machine Learning projects\model_zone1.pkl'
model_zone2_path = r'C:\Users\BRIGHT WORLD\Documents\AWS\Machine Learning projects\model_zone2.pkl'
model_zone3_path = r'C:\Users\BRIGHT WORLD\Documents\AWS\Machine Learning projects\model_zone3.pkl'

# Load the pre-trained models
model_zone1 = joblib.load(model_zone1_path)
model_zone2 = joblib.load(model_zone2_path)
model_zone3 = joblib.load(model_zone3_path)

# Define a function for predictions
def predict_power_consumption(temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, month, day, hour, year=None):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'WindSpeed': [wind_speed],
        'GeneralDiffuseFlows': [general_diffuse_flows],
        'DiffuseFlows': [diffuse_flows],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Year': [year],
        'Temp_Humidity_Interaction': [temperature * humidity],
        'Wind_Power_Zone1_Interaction': [wind_speed],
        'Wind_Power_Zone2_Interaction': [wind_speed],
        'Wind_Power_Zone3_Interaction': [wind_speed],
        'GeneralDiffuseFlows_Impact': [general_diffuse_flows],
        'DiffuseFlows_Impact': [diffuse_flows],
        # Placeholder for lagged features or any other features used during training
        'Lag_Temperature': [None],  # Placeholder if needed, replace with actual lag if available
        'Lag_Humidity': [None],     # Placeholder if needed, replace with actual lag if available
        'Lag_WindSpeed': [None],    # Placeholder if needed, replace with actual lag if available
        'Lag_GeneralDiffuseFlows': [None], # Placeholder if needed, replace with actual lag if available
        'Lag_DiffuseFlows': [None], # Placeholder if needed, replace with actual lag if available
    })

    # Reorder columns to match the order of features used during training
    input_data = input_data[model_zone1.feature_names_in_]

    # Predict power consumption for each zone
    zone1_pred = model_zone1.predict(input_data)[0]
    zone2_pred = model_zone2.predict(input_data)[0]
    zone3_pred = model_zone3.predict(input_data)[0]

    return zone1_pred, zone2_pred, zone3_pred
# Streamlit App
st.title('Power Consumption Prediction App')

st.header('Input Weather and Time Data')
temperature = st.number_input('Temperature (°C)', value=20.0)
humidity = st.number_input('Humidity (%)', value=50.0)
wind_speed = st.number_input('Wind Speed (m/s)', value=3.0)
general_diffuse_flows = st.number_input('General Diffuse Flows', value=0.1)
diffuse_flows = st.number_input('Diffuse Flows', value=0.1)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
if st.button('Predict'):
    zone1_pred, zone2_pred, zone3_pred = predict_power_consumption(
        temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, month, day, hour
    )
    
    st.subheader('Predicted Power Consumption')
    st.write(f"Zone 1: {zone1_pred:.2f} kWh")
    st.write(f"Zone 2: {zone2_pred:.2f} kWh")
    st.write(f"Zone 3: {zone3_pred:.2f} kWh")

# Saving the trained models (if this is part of your training script)
# Assuming models are trained as `model_zone1`, `model_zone2`, `model_zone3`
joblib.dump(model_zone1, model_zone1_path)
joblib.dump(model_zone2, model_zone2_path)
joblib.dump(model_zone3, model_zone3_path)
