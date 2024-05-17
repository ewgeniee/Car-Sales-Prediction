import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to load label encoders
def load_label_encoders():
    
    label_encoders = {}
    
    for col in ["engine_type", "make_name", "model_name", "body_type", "fuel_type", "transmission", "wheel_system_display"]:
        le = LabelEncoder()
        le.classes_ = np.load(f"label_encoder_{col}.npy", allow_pickle = True)
        label_encoders[col] = le
    return label_encoders

label_encoders = load_label_encoders()
min_max_scaler = joblib.load("min_max_scaler.pkl")
best_model = joblib.load("best_model_xgboost.pkl")

# List of feature names in the correct order as used during training
feature_names = ["engine_type", "mileage", "year", "horsepower", "model_name", "make_name",
                 "body_type", "fuel_type", "transmission", "wheel_system_display", "average_fuel_economy"]

def encode_features(data):
    
    for col, le in label_encoders.items():
        data[col] = le.transform(data[col])
        
    return data

def preprocess_input(make_name, model_name, year, mileage, transmission, body_type, fuel_type, horsepower, city_fuel_economy, highway_fuel_economy, engine_type, wheel_system_display):
    
    average_fuel_economy = (city_fuel_economy + highway_fuel_economy) / 2
    
    data = pd.DataFrame({
        "engine_type": [engine_type],
        "mileage": [mileage],
        "year": [year],
        "horsepower": [horsepower],
        "model_name": [model_name],
        "make_name": [make_name],
        "body_type": [body_type],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "wheel_system_display": [wheel_system_display],
        "average_fuel_economy": [average_fuel_economy]
    })
    
    data = encode_features(data)
    
    data = data[feature_names]
    
    # Transform data using MinMaxScaler and preserve feature names
    data_scaled = min_max_scaler.transform(data)
    
    data_scaled = pd.DataFrame(data_scaled, columns = feature_names)
    
    return data_scaled

def main():
    
    st.title("Vehicle Price Predictor")
    
    st.header("Input Vehicle Details")
    
    make_name = st.selectbox("Make Name", label_encoders["make_name"].classes_)
    model_name = st.selectbox("Model Name", label_encoders["model_name"].classes_)
    mileage = st.number_input("Mileage", 1000, 1000000, 0)
    year = st.slider("Year", 1, 2024, 1950)
    fuel_type = st.selectbox("Fuel Type", label_encoders["fuel_type"].classes_)
    transmission = st.selectbox("Transmission", label_encoders["transmission"].classes_)
    body_type = st.selectbox("Body Type", label_encoders["body_type"].classes_)
    engine_type = st.selectbox("Engine Type", label_encoders["engine_type"].classes_)
    horsepower = st.number_input("Horsepower", step=1)
    city_fuel_economy = st.number_input("City Fuel Economy")
    highway_fuel_economy = st.number_input("Highway Fuel Economy")
    wheel_system_display = st.selectbox("Wheel System Display", label_encoders["wheel_system_display"].classes_)
    
    if st.button("Predict Price"):
        input_data = preprocess_input(engine_type, make_name, model_name, body_type, fuel_type, transmission,
                                      wheel_system_display, mileage, year, horsepower, city_fuel_economy, highway_fuel_economy)
        
        price_log = best_model.predict(input_data)
        price = np.expm1(price_log)[0]
        st.write(f"Predicted Price: ${price:.2f}")
        
    st.header("Calculate Future Value and Usage Costs")
    
    usage_years = st.number_input("Usage Period (Years)", step=1)
    annual_mileage = st.number_input("Annual Mileage")
    
    fuel_type_usage = st.selectbox("Fuel Type for Usage Costs", ["Gasoline", "Diesel", "Electro"])
    
    fuel_cost = 0
    if fuel_type_usage == "Gasoline":
        fuel_cost = 0.95 / 100
    elif fuel_type_usage == "Diesel":
        fuel_cost = 1.03 / 100
    elif fuel_type_usage == "Electro":
        fuel_cost = 0.1545 / 100
        
    if st.button("Calculate Future Value and Costs"):
        input_data = preprocess_input(engine_type, make_name, model_name, body_type, fuel_type, transmission,
                                      wheel_system_display, mileage, year, horsepower, city_fuel_economy, highway_fuel_economy)
        
        price_log = best_model.predict(input_data)
        price = np.expm1(price_log)[0]
        
        total_mileage = mileage + (annual_mileage * usage_years)
        average_fuel_economy = (city_fuel_economy + highway_fuel_economy) / 2
        usage_costs = annual_mileage * average_fuel_economy * fuel_cost * usage_years
        
        depreciation_rate = 0.15
        future_price = price * (1 - depreciation_rate) ** usage_years
        
        st.write(f"Future Value: ${future_price:.2f}")
        st.write(f"Total Usage Costs: ${usage_costs:.2f}")
        
if __name__ == "__main__":
    main()
