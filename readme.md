# Vehicle Price Predictor

## Overview
This Streamlit application provides an interface for predicting the current and future prices of vehicles using a trained model. The application allows users to input vehicle details and get a predicted price based on the model. Additionally, it calculates future value and usage costs, considering depreciation and fuel costs.

## Features
- Predict current price of a vehicle based on its details.
- Calculate the future value of a vehicle considering its usage period and mileage.
- Estimate total usage costs based on fuel type and consumption.

## Installation
To run this application, you will need to install several Python packages including Streamlit, pandas, and scikit-learn.

```
pip install streamlit pandas numpy scikit-learn joblib
```
Once the application is running, input the required vehicle details into the app interface and click the "Predict Price" button to view the estimated market price. For future value and usage costs, provide the expected usage period and yearly mileage, then click the "Calculate Future Value and Costs" button.

## Model Details
The predictive model is built using XGBoost, optimized for robust performance on a wide range of vehicle data. The model uses features such as engine type, mileage, year, and more to predict prices.