# ---------------------------------------------------------------
# File: test_predict_api.py
# Description:
# Script to test the California Housing Prediction REST API.
# Sends a sample POST request with input features and prints the response.
# Part of: MLOps Pipeline - Model Inference Testing
# ---------------------------------------------------------------

import requests

url = "http://localhost:8000/predict/"  # Make sure your API is running on this port

input_data = {
    "longitude": -118.0,
    "latitude": 34.0,
    "housing_median_age": 41.0,
    "total_rooms": 6000,
    "total_bedrooms": 1200,
    "population": 1000,
    "households": 500,
    "median_income": 5.5,
    "ocean_proximity": "INLAND"
}

try:
    response = requests.post(url, json=input_data)
    response.raise_for_status()  # Raise an error for bad status codes
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print("API request failed:", e)
