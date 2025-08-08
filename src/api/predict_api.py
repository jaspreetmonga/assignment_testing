#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from logger import get_logger
from api.schemas import HousingInput
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)

app = FastAPI(title="California Housing Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    raise e

# List the exact categories your model expects for ocean_proximity, same as training
OCEAN_CATEGORIES = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN"
]

# The full list of features expected by scaler (numeric + one-hot encoded)
FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

N_FEATURES = len(FEATURE_COLUMNS)

@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is up!"}

@app.post("/predict/")
def predict(input_data: HousingInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])

        # One-hot encode ocean_proximity with fixed categories and prefix
        ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')

        # Add missing ocean categories with zeros if not present in input
        for cat in OCEAN_CATEGORIES:
            col_name = f'ocean_proximity_{cat}'
            if col_name not in ocean_dummies.columns:
                ocean_dummies[col_name] = 0

        # Drop original ocean_proximity column and concat one-hot columns
        df = df.drop(columns=['ocean_proximity'])
        df = pd.concat([df, ocean_dummies], axis=1)

        # Reorder columns exactly as expected by scaler/model
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(input_scaled)

        logger.info(f"Prediction made for input: {input_data.dict()}")
        return {"predictions": prediction.tolist()}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Register Prometheus instrumentation here (before startup)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
