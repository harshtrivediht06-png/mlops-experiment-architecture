from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Prediction API", version="1.0.0")

# Load the saved Scikit-learn Pipeline
model = joblib.load("models/model.joblib")

# Load feature names dynamically to match the training DataFrame structure
feature_names = load_breast_cancer().feature_names

# Define the expected request schema
class PredictionInput(BaseModel):
    features: list[float]  # Expects a list of 30 float values

@app.get("/health")
def health():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionInput):
    """Prediction endpoint that accepts feature data and returns a class."""
    # 1. Validate input length
    if len(payload.features) != 30:
        raise HTTPException(
            status_code=422, 
            detail=f"Expected exactly 30 features, but received {len(payload.features)}."
        )
    
    # 2. Construct DataFrame with exact feature names preserved
    df = pd.DataFrame([payload.features], columns=feature_names)
    
    # 3. Generate prediction
    result = model.predict(df)
    
    # 4. Return formatted response
    return {"prediction": int(result[0])}