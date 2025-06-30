from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow Model Registry
model_name = "CreditRiskModel"
model_version = "latest"  # Or specify a version
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Predict risk probability
    risk_prob = model.predict_proba(input_data)[:, 1][0]
    
    return PredictionResponse(
        customer_id=data.CustomerId if hasattr(data, "CustomerId") else "unknown",
        risk_probability=float(risk_prob)
    )