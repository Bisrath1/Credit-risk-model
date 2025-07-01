from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import sys
import os

# ✅ Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ✅ Now import using relative path to where the script is located
import sys
import os

# ✅ Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

mlflow.set_tracking_uri("file:///C:/10x AIMastery/Credit-risk-model/mlruns")


app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow Model Registry
model_name = "CreditRiskModel"
model_version = "latest"  # Can be specific like "1"
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])

    # Predict risk probability
    risk_prob = model.predict_proba(input_data)[:, 1][0]

    return PredictionResponse(
        customer_id=data.CustomerId if hasattr(data, "CustomerId") else "unknown",
        risk_probability=float(risk_prob)
     )
