from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    ProductCategory_airtime: int
    ProductCategory_financial_services: int
    ProductCategory_movies: int
    # Add other one-hot encoded or numerical features as per your processed dataset

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float