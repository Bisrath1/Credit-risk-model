from pydantic import BaseModel, Field
from typing import Optional

class CustomerData(BaseModel):
    CustomerId: Optional[str] = Field(default=None, description="Customer unique identifier")
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Gender_Female: int
    Geography_Germany: int
    Geography_Spain: int

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
