from pydantic import BaseModel
from typing import Optional

class CustomerFeatures(BaseModel):
    # Keep flexible: accept any fields present in your feature table
    # We'll validate minimally and pass through to model pipeline.
    data: dict

class PredictionOut(BaseModel):
    churn_probability: float
    risk_segment: str
    revenue_at_risk: float