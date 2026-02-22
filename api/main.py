from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import CustomerFeatures, PredictionOut

MODEL_PATH = Path("models/model.pkl")
FEATURE_PATH = Path("data/processed/features.parquet")

app = FastAPI(title="Telco Churn Live Scoring API", version="1.0")


def risk_segment(p: float) -> str:
    if p >= 0.75:
        return "High Risk"
    elif p >= 0.50:
        return "Medium Risk"
    elif p >= 0.25:
        return "Low Risk"
    else:
        return "Very Low Risk"


@app.on_event("startup")
def load_artifacts():
    global model, template_row
    model = joblib.load(MODEL_PATH)

    # Load a template row from training features
    df = pd.read_parquet(FEATURE_PATH)
    template_row = df.drop(columns=["Churn", "ChurnFlag"], errors="ignore").iloc[[0]].copy()


@app.post("/predict", response_model=PredictionOut)
def predict(payload: CustomerFeatures):
    # Start from template (ensures all required columns exist)
    row = template_row.copy()

    # Override only fields provided by user
    for k, v in payload.data.items():
        if k in row.columns:
            row[k] = v

    proba = float(model.predict_proba(row)[:, 1][0])

    monthly = float(row.get("MonthlyCharges", pd.Series([0])).iloc[0] or 0)
    revenue_at_risk = proba * monthly * 12

    return PredictionOut(
        churn_probability=proba,
        risk_segment=risk_segment(proba),
        revenue_at_risk=revenue_at_risk
    )