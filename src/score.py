from pathlib import Path
import pandas as pd
import joblib

MODEL_PATH = Path("models/model.pkl")
FEATURE_PATH = Path("data/processed/features.parquet")
OUT_PATH = Path("data/mart/customer_scoring.csv")


def risk_segment(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.50:
        return "Medium Risk"
    elif prob >= 0.25:
        return "Low Risk"
    else:
        return "Very Low Risk"


def main():
    print("Running scoring pipeline...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError("Features not found. Run: python src/features.py")

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURE_PATH)

    id_col = "customerID"

    X = df.drop(columns=["Churn", "ChurnFlag"], errors="ignore")

    print("Scoring customers...")
    proba = model.predict_proba(X)[:, 1]

    df["ChurnProbability"] = proba
    df["RiskSegment"] = df["ChurnProbability"].apply(risk_segment)
    df["RevenueAtRisk"] = df["ChurnProbability"] * df["MonthlyCharges"] * 12

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        id_col,
        "ChurnProbability",
        "RiskSegment",
        "MonthlyCharges",
        "AnnualizedRevenue",
        "RevenueAtRisk"
    ]

    export_cols = [c for c in export_cols if c in df.columns]

    df[export_cols].to_csv(OUT_PATH, index=False)

    print("Saved scoring table ->", OUT_PATH.resolve())
    print("Done.")


if __name__ == "__main__":
    main()