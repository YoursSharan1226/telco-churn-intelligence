from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/clean.parquet")
OUT_FEATURES = Path("data/processed/features.parquet")
OUT_CUSTOMER_BASE = Path("data/mart/customer_base.csv")

def main():
    print("Running features pipeline...")

    print("Looking for:", IN_PATH.resolve())
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run: python src/clean.py")

    df = pd.read_parquet(IN_PATH)
    print("Loaded clean data:", df.shape)

    # Tenure buckets
    bins = [-1, 0, 12, 24, 36, 48, 60, 120]
    labels = ["0", "1-12", "13-24", "25-36", "37-48", "49-60", "60+"]
    df["TenureBucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Flags
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["IsPaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    df["HasFiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Add-on services count
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for c in addon_cols:
        df[c] = df[c].fillna("No")
        df[f"{c}_Yes"] = (df[c] == "Yes").astype(int)

    df["AddonCount"] = df[[f"{c}_Yes" for c in addon_cols]].sum(axis=1)

    # Revenue features
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TenureSafe"] = df["tenure"].replace(0, np.nan)
    df["AvgMonthlyCharge_fromTotal"] = (df["TotalCharges"] / df["TenureSafe"]).fillna(df["MonthlyCharges"])
    df["AnnualizedRevenue"] = df["MonthlyCharges"] * 12

    # Save outputs
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FEATURES, index=False)
    print("Saved features ->", OUT_FEATURES.resolve(), "| shape=", df.shape)

    OUT_CUSTOMER_BASE.parent.mkdir(parents=True, exist_ok=True)
    bi_cols = [
        "customerID", "Churn", "ChurnFlag", "tenure", "TenureBucket",
        "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
        "MonthlyCharges", "TotalCharges", "AnnualizedRevenue",
        "AddonCount", "IsMonthToMonth", "IsElectronicCheck", "HasFiber"
    ]
    bi_cols = [c for c in bi_cols if c in df.columns]
    df[bi_cols].to_csv(OUT_CUSTOMER_BASE, index=False)
    print("Saved BI base ->", OUT_CUSTOMER_BASE.resolve(), "| cols=", len(bi_cols))

    print("Done.")

if __name__ == "__main__":
    main()