from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/telco_customer_churn.csv")
OUT_PATH = Path("data/processed/clean.parquet")


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges sometimes contains blanks -> convert safely to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    return df


def standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    # Churn: Yes/No -> 1/0 (keep original too if you want)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def main():
    print("Running clean pipeline...")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw file not found at {RAW_PATH}. Run: python src/ingest.py first."
        )

    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df.columns = [c.strip() for c in df.columns]  # clean column names
    df = clean_total_charges(df)

    # Remove rows where TotalCharges became null due to blanks (usually tenure=0)
    # We’ll keep them out of modeling, but later we can decide business handling.
    df = df.dropna(subset=["TotalCharges"]).copy()

    df = standardize_target(df)

    # Ensure output directory exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved clean data -> {OUT_PATH}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    # Quick sanity prints
    print("Churn distribution:")
    print(df["Churn"].value_counts(dropna=False))
    print("\nNulls summary (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()