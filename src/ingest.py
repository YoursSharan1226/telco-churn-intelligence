from pathlib import Path
import requests

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "telco_customer_churn.csv"

    print("Starting download...")
    print("URL:", DATA_URL)
    print("Saving to:", out_path.resolve())

    resp = requests.get(DATA_URL, timeout=60)
    print("HTTP status:", resp.status_code)

    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    print("Saved bytes:", out_path.stat().st_size)
    print("Done.")

if __name__ == "__main__":
    main()