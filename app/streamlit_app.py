import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.title("Telco Churn Live Scoring")

st.write("Fill details and score churn risk (calls FastAPI).")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=6)
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=80.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payload = {
    "data": {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless
    }
}

if st.button("Score Customer"):
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        res.raise_for_status()
        out = res.json()

        st.success(f"Churn Probability: {out['churn_probability']:.3f}")
        st.info(f"Risk Segment: {out['risk_segment']}")
        st.warning(f"Revenue At Risk (Annual): ${out['revenue_at_risk']:.2f}")

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.write("Make sure FastAPI is running: uvicorn api.main:app --reload")