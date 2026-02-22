# **Telco Churn Intelligence**
End-to-End Customer Churn Prediction System with ML, API Deployment, and Interactive UI
## **Project Overview**
Telco Churn Intelligence is a production-style machine learning project that predicts customer churn and enables business-driven retention targeting.
This system goes beyond offline model training by implementing:
* A trained churn prediction model
* Business-aligned targeting metrics (Top-K strategy)
* A FastAPI backend for real-time scoring
* A Streamlit UI for interactive predictions
* Modular, deployment-ready architecture
The objective is not only to predict churn accurately, but to translate predictions into actionable marketing decisions.
## **Business Objective**
Instead of targeting all customers for retention campaigns, this system enables:
* Ranking customers by churn probability
* Targeting the top 30% highest-risk customers
* Capturing a majority of likely churners
* Improving marketing efficiency and ROI
This aligns machine learning outputs with measurable business impact.
## **System Architecture**
Streamlit UI → FastAPI Backend → ML Model → Prediction Response
* Streamlit provides an interactive interface for business users
* FastAPI exposes REST endpoints for real-time inference
* The trained model returns churn probability and classification
* The system supports integration with BI tools or automation systems
## **Application Demo**
### **Streamlit Interface**
The Streamlit UI allows users to:
* Input customer attributes
* Generate churn probability instantly
* Simulate retention decision-making
### **FastAPI Backend**
Swagger Documentation:
Sample API Response:
The FastAPI backend provides:
* Live model inference
* Structured JSON prediction output
* API-first architecture suitable for integration
### **Model Performance**
* Model trained using supervised learning (e.g., Logistic
Regression / Random Forest)
* Evaluated using ROC-AUC and classification metrics
* Business metric optimization using Top-K targeting
* Capture rate and response rate evaluated for marketing relevance
* This ensures the model is optimized for business deployment
### Tech Stack
* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Streamlit
* Git
How to Run Locally
#### **1. Clone the Repository**
git clone <https://github.com/YoursSharan1226/telco-churn-intelligence.git>
cd telco-churn-intelligence
#### **2. Install Dependencies**
pip install -r requirements.txt
#### **3. Start the FastAPI Server**
uvicorn api.main:app --reload
**Open Swagger documentation:**
http://127.0.0.1:8000/docs
#### **4. Start the Streamlit Application**
**Open a second terminal and run:**
streamlit run app/streamlit_app.py
**Open the UI:**
http://localhost:8501
### **Project Structure**
telco-churn-intelligence/
│
├── api/                  # FastAPI backend
├── app/                  # Streamlit UI
├── models/               # Saved ML model artifacts
├── data/                 # Dataset files
├── docs/
│   └── images/           # README screenshots
├── train.py              # Model training script
├── score.py              # Scoring logic
├── requirements.txt
└── README.md
### **Deployment Note**
The screenshots included in this README were captured from the locally running application.
![alt text](StreamLit_Prediction.jpeg)
![alt text](API_Swagger.jpeg)
![alt text](<API Response.jpeg>)
To make the system publicly accessible at all times, the API and Streamlit application can be deployed to a cloud platform such as Streamlit Community Cloud, Render, or AWS.
**Key Highlights**
* End-to-end ML pipeline
* Business-aligned evaluation metrics
* API-based inference architecture
* Interactive front-end interface
* Modular and production-oriented structure
This project demonstrates the ability to bridge machine learning modeling with real-world deployment and business decision systems.
