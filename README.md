# **Telco Churn Intelligence**
End-to-End Customer Churn Prediction System with ML, API Deployment, Interactive UI, and Executive-Level Tableau Dashboard
## **Project Overview**
Telco Churn Intelligence is a production-style machine learning project designed to predict customer churn and enable business-driven retention targeting.
This system extends beyond offline model training by implementing:
* A trained churn prediction model
* Business-aligned targeting metrics (Top-K strategy)
* A FastAPI backend for real-time scoring
* A Streamlit UI for interactive predictions
* An executive-level Tableau dashboard for risk and revenue analysis
* A modular, deployment-ready architecture
The objective is not only to predict churn accurately, but to convert predictions into measurable and actionable business decisions.
## **Business Objective**
Rather than targeting all customers for retention campaigns, this system enables:
* Ranking customers by churn probability
* Segmenting customers into risk tiers (High, Medium, Low, Very Low)
* Targeting the top 30% highest-risk customers
* Capturing a majority of likely churners
* Improving marketing efficiency and return on investment
This approach aligns machine learning outputs directly with strategic business impact.
## **System Architecture**
Streamlit UI → FastAPI Backend → ML Model → Prediction Response → Tableau Executive Dashboard
* Streamlit provides an interactive interface for business users
* FastAPI exposes REST endpoints for real-time inference
* The trained model returns churn probability and classification
* The Tableau dashboard transforms prediction outputs into executive insights
* The system supports integration with BI workflows, automation systems, and reporting environments
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
### **Tableau Executive Dashboard**
An executive-level Tableau dashboard was developed using the scored customer dataset to translate model outputs into business intelligence insights.
The dashboard includes:
* Customer risk segmentation distribution
* Average churn probability by risk segment (lollipop visualization)
* Revenue at Risk analysis by segment
* KPI summary including total customers, high-risk population, and financial exposure
This enables leadership teams to:
* Quantify financial exposure from churn
* Prioritize high-risk customer segments
* Align retention strategy with revenue impact
The Tableau layer bridges predictive modeling with executive decision-making.
## **Model Performance**
* The model achieved a **ROC–AUC of 0.8343** and an **accuracy of 72.64%**, demonstrating strong ability to distinguish churners from non-churners across probability thresholds.
* From a business standpoint, the key metric is **Recall for churners (78.34%)**. This means the model successfully identifies nearly **8 out of 10 customers who are likely to churn**, significantly reducing the risk of losing high-value customers unnoticed. While precision for churners is **49.08%**, this trade-off is intentional — in retention campaigns, it is typically more acceptable to contact a few extra safe customers than to miss actual churners.
* The confusion matrix (TP: 293, FN: 81) confirms that missed churners are relatively limited compared to correctly identified ones. This performance supports **Top-K targeting strategies**, enabling the business to rank customers by churn probability and focus retention budgets on the highest-risk segment.
* Overall, the model is optimized not just for statistical accuracy, but for **practical marketing impact, risk prioritization, and deployment-ready decision support.**
## **Tech Stack**
* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Streamlit
* Tableau
* Git
## **How to Run Locally**
### **1. Clone the Repository**
git clone https://github.com/YoursSharan1226/telco-churn-intelligence.git
cd telco-churn-intelligence
### **2. Install Dependencies**
pip install -r requirements.txt
### **3. Start the FastAPI Server**
uvicorn api.main:app --reload
Open Swagger documentation:
http://127.0.0.1:8000/docs
### **4. Start the Streamlit Application**
Open a second terminal and run:
streamlit run app/streamlit_app.py
Open the UI:
http://localhost:8501
### **Deployment Note**
* The screenshots included in this README were captured from the locally running application.
* To make the system publicly accessible at all times, the API and Streamlit application can be deployed to a cloud platform such as Streamlit Community Cloud, Render, or AWS.
* The Tableau dashboard can be shared via Tableau Public or distributed as a packaged workbook for executive reporting.
## **Key Highlights** 
* End-to-end ML pipeline
* Business-aligned evaluation metrics
* API-based inference architecture
* Interactive front-end interface
* Executive-level Tableau dashboard integration
* Modular and production-oriented structure
This project demonstrates the ability to connect machine learning modeling, API deployment, and executive analytics into a unified, business-ready decision support system.
