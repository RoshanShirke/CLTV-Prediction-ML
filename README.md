# 🚀 Customer Lifetime Value Prediction (ML + Streamlit)

## 📌 Overview
This project predicts Customer Lifetime Value (CLTV) using machine learning on structured customer and policy data.  
The goal is to help businesses identify high-value customers and optimize marketing, retention, and resource allocation strategies.

---

## 🌐 Live Demo
👉 https://cltv-prediction-ml.streamlit.app

---

## 🎯 Problem Statement
Customer Lifetime Value (CLTV) is a critical metric for business growth.  
This project aims to predict CLTV using customer demographics, policy information, and claim history.

---

## 🧠 Approach

### 🔹 Data Processing
- Handled categorical + numerical features
- Mapped ordinal variables (income, policies)
- No missing values in dataset

### 🔹 Feature Engineering (Key Highlight 🔥)
- Ratio Features:
  - income_per_policy
  - claim_per_policy
  - claim_to_income
  - claim_per_year
- Interaction Features:
  - income × claim
- Behavioral Features:
  - policy_per_year
- Log Transformation:
  - log_claim

👉 Feature engineering significantly improved model performance.

---

### 🔹 Encoding
- Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`)

---

### 🔹 Models Used

#### 1. XGBoost Regressor
- Tuned hyperparameters for stability and performance

#### 2. Random Forest Regressor
- Used as a secondary model for ensemble

---

### 🔥 Ensemble Strategy
Final prediction: (0.7 * XGBoost) + (0.3 * RandomForest)


---

### 🔻 Prediction Optimization
- Applied clipping to control extreme values:np.clip(predictions, 25000, 650000)

---

## 📊 Results

- **R² Score:** ~0.157
- **Leaderboard Rank:** Top 100 (#64)
- Stable performance across validation and leaderboard

---

## 🖥️ Streamlit App Features

- 📊 Real-time CLTV prediction
- 🎛️ Interactive sidebar inputs
- 📈 Visual charts (CLTV + comparison)
- 🔍 Feature importance (model explainability)
- 🧾 Customer input summary

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Joblib

---

## 📁 Project Structure
CLTV-Prediction-ML/
│
├── models/ # Saved ML models
├── src/
│ ├── main.py # Model training
│ └── app.py # Streamlit app
├── submission/
│ └── submission.csv
├── requirements.txt
├── README.md
├── .gitignore


---

## 🏆 Achievements

- 🥇 Ranked **Top 100 (#64)** in Analytics Vidhya Hackathon
- Built end-to-end ML pipeline from scratch
- Successfully deployed ML model as a web application

---

## 🚀 Highlights

- End-to-end ML pipeline (data → model → deployment)
- Strong feature engineering impact
- Ensemble learning for improved performance
- Real-time prediction dashboard
- Production-ready project structure

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py

💼 Author

Roshan Shirke
📍 Pune, India
🔗 GitHub: https://github.com/RoshanShirke
🔗 LinkedIn: https://www.linkedin.com/in/roshan-shirke-527089306