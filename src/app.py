import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== LOAD MODELS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb_model_path = os.path.join(BASE_DIR, "..", "models", "xgb_model.pkl")
rf_model_path = os.path.join(BASE_DIR, "..", "models", "rf_model.pkl")
columns_path = os.path.join(BASE_DIR, "..", "models", "columns.pkl")

xgb_model = joblib.load(xgb_model_path)
rf_model = joblib.load(rf_model_path)
model_columns = joblib.load(columns_path)

# ===== PAGE CONFIG =====
st.set_page_config(page_title="CLTV Predictor", layout="centered")

st.title("🚀 CLTV Prediction Dashboard")

st.markdown("""
Predict Customer Lifetime Value using Machine Learning  
**Model:** XGBoost + Random Forest Ensemble  
""")

# ===== INPUTS =====
st.sidebar.header("⚙️ Customer Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
area = st.sidebar.selectbox("Area", ["Urban", "Rural"])
qualification = st.sidebar.selectbox("Qualification", ["High School", "Bachelor", "Others"])
income = st.sidebar.selectbox("Income", ["0-5L", "5L-10L", "More than 10L"])
num_policies = st.sidebar.selectbox("Number of Policies", ["1", "More than 1"])
policy = st.sidebar.selectbox("Policy Type", ["A", "B", "C"])
type_of_policy = st.sidebar.selectbox("Policy Level", ["Gold", "Silver", "Platinum"])
marital_status = st.sidebar.selectbox("Marital Status", [0, 1])
vintage = st.sidebar.slider("Customer Vintage", 0, 10)
claim_amount = st.sidebar.slider("Claim Amount", 0, 500000, 50000)

# ===== PREDICT BUTTON =====
if st.button("Predict CLTV"):

    # ===== CREATE INPUT DATA =====
    input_data = pd.DataFrame({
        "gender": [gender],
        "area": [area],
        "qualification": [qualification],
        "income": [income],
        "num_policies": [num_policies],
        "policy": [policy],
        "type_of_policy": [type_of_policy],
        "marital_status": [marital_status],
        "vintage": [vintage],
        "claim_amount": [claim_amount]
    })

    # ===== PREPROCESSING =====
    input_data['num_policies'] = input_data['num_policies'].replace({'More than 1': 2}).astype(int)

    income_map = {
        '0-5L': 1,
        '5L-10L': 2,
        'More than 10L': 3
    }
    input_data['income'] = input_data['income'].map(income_map)

    # ===== FEATURE ENGINEERING =====
    input_data['income_per_policy'] = input_data['income'] / (input_data['num_policies'] + 1)
    input_data['claim_per_policy'] = input_data['claim_amount'] / (input_data['num_policies'] + 1)
    input_data['policy_per_year'] = input_data['num_policies'] / (input_data['vintage'] + 1)
    input_data['income_x_claim'] = input_data['income'] * input_data['claim_amount']
    input_data['log_claim'] = np.log1p(input_data['claim_amount'])
    input_data['claim_to_income'] = input_data['claim_amount'] / (input_data['income'] + 1)
    input_data['claim_per_year'] = input_data['claim_amount'] / (input_data['vintage'] + 1)

    # ===== ENCODING =====
    input_data = pd.get_dummies(input_data, drop_first=True)

    # ===== ALIGN COLUMNS =====
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # ===== PREDICTION =====
    pred1 = xgb_model.predict(input_data)
    pred2 = rf_model.predict(input_data)

    final_pred = (0.7 * pred1) + (0.3 * pred2)

    # ===== CLIPPING =====
    final_pred = np.clip(final_pred, 25000, 650000)

    # ===== OUTPUT =====
    st.markdown("## 📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Predicted CLTV", f"₹ {int(final_pred[0])}")

    with col2:
        if final_pred[0] > 500000:
            st.success("🌟 High Value Customer")
        elif final_pred[0] > 200000:
            st.warning("⚡ Medium Value Customer")
        else:
            st.error("🔻 Low Value Customer")

    # ===== CUSTOMER SUMMARY =====
    st.markdown("### 🧾 Customer Summary")
    st.dataframe(input_data)

    # ===== VISUALIZATION =====
    st.markdown("## 📈 CLTV Visualization")

    fig, ax = plt.subplots()
    ax.barh(["CLTV"], [final_pred[0]])
    ax.set_title("Customer Lifetime Value")
    st.pyplot(fig)

    # ===== COMPARISON CHART =====
    st.markdown("## 📊 Value Comparison")

    labels = ["Low", "Medium", "High"]
    values = [200000, 400000, 600000]

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, values)
    ax2.axhline(y=final_pred[0])
    st.pyplot(fig2)

    # ===== FEATURE IMPORTANCE =====
    st.markdown("## 🔍 Feature Importance")

    importance = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature").head(10))