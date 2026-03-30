import streamlit as st
import numpy as np

st.set_page_config(page_title="CLTV Predictor", layout="centered")

st.title("💰 Customer Lifetime Value Prediction")
st.write("Enter customer details to predict CLTV")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
area = st.selectbox("Area", ["Urban", "Rural"])
qualification = st.selectbox("Qualification", ["High School", "Bachelor", "Others"])
income = st.selectbox("Income", ["0-5L", "5L-10L", "More than 10L"])
num_policies = st.selectbox("Number of Policies", ["1", "More than 1"])
policy = st.selectbox("Policy Type", ["A", "B", "C"])
type_of_policy = st.selectbox("Policy Level", ["Gold", "Silver", "Platinum"])

marital_status = st.selectbox("Marital Status", [0, 1])
vintage = st.slider("Customer Vintage", 0, 10)
claim_amount = st.number_input("Claim Amount", 0, 50000)

# ===== SIMPLE MODEL LOGIC (TEMP DEMO) =====
def simple_predict():
    base = 50000
    
    if income == "More than 10L":
        base += 50000
    elif income == "5L-10L":
        base += 30000

    if num_policies == "More than 1":
        base += 20000

    base += claim_amount * 1.5
    base += vintage * 2000

    return base

# Predict button
if st.button("Predict CLTV"):
    prediction = simple_predict()
    st.success(f"Estimated CLTV: ₹ {int(prediction)}")