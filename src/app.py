import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== PAGE CONFIG (must be FIRST streamlit call) =====
st.set_page_config(
    page_title="CLTV Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== LOAD MODEL (cached - loads once, instant for all visitors) =====
@st.cache_resource(show_spinner=False)
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rf_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "rf_model.pkl"))
    model_columns = joblib.load(os.path.join(BASE_DIR, "..", "models", "columns.pkl"))
    return rf_model, model_columns

# Show spinner AFTER defining load_model
with st.spinner("🚀 Loading model... please wait a moment"):
    rf_model, model_columns = load_model()

# ===== HEADER =====
st.title("🚀 CLTV Prediction Dashboard")
st.markdown("""
Predict **Customer Lifetime Value** using Machine Learning &nbsp;|&nbsp; **Model:** Random Forest
""")
st.divider()

# ===== SIDEBAR INPUTS =====
st.sidebar.header("⚙️ Customer Inputs")

gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
area           = st.sidebar.selectbox("Area", ["Urban", "Rural"])
qualification  = st.sidebar.selectbox("Qualification", ["High School", "Bachelor", "Others"])
income         = st.sidebar.selectbox("Income", ["0-5L", "5L-10L", "More than 10L"])
num_policies   = st.sidebar.selectbox("Number of Policies", ["1", "More than 1"])
policy         = st.sidebar.selectbox("Policy Type", ["A", "B", "C"])
type_of_policy = st.sidebar.selectbox("Policy Level", ["Gold", "Silver", "Platinum"])
marital_status = st.sidebar.selectbox("Marital Status", [0, 1])
vintage        = st.sidebar.slider("Customer Vintage (years)", 0, 10, 3)
claim_amount   = st.sidebar.slider("Claim Amount (₹)", 0, 500000, 50000, step=1000)

# ===== PREDICT BUTTON =====
if st.sidebar.button("🔮 Predict CLTV", use_container_width=True, type="primary"):

    # ===== BUILD INPUT DATAFRAME =====
    input_data = pd.DataFrame({
        "gender":          [gender],
        "area":            [area],
        "qualification":   [qualification],
        "income":          [income],
        "num_policies":    [num_policies],
        "policy":          [policy],
        "type_of_policy":  [type_of_policy],
        "marital_status":  [marital_status],
        "vintage":         [vintage],
        "claim_amount":    [claim_amount]
    })

    # ===== PREPROCESSING =====
    input_data["num_policies"] = input_data["num_policies"].replace({"More than 1": 2}).astype(int)

    income_map = {"0-5L": 1, "5L-10L": 2, "More than 10L": 3}
    input_data["income"] = input_data["income"].map(income_map)

    # ===== FEATURE ENGINEERING =====
    input_data["income_per_policy"] = input_data["income"] / (input_data["num_policies"] + 1)
    input_data["claim_per_policy"]  = input_data["claim_amount"] / (input_data["num_policies"] + 1)
    input_data["policy_per_year"]   = input_data["num_policies"] / (input_data["vintage"] + 1)
    input_data["income_x_claim"]    = input_data["income"] * input_data["claim_amount"]
    input_data["log_claim"]         = np.log1p(input_data["claim_amount"])
    input_data["claim_to_income"]   = input_data["claim_amount"] / (input_data["income"] + 1)
    input_data["claim_per_year"]    = input_data["claim_amount"] / (input_data["vintage"] + 1)

    # ===== ENCODING + ALIGN COLUMNS =====
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # ===== PREDICT =====
    final_pred = rf_model.predict(input_data)
    cltv = int(final_pred[0])

    # ===== RESULTS =====
    st.markdown("## 📊 Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Predicted CLTV", f"₹ {cltv:,}")

    with col2:
        if cltv > 500000:
            st.success("🌟 High Value Customer")
        elif cltv > 200000:
            st.warning("⚡ Medium Value Customer")
        else:
            st.error("🔻 Low Value Customer")

    with col3:
        percentile = min(100, int((cltv / 650000) * 100))
        st.metric("📈 Value Percentile", f"{percentile}%")

    st.divider()

    # ===== CHARTS =====
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 Your CLTV vs Benchmarks")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        labels = ["Low\nThreshold", "Medium\nThreshold", "High\nThreshold", "Your\nCLTV"]
        values = [200000, 400000, 600000, cltv]
        colors = ["#ff6b6b", "#ffa94d", "#51cf66", "#339af0"]
        bars = ax2.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        ax2.axhline(y=cltv, color="#339af0", linestyle="--", linewidth=1.5, label=f"Your CLTV: ₹{cltv:,}")
        ax2.set_ylabel("CLTV (₹)")
        ax2.set_title("CLTV Benchmark Comparison")
        ax2.legend(fontsize=8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x/1000)}K"))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_b:
        st.markdown("#### 🔍 Top 10 Feature Importances")
        importance_df = pd.DataFrame({
            "Feature":    model_columns,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#339af0")
        ax3.set_xlabel("Importance Score")
        ax3.set_title("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()

    # ===== CUSTOMER SUMMARY =====
    with st.expander("🧾 View Processed Input Data"):
        st.dataframe(input_data, use_container_width=True)

else:
    # Default state — show instructions
    st.info("👈 Fill in the customer details in the sidebar and click **Predict CLTV** to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌟 High Value", "> ₹5,00,000")
    with col2:
        st.metric("⚡ Medium Value", "₹2,00,000 – ₹5,00,000")
    with col3:
        st.metric("🔻 Low Value", "< ₹2,00,000")