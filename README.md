# 🚀 Customer Lifetime Value Prediction (ML Project)

## 📌 Overview
This project focuses on predicting Customer Lifetime Value (CLTV) using machine learning techniques. The goal is to identify high-value customers based on demographic and policy-related data to support better business decisions.

---

## 🎯 Problem Statement
Predict the CLTV of customers using structured data such as income, policy details, and claim history.

---

## 🧠 Approach

- Performed data preprocessing and cleaning
- Applied feature engineering:
  - Ratio features (income per policy, claim per policy)
  - Interaction features (income × claim, claim × vintage)
  - Log transformation for skewed data
- Used One-Hot Encoding for categorical variables
- Built an XGBoost regression model
- Implemented Cross Validation to ensure generalization
- Applied Ensemble Learning (XGBoost + RandomForest)
- Optimized predictions using weighted averaging and clipping

---

## 📊 Results

- **R² Score:** 0.155+
- **Leaderboard Rank:** Top 100 (#64)
- Achieved stable performance across validation and leaderboard

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost

---

## 📁 Project Structure
CLTV-Prediction-ML/
│
├── data/ # Dataset (ignored in repo)
├── src/ # Source code
│ └── main.py
├── submission/ # Final submission file
│ └── submission.csv
├── README.md
├── .gitignore


---

## 🏆 Achievement

- Ranked **Top 100 (#64)** in Analytics Vidhya Data Scientist Hiring Hackathon
- Improved model performance through iterative experimentation
- Built a complete ML pipeline from scratch

---

## 🚀 Highlights

- Solved a real-world business problem (CLTV prediction)
- Applied advanced feature engineering techniques
- Used ensemble learning to improve model stability
- Demonstrated iterative model improvement and debugging

---

## 💼 Author

**Roshan Shirke**