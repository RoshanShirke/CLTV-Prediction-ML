import pandas as pd
import numpy as np

RUN_CV = False   # Turn OFF before final submission

# ====== LOAD DATA ======
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

test_ids = test['id']

# ====== BASIC CONVERSIONS ======

train['num_policies'] = train['num_policies'].replace({'More than 1': 2}).astype(int)
test['num_policies'] = test['num_policies'].replace({'More than 1': 2}).astype(int)

income_map = {
    '0-5L': 1,
    '5L-10L': 2,
    'More than 10L': 3
}
train['income'] = train['income'].map(income_map)
test['income'] = test['income'].map(income_map)

# ====== FEATURE ENGINEERING ======

train['income_per_policy'] = train['income'] / (train['num_policies'] + 1)
test['income_per_policy'] = test['income'] / (test['num_policies'] + 1)

train['claim_per_policy'] = train['claim_amount'] / (train['num_policies'] + 1)
test['claim_per_policy'] = test['claim_amount'] / (test['num_policies'] + 1)

train['policy_per_year'] = train['num_policies'] / (train['vintage'] + 1)
test['policy_per_year'] = test['num_policies'] / (test['vintage'] + 1)

train['income_x_claim'] = train['income'] * train['claim_amount']
test['income_x_claim'] = test['income'] * test['claim_amount']

train['log_claim'] = np.log1p(train['claim_amount'])
test['log_claim'] = np.log1p(test['claim_amount'])

train['claim_to_income'] = train['claim_amount'] / (train['income'] + 1)
test['claim_to_income'] = test['claim_amount'] / (test['income'] + 1)

train['claim_per_year'] = train['claim_amount'] / (train['vintage'] + 1)
test['claim_per_year'] = test['claim_amount'] / (test['vintage'] + 1)

# ====== ONE HOT ENCODING ======

combined = pd.concat([train.drop('cltv', axis=1), test], axis=0)
combined = pd.get_dummies(combined, drop_first=True)

train_encoded = combined.iloc[:len(train), :].copy()
test_encoded = combined.iloc[len(train):, :].copy()

train_encoded['cltv'] = train['cltv'].values

# ====== PREPARE DATA ======

X = train_encoded.drop(['id', 'cltv'], axis=1)
y = train_encoded['cltv']   # ❗ NO LOG TRANSFORM

# ====== MODELS ======

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

model = XGBRegressor(
    n_estimators=1100,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.3,
    random_state=42,
    n_jobs=-1
)

model2 = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# ====== CROSS VALIDATION ======

if RUN_CV:
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_tr, y_tr)
        model2.fit(X_tr, y_tr)

        pred1 = model.predict(X_val)
        pred2 = model2.predict(X_val)

        preds = (0.7 * pred1) + (0.3 * pred2)

        score = r2_score(y_val, preds)
        scores.append(score)

    print("\nCross Validation R2 Scores:", scores)
    print("Mean R2 Score:", np.mean(scores))

# ====== FINAL TRAIN ======

model.fit(X, y)
model2.fit(X, y)

# ====== TEST PREDICTION ======

test_data = test_encoded.drop('id', axis=1)

pred1 = model.predict(test_data)
pred2 = model2.predict(test_data)

predictions = (0.7 * pred1) + (0.3 * pred2)

# 🔥 SMART CLIPPING
predictions = np.clip(predictions, 25000, 650000)

# ====== SUBMISSION ======

submission = pd.DataFrame({
    'id': test_ids,
    'cltv': predictions
})

submission.to_csv("../submission/submission.csv", index=False)

print("\nSubmission file created successfully ✅")

import joblib
import os

# Create models folder if not exists
os.makedirs("../models", exist_ok=True)

# Save models
joblib.dump(model, "../models/xgb_model.pkl")
joblib.dump(model2, "../models/rf_model.pkl")

# Save column structure
joblib.dump(X.columns, "../models/columns.pkl")

print("✅ Models saved successfully")