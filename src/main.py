import pandas as pd
import numpy as np

# ====== LOAD DATA ======
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test['id']

# ====== BASIC CONVERSIONS ======

# num_policies
train['num_policies'] = train['num_policies'].replace({'More than 1': 2}).astype(int)
test['num_policies'] = test['num_policies'].replace({'More than 1': 2}).astype(int)

# income mapping
income_map = {
    '0-5L': 1,
    '5L-10L': 2,
    'More than 10L': 3
}
train['income'] = train['income'].map(income_map)
test['income'] = test['income'].map(income_map)

# ====== FEATURE ENGINEERING ======

# ratios
train['income_per_policy'] = train['income'] / (train['num_policies'] + 1)
test['income_per_policy'] = test['income'] / (test['num_policies'] + 1)

train['claim_ratio'] = train['claim_amount'] / (train['income'] + 1)
test['claim_ratio'] = test['claim_amount'] / (test['income'] + 1)

train['policy_per_year'] = train['num_policies'] / (train['vintage'] + 1)
test['policy_per_year'] = test['num_policies'] / (test['vintage'] + 1)

# 🔥 interaction features
train['income_x_claim'] = train['income'] * train['claim_amount']
test['income_x_claim'] = test['income'] * test['claim_amount']

train['policy_x_vintage'] = train['num_policies'] * train['vintage']
test['policy_x_vintage'] = test['num_policies'] * test['vintage']

# 🔥 log transform (very powerful)
train['log_claim'] = np.log1p(train['claim_amount'])
test['log_claim'] = np.log1p(test['claim_amount'])

# ====== ONE HOT ENCODING ======
combined = pd.concat([train.drop('cltv', axis=1), test], axis=0)
combined = pd.get_dummies(combined, drop_first=True)

train_encoded = combined.iloc[:len(train), :]
test_encoded = combined.iloc[len(train):, :]

train_encoded['cltv'] = train['cltv'].values

# ====== PREPARE DATA ======
X = train_encoded.drop(['id', 'cltv'], axis=1)
y = train_encoded['cltv']

# ====== SPLIT ======
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== FINAL MODEL (TUNED XGBOOST) ======
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Training Completed ✅")

# ====== VALIDATION ======
from sklearn.metrics import r2_score

val_pred = model.predict(X_val)
score = r2_score(y_val, val_pred)

print("\nValidation R2 Score:", score)

# ====== FINAL TRAIN ======
model.fit(X, y)

# ====== PREDICT ======
test_data = test_encoded.drop('id', axis=1)
predictions = model.predict(test_data)

# ====== SUBMISSION ======
submission = pd.DataFrame({
    'id': test_ids,
    'cltv': predictions
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully ✅")