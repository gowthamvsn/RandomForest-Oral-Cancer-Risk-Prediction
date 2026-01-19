# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv("oral_cancer_prediction_dataset.csv")

leakage_cols = [
    "Tumor Size (cm)", "Cancer Stage",
    "Survival Rate (5-Year, %)", "Cost of Treatment (USD)",
    "Economic Burden (Lost Workdays per Year)", "Early Diagnosis"
]

data = data.drop(columns=leakage_cols)
y = (data["Oral Cancer (Diagnosis)"] == "Yes").astype(int)
X = pd.get_dummies(data.drop(columns=["Oral Cancer (Diagnosis)", "ID"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_proba = rf.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

RocCurveDisplay.from_estimator(rf, X_test_scaled, y_test)
plt.show()
