# ============================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
data = pd.read_csv("oral_cancer_prediction_dataset.csv")

# Encode categorical variables
df_encoded = data.copy()
label_encoders = {}

if "Country" in df_encoded.columns:
    df_encoded = pd.get_dummies(df_encoded, columns=["Country"], drop_first=True)

categorical_cols = df_encoded.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df_encoded.drop(columns=["ID"]).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Remove multicollinearity using VIF
X = df_encoded.drop(columns=["Oral Cancer (Diagnosis)", "ID"])
y = df_encoded["Oral Cancer (Diagnosis)"]

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

high_vif_cols = vif[vif["VIF"] > 5]["Feature"].tolist()
X_reduced = X.drop(columns=high_vif_cols)

print("Removed high-VIF features:", high_vif_cols)
