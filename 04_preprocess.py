"""
Module 04: Feature Engineering and Encoding (Updated)
=============================================
 - Converts specified columns to categorical type
 - One-hot encodes categorical columns
 - Standard-scales numeric columns
 - Engineers 'Tenure_Ratio' feature
 - Saves X, y, scaler, and feature names via joblib
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")

print("=" * 60)
print("Feature Engineering & Encoding")
print("=" * 60)

# ── Convert specified columns to categorical type ─────────────────────────────
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Only convert columns that exist in df
existing_categoricals = [col for col in categorical_columns if col in df.columns]
for col in existing_categoricals:
    df[col] = df[col].astype('category')
    print(f"  ✔  Converted '{col}' to categorical type")

print(f"\n  Processed {len(existing_categoricals)}/{len(categorical_columns)} categorical columns")

# ── Drop identifier column ────────────────────────────────────────────────────
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# ── Target ────────────────────────────────────────────────────────────────────
y = df["Churn"].copy()
df.drop(columns=["Churn"], inplace=True)

# ── Feature engineering: Tenure_Ratio ─────────────────────────────────────────
# Guard against division by zero
df["Tenure_Ratio"] = df["tenure"] / df["MonthlyCharges"].replace(0, 1)
print("\n  ✔  Engineered feature: Tenure_Ratio = tenure / MonthlyCharges")

# ── Categorical one-hot encoding ──────────────────────────────────────────────
dummies = pd.get_dummies(df[existing_categoricals], drop_first=True)
df.drop(columns=existing_categoricals, inplace=True)
df = pd.concat([df, dummies], axis=1)
print(f"  ✔  One-hot encoded {len(existing_categoricals)} categorical columns → {dummies.shape[1]} dummy cols")

# ── Numeric scaling ───────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(f"  ✔  StandardScaler applied to {len(num_cols)} numeric columns")

X = df.copy()
print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target vector shape        : {y.shape}")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(X, "X.pkl")
joblib.dump(y, "y.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
# Save dummy column names separately (useful in Streamlit app)
joblib.dump(list(dummies.columns), "dummy_columns.pkl")
print("\nSaved: X.pkl, y.pkl, scaler.pkl, feature_names.pkl, dummy_columns.pkl")
print("\n[04_preprocess_updated.py] ✔ Preprocessing complete.")