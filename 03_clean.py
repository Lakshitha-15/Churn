"""
Module 03: Data Cleaning
========================
Cleans the raw Telco Churn dataset:
  - Coerces TotalCharges to numeric
  - Fills nulls with column medians
  - Replaces service-placeholder strings with 'No'
  - Encodes the Churn target as 0/1
Saves the cleaned DataFrame to 'cleaned_data.csv'.
"""

import pandas as pd

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 60)
print("Data Cleaning")
print("=" * 60)

# ── Null counts BEFORE cleaning ───────────────────────────────────────────────
print("\nNull counts BEFORE cleaning:")
null_before = df.isnull().sum()
print(null_before[null_before > 0].to_string() if null_before.sum() > 0 else "  (none)")

# ── Fix TotalCharges ──────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\n  ✔  TotalCharges coerced to numeric (NaNs introduced: "
      f"{df['TotalCharges'].isnull().sum()})")

# ── Fill numeric nulls with median ───────────────────────────────────────────
df.fillna(df.median(numeric_only=True), inplace=True)

# ── Replace service-placeholder strings ───────────────────────────────────────
replace_map = {"No internet service": "No", "No phone service": "No"}
df.replace(replace_map, inplace=True)
print("  ✔  Replaced 'No internet service' and 'No phone service' with 'No'")

# ── Encode Churn ──────────────────────────────────────────────────────────────
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("  ✔  Churn encoded as integer (1 = Yes, 0 = No)")

# ── Null counts AFTER cleaning ────────────────────────────────────────────────
print("\nNull counts AFTER cleaning:")
null_after = df.isnull().sum()
print(null_after[null_after > 0].to_string() if null_after.sum() > 0 else "  (none — all clear!)")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("cleaned_data.csv", index=False)
print(f"\nSaved cleaned dataset → cleaned_data.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")
print("\n[03_clean.py] ✔ Data cleaning complete.")
