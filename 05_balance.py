"""
Module 05: Handle Class Imbalance with SMOTE
=============================================
Applies Synthetic Minority Oversampling Technique (SMOTE) to balance the
training data and saves the resampled arrays.
"""

import joblib
from imblearn.over_sampling import SMOTE
import pandas as pd

# ── Load features & target ────────────────────────────────────────────────────
X = joblib.load("X.pkl")
y = joblib.load("y.pkl")

print("=" * 60)
print("Handling Class Imbalance with SMOTE")
print("=" * 60)

# ── Before ────────────────────────────────────────────────────────────────────
print("\nClass distribution BEFORE SMOTE:")
before = y.value_counts()
print(f"  Class 0 (No Churn) : {before.get(0, 0):>5} samples")
print(f"  Class 1 (Churn)    : {before.get(1, 0):>5} samples")
print(f"  Imbalance ratio    :  {before.get(1, 0)/before.get(0, 1):.3f}")

# ── Apply SMOTE ───────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# ── After ─────────────────────────────────────────────────────────────────────
after = pd.Series(y_res).value_counts()
print("\nClass distribution AFTER SMOTE:")
print(f"  Class 0 (No Churn) : {after.get(0, 0):>5} samples")
print(f"  Class 1 (Churn)    : {after.get(1, 0):>5} samples")
print(f"  Balance ratio      :  {after.get(1, 0)/after.get(0, 1):.3f}")

# ── Save resampled data ───────────────────────────────────────────────────────
joblib.dump(X_res, "X_resampled.pkl")
joblib.dump(y_res, "y_resampled.pkl")
print("\nSaved: X_resampled.pkl, y_resampled.pkl")
print("\n[05_balance.py] ✔ SMOTE resampling complete.")
