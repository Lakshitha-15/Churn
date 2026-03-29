"""
Module 05: Handle Imbalance with SMOTE (CRITICAL FIX)
=============================================
 - Forces X to numeric (handles hidden NaNs/strings)
 - Cleans data before SMOTE
 - Saves resampled numpy arrays
"""

import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

print("=" * 60)
print("Handling Class Imbalance with SMOTE")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading X.pkl and y.pkl...")
X_df = joblib.load("X.pkl")
y = joblib.load("y.pkl")

print(f"Initial X shape: {X_df.shape}, dtypes: {X_df.dtypes.value_counts()}")

# ── CRITICAL FIX: Force numeric conversion + handle NaNs ──────────────────────
print("\n🔧 Converting to numeric and cleaning...")

# Method 1: Replace problematic values first
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(0)  # Fill NaNs with 0 (common for scaled features)

# Method 2: Force ALL columns to float64
for col in X_df.columns:
    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

# Final cleanup
X_df = X_df.fillna(0)
X_df.replace([np.inf, -np.inf], 0, inplace=True)

# Convert to numpy array
X = X_df.values.astype(np.float64)
print(f"Cleaned X shape: {X.shape}, dtype: {X.dtype}")
print(f"Any NaN? {np.isnan(X).any()}")
print(f"Any inf? {np.isinf(X).any()}")

# ── Check original distribution ───────────────────────────────────────────────
print("\nClass distribution BEFORE SMOTE:")
orig_dist = Counter(y)
print(f"  Class 0 (No Churn) : {orig_dist[0]} samples")
print(f"  Class 1 (Churn)    : {orig_dist[1]} samples")
print(f"  Imbalance ratio    : {orig_dist[1]/orig_dist[0]:.3f}")

# ── Apply SMOTE ───────────────────────────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)

print("\n✅ SMOTE applied successfully!")
print("Class distribution AFTER SMOTE:")
res_dist = Counter(y_res)
print(f"  Class 0 (No Churn) : {res_dist[0]} samples")
print(f"  Class 1 (Churn)    : {res_dist[1]} samples")
print(f"  New shape          : {X_res.shape}")

# ── Save resampled data ───────────────────────────────────────────────────────
joblib.dump(X_res, "X_resampled.pkl")
joblib.dump(y_res, "y_resampled.pkl")
joblib.dump(X_df.columns.tolist(), "feature_names_clean.pkl")  # Save clean feature names
print("\nSaved: X_resampled.pkl, y_resampled.pkl, feature_names_clean.pkl")

print("\n[05_balance.py] ✅ Balancing complete!")