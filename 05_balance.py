"""
Module 05: Handle Imbalance with SMOTE (Fixed)
=============================================
 - Loads X as numpy array (ensures numeric-only)
 - Applies SMOTE for class balancing
 - Saves resampled data
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
X_df = joblib.load("X.pkl")  # Load as DataFrame first
y = joblib.load("y.pkl")

# CRITICAL FIX: Convert to numpy array (SMOTE requirement)
X = X_df.values  # .values ensures numeric numpy array
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, unique values: {np.unique(y)}")

# Verify all numeric
if not np.issubdtype(X.dtype, np.number):
    raise ValueError("X contains non-numeric data! Check preprocessing.")

# ── Check original distribution ───────────────────────────────────────────────
print("\nClass distribution BEFORE SMOTE:")
orig_dist = Counter(y)
print(f"  Class 0 (No Churn) : {orig_dist[0]} samples")
print(f"  Class 1 (Churn)    : {orig_dist[1]} samples")
print(f"  Imbalance ratio    : {orig_dist[1]/orig_dist[0]:.3f}")

# ── Apply SMOTE ───────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)

print("\nSMOTE applied successfully!")
print("Class distribution AFTER SMOTE:")
res_dist = Counter(y_res)
print(f"  Class 0 (No Churn) : {res_dist[0]} samples")
print(f"  Class 1 (Churn)    : {res_dist[1]} samples")
print(f"  New shape          : {X_res.shape}")

# ── Save resampled data ───────────────────────────────────────────────────────
joblib.dump(X_res, "X_resampled.pkl")
joblib.dump(y_res, "y_resampled.pkl")
print("\nSaved: X_resampled.pkl, y_resampled.pkl")

print("\n[05_balance.py] ✔ Balancing complete.")
