"""
Module 06: Train-Test Split
============================
Splits the SMOTE-resampled data into 80% train / 20% test sets
with stratification to preserve class balance.
"""

import joblib
from sklearn.model_selection import train_test_split

# ── Load resampled data ───────────────────────────────────────────────────────
X_res = joblib.load("X_resampled.pkl")
y_res = joblib.load("y_resampled.pkl")

print("=" * 60)
print("Train-Test Split  (80% / 20%, stratified)")
print("=" * 60)
print(f"\nFull resampled dataset : {X_res.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res,
    test_size=0.2,
    stratify=y_res,
    random_state=42,
)

# ── Print shapes ──────────────────────────────────────────────────────────────
print(f"\n  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")

# ── Save splits ───────────────────────────────────────────────────────────────
joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test,  "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test,  "y_test.pkl")
print("\nSaved: X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl")
print("\n[06_split.py] ✔ Train-test split complete.")
