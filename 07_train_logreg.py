"""
Module 07: Train Logistic Regression
======================================
Trains a Logistic Regression classifier on the training split,
evaluates on the test split, and saves the fitted model.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Load splits ───────────────────────────────────────────────────────────────
X_train = joblib.load("X_train.pkl")
X_test  = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print("=" * 60)
print("Training Logistic Regression")
print("=" * 60)

# ── Train ─────────────────────────────────────────────────────────────────────
logreg = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
logreg.fit(X_train, y_train)
print("\n  ✔  Model trained successfully")

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = logreg.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, preds)
print(f"\n  ROC-AUC Score (Logistic Regression): {auc:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(logreg, "logreg_model.pkl")
print("\nSaved: logreg_model.pkl")
print("\n[07_train_logreg.py] ✔ Logistic Regression training complete.")
