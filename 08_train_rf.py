"""
Module 08: Train Random Forest
================================
Trains a Random Forest classifier, reports AUC and top-5 feature importances,
and saves the fitted model.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── Load splits ───────────────────────────────────────────────────────────────
X_train = joblib.load("X_train.pkl")
X_test  = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test  = joblib.load("y_test.pkl")

print("=" * 60)
print("Training Random Forest Classifier")
print("=" * 60)

# ── Train ─────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("\n  ✔  Model trained (100 trees)")

# ── AUC ───────────────────────────────────────────────────────────────────────
preds = rf.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, preds)
print(f"\n  ROC-AUC Score (Random Forest): {auc:.4f}")

# ── Top-5 feature importances ─────────────────────────────────────────────────
feature_names = joblib.load("feature_names.pkl")
importances = pd.Series(rf.feature_importances_, index=feature_names)
top5 = importances.nlargest(5)
print("\n  Top-5 Feature Importances:")
for feat, imp in top5.items():
    print(f"    {feat:<40} {imp:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(rf, "rf_model.pkl")
print("\nSaved: rf_model.pkl")
print("\n[08_train_rf.py] ✔ Random Forest training complete.")
