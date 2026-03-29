"""
Module 09: Model Evaluation
=============================
Loads both trained models and produces a comprehensive comparison:
  - Accuracy, F1, Confusion Matrix, Classification Report
  - ROC Curve overlay plot
  - Saves 'eval_results.csv'
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve,
)

# ── Load data & models ────────────────────────────────────────────────────────
X_test  = joblib.load("X_test.pkl")
y_test  = joblib.load("y_test.pkl")
logreg  = joblib.load("logreg_model.pkl")
rf      = joblib.load("rf_model.pkl")

print("=" * 60)
print("Model Evaluation & Comparison")
print("=" * 60)

models = {"Logistic Regression": logreg, "Random Forest": rf}
results = []

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC":  round(auc, 4),
        "TN": cm[0, 0], "FP": cm[0, 1],
        "FN": cm[1, 0], "TP": cm[1, 1],
    })

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

# ── ROC plot ──────────────────────────────────────────────────────────────────
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("\n  ✔  Saved: roc_curves.png")

# ── Comparison table ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Comparison Table")
print("=" * 60)
print(results_df.to_string(index=False))

# ── Save results ──────────────────────────────────────────────────────────────
results_df.to_csv("eval_results.csv", index=False)
print("\nSaved: eval_results.csv")
print("\n[09_eval.py] ✔ Evaluation complete  (RF expected to outperform LogReg).")
