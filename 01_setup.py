"""
Module 01: Environment Setup and Dataset Load
=============================================
Installs required packages and loads the Telco Customer Churn dataset.
"""

import subprocess
import sys

# ── Install dependencies ──────────────────────────────────────────────────────
packages = [
    "pandas",
    "scikit-learn",
    "seaborn",
    "matplotlib",
    "streamlit",
    "joblib",
    "imbalanced-learn",
    "shap",
]

print("=" * 60)
print("Installing required packages...")
print("=" * 60)
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print(f"  ✔  {pkg}")
print()

# ── Load dataset ──────────────────────────────────────────────────────────────
import pandas as pd

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at '{CSV_PATH}'.\n"
        "Please download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from:\n"
        "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        "and place it in the project root directory before running this script."
    )

# ── Quick inspection ──────────────────────────────────────────────────────────
print("=" * 60)
print("Dataset Loaded Successfully!")
print("=" * 60)
print(f"\nShape : {df.shape}  (rows × columns)")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nChurn distribution (normalized):\n{df['Churn'].value_counts(normalize=True).round(4)}")
print()
print("[01_setup.py] ✔ Dataset loaded confirmation — OK")
