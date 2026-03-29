"""
Module 12: Master Runner Script
=================================
Runs modules 01–11 in order, skipping those whose output files already exist.
Usage:  python 12_run_all.py
"""

import subprocess
import sys
import os

# ── Pipeline definition ───────────────────────────────────────────────────────
# (script, output_sentinel_files_that_mark_it_as_done)
PIPELINE = [
    ("01_setup.py",       []),                          # no output file — always run
    ("02_eda.py",         ["eda_tenure.png", "eda_charges.png"]),
    ("03_clean.py",       ["cleaned_data.csv"]),
    ("04_preprocess.py",  ["X.pkl", "y.pkl", "scaler.pkl"]),
    ("05_balance.py",     ["X_resampled.pkl", "y_resampled.pkl"]),
    ("06_split.py",       ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]),
    ("07_train_logreg.py",["logreg_model.pkl"]),
    ("08_train_rf.py",    ["rf_model.pkl"]),
    ("09_eval.py",        ["eval_results.csv", "roc_curves.png"]),
    ("10_shap.py",        ["shap_summary.png"]),
    ("11_app.py",         []),                          # Streamlit app — not auto-run
]

def all_exist(files):
    return all(os.path.exists(f) for f in files) if files else False

print("=" * 60)
print("Customer Churn Prediction — Master Runner")
print("=" * 60)

for script, outputs in PIPELINE:
    if script == "11_app.py":
        # Skip; instruct user to run manually
        continue

    if outputs and all_exist(outputs):
        print(f"\n  ⏭  SKIP  {script}  (outputs already exist)")
        continue

    print(f"\n  ▶  Running {script} ...")
    print("  " + "─" * 56)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # stream output live
    )
    if result.returncode != 0:
        print(f"\n  ✖  ERROR in {script}  (exit code {result.returncode})")
        print("     Pipeline halted. Fix the error above and re-run.")
        sys.exit(result.returncode)
    print(f"\n  ✔  {script} completed successfully")

print("\n" + "=" * 60)
print("Pipeline complete!  All modules executed successfully.")
print("=" * 60)
print("\nNext step — launch the Streamlit dashboard:")
print("\n    streamlit run 11_app.py\n")
