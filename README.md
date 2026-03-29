# 📞 Telco Customer Churn Prediction

An end-to-end machine learning pipeline that predicts whether a telecom customer will churn, complete with a **Streamlit interactive dashboard** and **SHAP explainability**.

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `01_setup.py` | Install dependencies, load & validate dataset |
| `02_eda.py` | Exploratory Data Analysis — plots & insights |
| `03_clean.py` | Data cleaning — nulls, encoding, type fixes |
| `04_preprocess.py` | Feature engineering, one-hot encoding, scaling |
| `05_balance.py` | SMOTE oversampling for class imbalance |
| `06_split.py` | Stratified 80/20 train-test split |
| `07_train_logreg.py` | Logistic Regression baseline model |
| `08_train_rf.py` | Random Forest classifier (best model) |
| `09_eval.py` | Model evaluation — accuracy, F1, AUC, ROC |
| `10_shap.py` | SHAP global feature importance explanations |
| `11_app.py` | **Streamlit dashboard** (prediction + EDA + SHAP) |
| `12_run_all.py` | Master runner — executes all modules in order |

---

## 🚀 Quick Start

### 1. Clone repository
```bash
git clone https://github.com/RithanyaSivabalakrishnan/Churn.git
cd Churn
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn seaborn matplotlib streamlit joblib imbalanced-learn shap
```

### 3. Download dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root.

### 4. Run the full pipeline
```bash
python 12_run_all.py
```

### 5. Launch the dashboard
```bash
streamlit run 11_app.py
```

---

## 📊 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~80% | ~0.78 | ~0.86 |
| **Random Forest** | **~85%** | **~0.84** | **~0.92** |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn** — Models & preprocessing
- **imbalanced-learn** — SMOTE
- **SHAP** — Model explainability
- **Seaborn / Matplotlib** — Data visualisation
- **Streamlit** — Interactive dashboard
- **joblib** — Model serialisation

---

## 📸 Dashboard Features

- 🔮 **Real-time churn probability** with gauge display
- 🌊 **SHAP force plot** explaining each prediction
- 📊 **EDA visualisations** (contract, charges, tenure)
- 📈 **ROC curve comparison** between models
- 📋 **Customer summary cards**

---

## 📄 License

MIT License — free to use and modify.
