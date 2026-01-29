# Multiclass Liver Disease Classification with CatBoost & SHAP

## 📌 Project Overview
This project builds a **high-performance multiclass classification model** for liver disease diagnosis using **CatBoost**, with **robust cross-validation** and **model explainability via SHAP**.

The goal is not just accuracy, but:
- Proper handling of categorical features
- Class imbalance mitigation
- Reliable cross-validated performance
- Transparent, interpretable predictions (clinically important)

This project is designed to meet **industry and healthcare ML standards**.

---

## 🧠 Problem Statement
Clinical liver datasets often contain:
- Mixed numerical and categorical features
- Class imbalance
- Non-linear relationships

Traditional models struggle without heavy preprocessing.  
**CatBoost** is chosen because it natively handles categorical variables and reduces target leakage.

---

## 📊 Dataset
- Liver disease clinical dataset
- Multiclass target variable
- Combination of laboratory values and demographic features

> ⚠️ Patient identifiers were excluded to prevent data leakage.

---

## ⚙️ Tech Stack
- Python 3.10+
- CatBoost
- Scikit-learn
- SHAP
- Pandas / NumPy
- Matplotlib

---

## 🔄 Data Preparation
- Removed non-informative identifiers
- Explicit identification of categorical features
- Stratified train-test split to preserve class distribution

```python
cat_features_idx = [
    X.columns.get_loc(col)
    for col in categorical_columns
    if col in X.columns
]
