# Multiclass Liver Disease Classification with CatBoost

## 📌 Project Overview
This project builds a **high-performance multiclass classification model** for liver disease diagnosis using **CatBoost**, with **robust cross-validation**.

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

CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    subsample=0.8,
    bootstrap_type='Bernoulli',
    auto_class_weights='Balanced',
    random_seed=42,
    cat_features=cat_features_idx,
    verbose=100
) 
```
---
### Why These Choices?

- TotalF1 → Better metric for imbalanced multiclass problems

- auto_class_weights → Prevents minority class collapse

- Depth & Regularization → Controls overfitting

- Bernoulli bootstrap → Improves generalization
  
```python
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring='f1_weighted'
)
```

- Metric: F1-weighted

- Why: Accuracy is misleading in imbalanced medical datasets

  Note: CatBoost CV is computationally expensive by design (this is expected, not a bug)
---
  ## 🏆 Performance

- Multiclass ROC-AUC: ~0.97

- Cross-validated F1-weighted: Stable across folds

- This level of performance indicates strong separability without obvious overfitting.
  
  ---

## 📁 Project Structure
```python
├── Liver_dataset_2.ipynb
├── README.md
├── requirements.txt
```
---
🧪 Reproducibility

- Fixed random seeds

- Stratified sampling

- Explicit feature handling
  
  ---
  ## 📌 Key Takeaways

- CatBoost is ideal for structured medical data

- Cross-validation cost ≠ inefficiency — its robustness

- Explainability is non-negotiable in healthcare ML

- This project is interview-ready and production-aware

  ---
  ## 🚀 Future Improvements

- External validation on unseen hospital data

- Probability calibration (Platt / Isotonic)

- Model deployment as a clinical decision support API
---
## 👤 Author

Joseph
Medical Laboratory Scientist → Data Analyst → ML / AI Engineer
