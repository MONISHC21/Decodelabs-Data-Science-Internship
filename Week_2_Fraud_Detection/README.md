# Week 2: Fraud Detection Pipeline

## Overview

This project focuses on building a machine learning pipeline to detect fraudulent financial transactions from a highly imbalanced dataset. Fraud detection is a critical application of Data Science in banking, fintech, and cybersecurity, where identifying rare fraudulent activities can prevent significant financial losses.

The project demonstrates a complete supervised learning workflow, including data preprocessing, class imbalance handling, model training, evaluation, and business insights generation.

---

## Objectives

* Analyze transaction data to identify fraudulent patterns.
* Handle severe class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
* Train and compare multiple classification algorithms.
* Evaluate model performance using Precision, Recall, F1-Score, and ROC-AUC.
* Generate actionable insights for fraud prevention.

---

## Dataset

**Dataset:** Credit Card Fraud Detection Dataset

### Features

* Transaction attributes and anonymized variables
* Time and Amount features
* Binary Target Variable:

  * 0 → Legitimate Transaction
  * 1 → Fraudulent Transaction

### Challenges

* Highly imbalanced classes
* Rare fraud events
* High cost of false negatives

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed:

* Dataset inspection
* Missing value analysis
* Transaction amount distribution
* Fraud vs Non-Fraud distribution
* Feature correlation analysis

Generated Visualizations:

* Class Distribution
* Transaction Amount Distribution
* Correlation Heatmap

---

### 2. Data Preprocessing

Steps performed:

* Missing value verification
* Feature scaling using StandardScaler
* Train-Test Split
* Class balancing using SMOTE

Benefits of SMOTE:

* Reduces model bias toward majority class
* Improves fraud detection capability
* Enhances recall performance

---

### 3. Machine Learning Models

The following algorithms were trained and evaluated:

#### Logistic Regression

A baseline linear classification model used for comparison.

#### Random Forest Classifier

An ensemble learning model capable of capturing complex fraud patterns.

#### Gradient Boosting Classifier

A boosting-based model that sequentially improves prediction performance.

---

### 4. Model Evaluation

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

Additional evaluations:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Cross Validation

---

## Generated Outputs

### Charts

Located in:

```text
charts/
```

Files:

```text
class_distribution.png
smote_distribution.png
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
feature_importance.png
```

### Models

Located in:

```text
models/
```

Files:

```text
random_forest.pkl
logistic_regression.pkl
gradient_boosting.pkl
```

### Predictions

Located in:

```text
outputs/
```

Files:

```text
fraud_predictions.csv
```

---

## Key Findings

* Fraud transactions represent only a small percentage of total transactions.
* Class imbalance significantly affects model performance.
* SMOTE improved the model's ability to identify minority fraud cases.
* Random Forest and Gradient Boosting demonstrated strong fraud detection capabilities.
* Precision and Recall provided more meaningful insights than Accuracy alone.

---

## Business Impact

A successful fraud detection model can:

* Reduce financial losses.
* Improve customer trust.
* Detect suspicious activities in real time.
* Support compliance and risk management initiatives.
* Strengthen overall cybersecurity infrastructure.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Joblib

---

## Repository Structure

```text
Week_2_Fraud_Detection/

├── data/
├── notebooks/
│   └── fraud_detection.ipynb
│
├── charts/
├── models/
├── outputs/
├── reports/
├── src/
└── README.md
```

---

## Future Improvements

* Hyperparameter optimization using GridSearchCV.
* XGBoost and LightGBM implementation.
* Real-time fraud detection deployment.
* Model monitoring and drift detection.
* Explainable AI using SHAP and LIME.

---

## Conclusion

This project successfully demonstrates an end-to-end Fraud Detection Pipeline using machine learning techniques. Through proper preprocessing, class balancing with SMOTE, model training, and evaluation, the project highlights how Data Science can be leveraged to identify fraudulent transactions and support data-driven risk management decisions.
