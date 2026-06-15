# Week 1: Advanced EDA & Feature Engineering

**Internship Organization:** Decode Labs  
**Project Phase:** Phase 1 — Ingestion, Advanced Exploratory Data Analysis, and Structural Preprocessing  
**Prepared By:** C Monish Nandha Balan  
**Date:** June 2026  

---

## 📌 Executive Summary

The objective of Week 1 was to establish a rigorous, production-ready data engineering and Exploratory Data Analysis (EDA) pipeline. Operating on raw retail transaction logs, this pipeline enforces statistical controls, handles missing data structures, neutralizes variance-distorting outliers, and constructs engineered mathematical vectors to prepare high-fidelity inputs for downstream machine learning estimators.

---

## 🗄️ Dataset Overview

The ingestion boundary processes a single source dataset containing transactional logs from e-commerce checkouts.

* **Total Records Processed:** 1,200 unique order logs
* **Initial Matrix Dimensions:** 14 structural columns
* **Data Chronology:** January 2023 to June 2025
* **Gross Revenue Ingested:** $1,264,761.96

### Core Ingestion Dimensions

| Column Name | Data Type | Missing Records | Analytical System Role |
| :--- | :--- | :--- | :--- |
| `OrderID` | String | 0 | Unique identifier; dropped before model ingestion |
| `Date` | Datetime | 0 | Temporal anchor; extracted into year/month vectors |
| `CustomerID` | String | 0 | User identifier; dropped before model ingestion |
| `Product` | Categorical | 0 | Categorical predictor; processed via One-Hot Encoding |
| `Quantity` | Integer | 0 | Primary transaction volume indicator |
| `UnitPrice` | Float | 0 | Numerical scalar; scaled during preprocessing |
| `PaymentMethod` | Categorical | 0 | Categorical predictor; processed via One-Hot Encoding |
| `ItemsInCart` | Integer | 0 | Strong continuous numerical predictor feature |
| `CouponCode` | Categorical | 309 | Contained missingness; explicitly imputed |
| `TotalPrice` | Float | 0 | Numerical target; dropped to prevent collinearity |

---

## 🛠️ Data Preprocessing & Cleaning Pipeline

Machine learning algorithms are sensitive to missing inputs, outliers, and highly correlated features. The data was processed through a strict pipeline to ensure structural integrity:

### 1. Missing Value Resolution
* **Issue:** The ingestion pipeline identified a 25.75% missingness footprint (309 records) exclusively within the `CouponCode` feature.
* **Action:** To avoid data loss through row deletion, null entries were explicitly imputed with the categorical token `"None"`.

### 2. Advanced Feature Engineering
* **Boolean Indicator:** Created a binary `HasCoupon` column (1 for used, 0 for None) to explicitly map promotional conversions.
* **Temporal Expansion:** Deconstructed the `Date` column into continuous categorical vectors (`Year`, `Month`, `DayOfWeek`, `Quarter`) to capture cyclical purchasing trends.

### 3. Outlier Isolation & Capping
Non-parametric Interquartile Range (IQR) boundaries were calculated to identify extreme anomalies:
* `Quantity`: 0 outliers detected.
* `UnitPrice`: 0 outliers detected.
* `TotalPrice`: 8 structural outliers identified.
* **Action:** Applied a vectorized bounding function (`numpy.clip()`) to cap the upper extremes of `TotalPrice` to prevent optimization slope distortion without losing transaction data.

### 4. Collinearity Eradication (Data Leakage)
* **Issue:** Pearson correlation matrices revealed a perfect 1.00 correlation between `TotalPrice` and variables like `Quantity` and `UnitPrice`. 
* **Action:** `TotalPrice` and logistical tracking columns were systematically dropped prior to modeling to prevent data leakage and artificial validation success.

---

## 📊 Exploratory Data Analysis (EDA) Highlights

Visual distribution diagnostics were run to verify the balance of the dataset.

* **Product Volume Balance:** The 7 product lines are evenly distributed. Printers led with 181 orders, while Phones recorded 156.
* **Revenue Generation:** Despite balanced order counts, Chairs and Printers dominated gross revenue (~$196k each), driven by their unit pricing.
* **Payment & Referral:** Checkout methods were highly diversified, with Online payments capturing a slight lead (21.5%). Instagram led the referral traffic (259 orders).
* **Pricing Density:** The `TotalPrice` distribution was identified as heavily right-skewed, requiring median tracking for accurate central parameter estimates.

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Statistical Analysis & ML:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 🚀 How to Run

1.  Clone this repository to your local machine.
2.  Navigate to the `Week_1_EDA_Feature_Engineering` directory.
3.  Ensure dependencies are installed: `pip install -r requirements.txt`
4.  Run the master script to execute the preprocessing and generate visual artifacts:
    ```bash
    python run_week1.py
    ```
5.  View the generated charts in the `/charts` directory and the cleaned dataset in the `/outputs` directory.
