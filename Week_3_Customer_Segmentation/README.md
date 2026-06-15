# Week 3: Customer Segmentation using K-Means Clustering

## Overview

This project focuses on Customer Segmentation using Unsupervised Machine Learning techniques applied to the Decode Labs E-Commerce Orders Dataset.

The objective is to identify distinct customer groups based on purchasing behavior, spending patterns, order frequency, cart activity, coupon usage, and referral sources.

By applying Principal Component Analysis (PCA) and K-Means Clustering, hidden customer segments were discovered and translated into meaningful business personas that can support marketing and customer retention strategies.

---

## Project Objective

The primary goals of this project are:

* Discover hidden customer groups without predefined labels.
* Reduce dataset dimensionality using PCA.
* Identify the optimal number of clusters using the Elbow Method.
* Evaluate cluster quality using Silhouette Score.
* Build customer personas from clustering results.
* Generate business recommendations based on customer behavior.

---

## Dataset Information

Dataset: Decode Labs E-Commerce Orders Dataset

### Dataset Summary

* Records: 1,200 Orders
* Features: 14 Columns
* Domain: E-Commerce
* Type: Unsupervised Learning

### Important Features Used

* Quantity
* UnitPrice
* TotalPrice
* ItemsInCart
* CouponCode
* PaymentMethod
* ReferralSource
* Product

---

## Methodology

### Step 1 – Data Cleaning

Performed:

* Missing value handling
* Duplicate removal
* Data validation

### Step 2 – Feature Engineering

Created additional analytical features:

* HasCoupon
* Order Value Categories
* Average Order Value
* Customer Spending Patterns

### Step 3 – Feature Scaling

Applied StandardScaler to normalize feature ranges.

### Step 4 – Principal Component Analysis (PCA)

Reduced dimensionality for visualization and clustering.

Benefits:

* Reduced noise
* Improved visualization
* Faster clustering performance

### Step 5 – Elbow Method

Determined optimal cluster count by evaluating Within Cluster Sum of Squares (WCSS).

### Step 6 – Silhouette Analysis

Measured cluster separation and quality.

### Step 7 – K-Means Clustering

Generated customer segments and assigned cluster labels.

---

## Generated Visualizations

Located in:

```text
charts/
```

Files:

```text
elbow_method.png
silhouette_score.png
pca_visualization.png
cluster_distribution.png
customer_personas.png
```

---

## Customer Personas

### VIP Customers

Characteristics:

* High order value
* High spending behavior
* Frequent purchases

Business Strategy:

* Loyalty programs
* Premium offers
* Exclusive rewards

---

### Regular Customers

Characteristics:

* Consistent purchasing patterns
* Moderate spending

Business Strategy:

* Cross-selling
* Personalized recommendations

---

### Budget Customers

Characteristics:

* Low spending
* Price-sensitive

Business Strategy:

* Discount campaigns
* Coupon-based promotions

---

### Occasional Customers

Characteristics:

* Infrequent purchases
* Low engagement

Business Strategy:

* Re-engagement campaigns
* Email marketing

---

## Model Outputs

### Models

Located in:

```text
models/
```

Files:

```text
kmeans_model.pkl
```

### Outputs

Located in:

```text
outputs/
```

Files:

```text
segmented_customers.csv
customer_personas.csv
```

---

## Key Findings

* Customer spending patterns vary significantly.
* PCA successfully reduced dimensionality while preserving information.
* K-Means effectively identified meaningful customer groups.
* A small number of high-value customers contribute disproportionately to revenue.
* Coupon usage and cart size strongly influence customer behavior.

---

## Business Recommendations

### Short-Term

* Target VIP customers with loyalty rewards.
* Offer discounts to budget customers.

### Medium-Term

* Implement personalized product recommendations.
* Improve customer engagement campaigns.

### Long-Term

* Build customer lifetime value models.
* Develop predictive customer segmentation systems.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* PCA
* K-Means Clustering
* Joblib

---

## Repository Structure

```text
Week_3_Customer_Segmentation/

├── data/
├── notebooks/
│   └── customer_segmentation.ipynb
├── charts/
├── models/
├── outputs/
├── reports/
├── src/
└── README.md
```

---

## Conclusion

This project demonstrates the application of Unsupervised Learning techniques to real-world E-Commerce customer behavior data. By combining PCA, K-Means Clustering, and business intelligence analysis, meaningful customer personas were created that can help improve marketing effectiveness, customer retention, and overall business performance.


