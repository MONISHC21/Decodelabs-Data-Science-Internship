# Week 3 — Customer Segmentation

This folder contains a complete customer segmentation pipeline suitable for an
internship portfolio. The pipeline performs data loading, cleaning, scaling,
PCA, elbow & silhouette analysis, K-Means clustering, persona generation, and
exports artifacts (models, charts, and CSVs) to the `models/`, `charts/`, and
`outputs/` folders.

Usage
-----
- Place your dataset under `Week_3_Customer_Segmentation/data/` (supported: CSV, XLSX).
- Open and run `notebooks/customer_segmentation.ipynb` from top to bottom.
- Alternatively, import functions from `src/` and use them in custom scripts.

Outputs
-------
- Charts: `charts/` (elbow, silhouette, PCA scatter, cluster distribution, personas)
- Models: `models/kmeans_model.pkl`
- Outputs: `outputs/cleaned_customers.csv`, `outputs/segmented_customers.csv`, `outputs/customer_personas.csv`

