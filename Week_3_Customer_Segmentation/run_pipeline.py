"""End-to-end runner for Week 3 customer segmentation.

This script uses the `src` helpers to load the dataset, clean it, scale
features, run PCA, evaluate cluster counts, train KMeans, and save outputs.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib

import sys
# Ensure the Week_3_Customer_Segmentation/src folder is importable as `src`
base = Path(__file__).resolve().parent
sys.path.insert(0, str(base))
sys.path.insert(0, str(base / 'src'))
from src.pca_analysis import perform_pca
from src.clustering import elbow_and_silhouette, run_kmeans
from src.visualization import save_personas_table


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / 'data'
    charts_dir = base / 'charts'
    models_dir = base / 'models'
    outputs_dir = base / 'outputs'
    for d in (charts_dir, models_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Prefer explicit file copied to data_dir
    candidates = list(data_dir.glob('*.csv')) + list(data_dir.glob('*.xlsx'))
    if not candidates:
        raise FileNotFoundError(f'No dataset found in {data_dir}')
    data_path = candidates[0]
    print('Using dataset:', data_path)
    if data_path.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print('Initial shape:', df.shape)

    # Cleaning
    df_clean = df.drop_duplicates().reset_index(drop=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError('No numeric columns detected')
    df_clean = df_clean.dropna(subset=numeric_cols)
    df_clean.to_csv(outputs_dir / 'cleaned_customers.csv', index=False)
    print('Saved cleaned dataset')

    # Feature matrix
    possible_id_cols = [c for c in df_clean.columns if 'id' in c.lower() or c.lower().startswith('customer')]
    X = df_clean.select_dtypes(include=[np.number]).copy()
    for cid in possible_id_cols:
        if cid in X.columns:
            X = X.drop(columns=[cid])
    print('Selected numeric features:', list(X.columns))

    # Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    joblib.dump(scaler, models_dir / 'scaler.pkl')

    # PCA
    pca_obj, X_pca = perform_pca(X_scaled, n_components=2, out_path=str(charts_dir / 'pca_variance_explained.png'))
    X_pca.to_csv(outputs_dir / 'customers_pca.csv', index=False)

    # Elbow & silhouette
    ks = list(range(1, 11))
    inertia_list, silhouette_list = elbow_and_silhouette(X_scaled, ks, out_dir=str(charts_dir))

    # choose k by silhouette
    import math
    sil_scores = [(k, s) for k, s in zip(ks, silhouette_list) if not (isinstance(s, float) and math.isnan(s))]
    chosen_k = max(sil_scores, key=lambda x: x[1])[0] if sil_scores else 3
    print('Chosen k:', chosen_k)

    # Run final KMeans
    model, labels = run_kmeans(X_scaled, chosen_k, out_dir=str(outputs_dir), models_dir=str(models_dir), original_df=df_clean)
    print('KMeans model saved')

    # Personas
    personas = df_clean.copy()
    personas['cluster'] = labels.values
    summary = personas.groupby('cluster').agg({col: 'mean' for col in X.columns})
    summary['count'] = personas.groupby('cluster').size()
    summary = summary.reset_index()
    summary.to_csv(outputs_dir / 'customer_personas.csv', index=False)
    save_personas_table(summary, str(charts_dir / 'customer_personas.png'))
    print('Customer personas saved')


if __name__ == '__main__':
    main()
