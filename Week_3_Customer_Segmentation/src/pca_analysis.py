"""PCA analysis helpers.

This module performs PCA on a numeric feature matrix, returns the fitted
PCA transformer and a DataFrame with the requested number of components.
"""
from typing import Tuple
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def perform_pca(X: pd.DataFrame, n_components: int = 2, out_path: str = None) -> Tuple[PCA, pd.DataFrame]:
    """Fit PCA on `X`, save explained variance plot if requested, and
    return the fitted PCA object and transformed DataFrame.

    Args:
        X: Numeric feature DataFrame.
        n_components: Number of principal components to keep.
        out_path: If provided, save the variance-explained plot to this path.

    Returns:
        pca: Fitted PCA object.
        X_pca: DataFrame with columns ['PC1', 'PC2', ...].
    """
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X)
    cols = [f'PC{i+1}' for i in range(components.shape[1])]
    X_pca = pd.DataFrame(components, columns=cols, index=X.index)

    if out_path:
        explained = pca.explained_variance_ratio_
        cum = explained.cumsum()
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, len(explained) + 1), explained, alpha=0.6, label='Individual')
        plt.step(range(1, len(cum) + 1), cum, where='mid', label='Cumulative')
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance ratio')
        plt.title('PCA explained variance')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    return pca, X_pca

