"""Visualization utilities for Week 1.

Functions produce and save charts required by Phase 1.
"""
import os
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_product_distribution(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'Product' not in df.columns:
        print('Skipping product_distribution: `Product` column not found')
        return
    plt.figure(figsize=(10, 6))
    prod_counts = df['Product'].value_counts().nlargest(20)
    sns.barplot(x=prod_counts.values, y=prod_counts.index, palette='viridis')
    plt.xlabel('Number of Orders')
    plt.ylabel('Product')
    plt.title('Top Products by Order Count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def revenue_by_product(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'Product' not in df.columns or 'TotalPrice' not in df.columns:
        print('Skipping revenue_by_product: required columns missing')
        return
    plt.figure(figsize=(10, 6))
    rev = df.groupby('Product')['TotalPrice'].sum().sort_values(ascending=False).nlargest(20)
    sns.barplot(x=rev.values, y=rev.index, palette='magma')
    plt.xlabel('Revenue')
    plt.ylabel('Product')
    plt.title('Revenue by Product (Top 20)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def payment_methods(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'PaymentMethod' not in df.columns:
        print('Skipping payment_methods: `PaymentMethod` column not found')
        return
    plt.figure(figsize=(8, 6))
    pm = df['PaymentMethod'].value_counts()
    sns.barplot(x=pm.index, y=pm.values, palette='pastel')
    plt.ylabel('Count')
    plt.xlabel('Payment Method')
    plt.title('Payment Methods Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def referral_sources(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'Referral' not in df.columns:
        print('Skipping referral_sources: `Referral` column not found')
        return
    plt.figure(figsize=(8, 6))
    ref = df['Referral'].value_counts()
    sns.barplot(x=ref.index, y=ref.values, palette='cool')
    plt.ylabel('Count')
    plt.xlabel('Referral Source')
    plt.title('Referral Sources')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def coupon_usage(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'HasCoupon' not in df.columns:
        # try to infer from CouponCode, treating 'Unknown' as no coupon
        if 'CouponCode' in df.columns:
            def _has_coupon_val(x):
                if pd.isna(x):
                    return False
                s = str(x).strip()
                if s == '':
                    return False
                if s.lower() in {'unknown', 'none', 'nan'}:
                    return False
                return True
            df['HasCoupon'] = df['CouponCode'].apply(_has_coupon_val)
        else:
            print('Skipping coupon_usage: no CouponCode or HasCoupon column')
            return
    plt.figure(figsize=(6, 6))
    has_coupon = df['HasCoupon'].value_counts()
    sns.barplot(x=has_coupon.index.astype(str), y=has_coupon.values, palette='Set2')
    plt.xlabel('Has Coupon')
    plt.ylabel('Count')
    plt.title('Coupon Usage')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def total_price_distribution(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if 'TotalPrice' not in df.columns:
        print('Skipping total_price_distribution: `TotalPrice` column not found')
        return
    plt.figure(figsize=(8, 6))
    sns.histplot(df['TotalPrice'].dropna(), kde=True, bins=50)
    plt.xlabel('Total Price')
    plt.title('Total Price Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    num = df.select_dtypes(include=['number'])
    if num.shape[1] < 2:
        print('Skipping correlation_heatmap: not enough numeric columns')
        return
    plt.figure(figsize=(8, 6))
    corr = num.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def outlier_boxplot(df: pd.DataFrame, column: str, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if column not in df.columns:
        print(f'Skipping outlier plot for {column}: column not found')
        return
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column].dropna())
    plt.title(f'Outliers - {column}')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_all_charts(df: pd.DataFrame, charts_dir: Optional[str] = 'charts') -> None:
    """Generate all required charts and save them to `charts_dir`."""
    os.makedirs(charts_dir, exist_ok=True)
    for fn, name in [
        (plot_product_distribution, 'product_distribution.png'),
        (revenue_by_product, 'revenue_by_product.png'),
        (payment_methods, 'payment_methods.png'),
        (referral_sources, 'referral_sources.png'),
        (coupon_usage, 'coupon_usage.png'),
        (total_price_distribution, 'total_price_distribution.png'),
        (correlation_heatmap, 'correlation_heatmap.png'),
    ]:
        try:
            fn(df, os.path.join(charts_dir, name))
        except Exception as e:
            print(f'Failed to generate {name}:', e)

    # Outlier plots
    for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
        try:
            out_path = os.path.join(charts_dir, f'outlier_{col.lower()}.png')
            outlier_boxplot(df, col, out_path)
        except Exception as e:
            print(f'Failed to generate outlier plot for {col}:', e)
