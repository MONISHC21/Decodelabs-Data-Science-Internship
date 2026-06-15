"""Data cleaning utilities for Week 1.

Functions:
- load_data: read the ecommerce Excel file
- validate_df: basic shape/columns/dtypes checks
- missing_value_report: returns counts and percentages
- clean_orders: runs full cleaning pipeline and returns cleaned DataFrame
"""
from typing import List, Tuple
import os
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from an Excel or CSV file.

    Args:
        path: filepath to load

    Returns:
        DataFrame
    """
    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    return pd.read_csv(path)


def validate_df(df: pd.DataFrame) -> dict:
    """Return basic validation info: shape, columns, dtypes."""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict()
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing counts and percentages for each column."""
    total = len(df)
    miss = df.isnull().sum()
    pct = (miss / total * 100).round(2)
    report = pd.DataFrame({'missing_count': miss, 'missing_pct': pct})
    return report.sort_values('missing_count', ascending=False)


def drop_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """Drop duplicate rows. Returns (df, num_dropped)."""
    before = len(df)
    df2 = df.drop_duplicates(subset=subset)
    return df2, before - len(df2)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with reasonable defaults.

    - Drop rows missing both CustomerID and OrderID
    - Fill numeric NaNs with 0 where appropriate
    - Forward-fill or fillna for categorical fields
    """
    df = df.copy()
    if 'CustomerID' in df.columns and 'OrderID' in df.columns:
        df = df.dropna(subset=['CustomerID', 'OrderID'], how='all')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Fill numeric columns with median
    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)

    # Fill categorical with 'Unknown'
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')

    return df


def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove or fix clearly invalid values (e.g., negative quantities or prices)."""
    df = df.copy()
    if 'Quantity' in df.columns:
        df = df[df['Quantity'].apply(lambda x: pd.notnull(x) and (x >= 0))]
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'].apply(lambda x: pd.notnull(x) and (x >= 0))]
    # Recompute TotalPrice if possible
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Return boolean Series where True indicates an outlier using IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)


def save_cleaned(df: pd.DataFrame, out_path: str) -> None:
    """Save cleaned DataFrame to CSV, creating parent folder if needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return cleaned DataFrame.

    Steps:
    - Basic validation
    - Handle missing values
    - Remove duplicates
    - Remove invalid values
    - Add or recalc `TotalPrice`
    """
    df = df.copy()

    # Ensure columns expected exist
    expected = ['OrderID', 'CustomerID', 'OrderDate', 'Product', 'Quantity', 'UnitPrice', 'PaymentMethod', 'Referral', 'CouponCode']
    # No strict enforcement, but log missing in validation if needed

    # Handle missing and obvious problems
    df = handle_missing(df)

    # Drop duplicates using OrderID if available
    if 'OrderID' in df.columns:
        df, dropped = drop_duplicates(df, subset=['OrderID'])

    # Remove invalid values and recalc total
    df = remove_invalid_values(df)

    # Ensure TotalPrice exists
    if 'TotalPrice' not in df.columns and {'Quantity', 'UnitPrice'}.issubset(df.columns):
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Convert OrderDate to datetime if present
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

    return df
