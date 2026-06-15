"""Runner for Week 1 EDA pipeline.

Usage: run this module from the `Week_1_EDA_Feature_Engineering/src` directory or via Python.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'ecommerce_orders.xlsx'
CHARTS_DIR = ROOT / 'charts'
OUTPUTS_DIR = ROOT / 'outputs'
OUTPUT_CSV = OUTPUTS_DIR / 'cleaned_dataset.csv'

sys.path.append(str(ROOT))

from src.data_cleaning import load_data, clean_orders, save_cleaned
from src.feature_engineering import create_features
from src.visualization import generate_all_charts


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f'Loading data from {DATA_PATH}')
    df = load_data(str(DATA_PATH))
    print('Loaded', df.shape)

    print('Cleaning data...')
    cleaned = clean_orders(df)
    print('After cleaning', cleaned.shape)

    print('Creating features...')
    fe = create_features(cleaned)
    print('After features', fe.shape)

    print('Generating charts...')
    generate_all_charts(fe, charts_dir=str(CHARTS_DIR))
    print('Charts saved to', CHARTS_DIR)

    print('Saving cleaned dataset...')
    save_cleaned(fe, str(OUTPUT_CSV))
    print('Cleaned dataset saved to', OUTPUT_CSV)


if __name__ == '__main__':
    main()
