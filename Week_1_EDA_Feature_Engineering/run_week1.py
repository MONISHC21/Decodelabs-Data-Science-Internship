"""Runner to execute Week 1 pipeline: load, clean, features, charts, save.

Run from repository root or this folder. Designed for automated execution in CI or locally.
"""
from pathlib import Path
import os
import sys


def main():
    base = Path(__file__).resolve().parent
    data_path = base / 'data' / 'ecommerce_orders.xlsx'
    charts_dir = base / 'charts'
    outputs_dir = base / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Import local modules
    sys.path.append(str(base))
    try:
        from src.data_cleaning import load_data, validate_df, missing_value_report, clean_orders, save_cleaned
        from src.feature_engineering import create_features
        from src.visualization import generate_all_charts
    except Exception as e:
        print('Failed to import local modules:', e)
        raise

    print('Loading data from', data_path)
    df = load_data(str(data_path))
    print('Initial shape:', df.shape)

    val = validate_df(df)
    print('Columns:', len(val['columns']))

    mreport = missing_value_report(df)
    mreport.to_csv(outputs_dir / 'missing_value_report.csv')
    print('Missing value report saved')

    cleaned = clean_orders(df)
    print('Cleaned shape:', cleaned.shape)

    fe = create_features(cleaned)
    print('Feature-engineered shape:', fe.shape)

    # generate charts
    generate_all_charts(fe, charts_dir=str(charts_dir))
    print('Charts generated in', charts_dir)

    # save cleaned
    save_cleaned(fe, str(outputs_dir / 'cleaned_dataset.csv'))
    print('Cleaned dataset saved to', outputs_dir / 'cleaned_dataset.csv')


if __name__ == '__main__':
    main()
