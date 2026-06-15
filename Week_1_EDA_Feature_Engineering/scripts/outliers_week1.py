import pandas as pd
from pathlib import Path
import sys
import json

# ensure src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_cleaning import detect_outliers_iqr

p = Path(__file__).resolve().parents[1] / 'outputs' / 'cleaned_dataset.csv'
df = pd.read_csv(p)

outliers = {}
for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
    if col in df.columns:
        mask = detect_outliers_iqr(df, col)
        outliers[col] = int(mask.sum())

print(json.dumps(outliers))
