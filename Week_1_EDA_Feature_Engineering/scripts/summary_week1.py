import pandas as pd
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'outputs' / 'cleaned_dataset.csv'
df = pd.read_csv(p)

summary = {
    'shape': df.shape,
    'n_customers': int(df['CustomerID'].nunique()),
    'total_revenue': float(df['TotalPrice'].sum()),
    'avg_order_value': float(df['AverageOrderValue'].mean()),
    'top_products': df['Product'].value_counts().head(5).to_dict(),
    'payment_methods': df['PaymentMethod'].value_counts().to_dict(),
    'coupon_pct': float(df['HasCoupon'].mean()*100) if 'HasCoupon' in df.columns else None
}

print(json.dumps(summary))
