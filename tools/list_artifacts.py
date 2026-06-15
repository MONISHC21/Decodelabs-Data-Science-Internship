from pathlib import Path

for label, path in [
    ('Week 1 outputs', 'Week_1_EDA_Feature_Engineering/outputs'),
    ('Week 1 charts', 'Week_1_EDA_Feature_Engineering/charts'),
    ('Week 2 models', 'Week_2_Fraud_Detection/models'),
    ('Week 2 charts', 'Week_2_Fraud_Detection/charts'),
    ('Week 2 outputs', 'Week_2_Fraud_Detection/outputs'),
    ('Week 3 models', 'Week_3_Customer_Segmentation/models'),
    ('Week 3 charts', 'Week_3_Customer_Segmentation/charts'),
    ('Week 3 outputs', 'Week_3_Customer_Segmentation/outputs'),
]:
    p = Path(path)
    print('---', label)
    if not p.exists():
        print('  (none)')
        continue
    files = sorted([f for f in p.iterdir() if f.is_file()])
    if not files:
        print('  (none)')
        continue
    for f in files:
        print(f'  {f.name} - {f.stat().st_size/1024:.2f} KB')
