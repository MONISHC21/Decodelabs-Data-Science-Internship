"""Run Week 2 fraud detection pipeline end-to-end.

Usage: run from repository root via the workspace Python executable.
"""
import os
import sys
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
data_path = base / 'data' / 'creditcard.csv'
charts_dir = base / 'charts'
models_dir = base / 'models'
outputs_dir = base / 'outputs'
for d in [charts_dir, models_dir, outputs_dir]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(base))
from src.preprocessing import preprocess, prepare_train_test, scale_features
from src.model_training import train_models, predict_models
from src.evaluation import save_class_distribution, evaluate_and_report

def main():
    print('Loading dataset from', data_path)
    df = pd.read_csv(data_path)
    print('Initial shape:', df.shape)

    # Preprocess
    df_clean = preprocess(df)
    print('After preprocess shape:', df_clean.shape)

    # Subsample if dataset too large to train in CI/timebox
    # Allow overriding via env var MAX_ROWS. Set to "None" (or empty) to disable subsampling.
    max_rows_env = os.getenv('MAX_ROWS', '300000')
    if str(max_rows_env).lower() in ('none', 'null', ''):
        max_rows = None
    else:
        try:
            max_rows = int(max_rows_env)
        except Exception:
            max_rows = 300000

    if max_rows is not None and df_clean.shape[0] > max_rows:
        print(f'Dataset has {df_clean.shape[0]} rows — subsampling to {max_rows} for faster training')
        # Stratified sampling to preserve class distribution when target exists
        if 'isFraud' in df_clean.columns:
            total = len(df_clean)
            counts = df_clean['isFraud'].value_counts().to_dict()
            samples = []
            for cls, cnt in counts.items():
                n = int(round(cnt / total * max_rows))
                n = max(1, n)
                sampled = df_clean[df_clean['isFraud'] == cls].sample(n=n, random_state=42)
                samples.append(sampled)
            df_clean = pd.concat(samples, axis=0)
            df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_clean = df_clean.sample(n=max_rows, random_state=42)

    # Debug: show structure before train/test split
    print('After subsample type:', type(df_clean))
    if hasattr(df_clean, 'columns'):
        print('After subsample columns:', list(df_clean.columns)[:50])
    else:
        print('After subsample has no columns attribute')

    # Prepare train/test
    X_train, X_test, y_train, y_test = prepare_train_test(df_clean, target='isFraud')
    print('Train/test sizes:', X_train.shape, X_test.shape)

    # Save original class distribution
    save_class_distribution(y_train, str(charts_dir / 'class_distribution.png'))

    # Apply scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Apply SMOTE to training set
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
        # save smote distribution
        save_class_distribution(y_res, str(charts_dir / 'smote_distribution.png'))
    except Exception as e:
        print('SMOTE not available or failed:', e)
        X_res, y_res = X_train_scaled, y_train

    # Train models
    models = train_models(X_res, y_res, models_dir=str(models_dir))
    print('Models trained and saved to', models_dir)

    # Evaluate
    results = evaluate_and_report(models, X_test_scaled, y_test, out_dir=str(charts_dir))
    print('Evaluation complete')

    # Save predictions from best model (choose random_forest)
    best = models.get('random_forest') or list(models.values())[0]
    preds = best.predict(X_test_scaled)
    out = X_test.copy()
    out['isFraud_pred'] = preds
    out['isFraud_true'] = y_test.values
    out.to_csv(outputs_dir / 'fraud_predictions.csv', index=False)
    print('Predictions saved to', outputs_dir / 'fraud_predictions.csv')

if __name__ == '__main__':
    main()
