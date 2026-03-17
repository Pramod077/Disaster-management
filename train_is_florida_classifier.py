r"""Train a classifier to predict whether a storm event is in Florida.

This script:
 - Loads the original CSV in the `machine learning` folder
 - Creates a binary label `is_florida` using a simple bbox
 - Trains a RandomForestClassifier on simple features
 - Writes the original rows plus `is_florida`, `is_florida_pred`, and `is_florida_proba` to the requested Excel output path

Usage (PowerShell):
 python .\train_is_florida_classifier.pyclear

The output file is written to:
 C:\Users\PramodPandit\Box\C ED Pramod Pandit\AIML\docker\StormEvents_locations-ftp_v1.0_d2024_c20250401.xlsx
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Paths (adjust if you keep a different layout)
CSV_PATH = Path(r"C:\Users\PramodPandit\Box\C ED Pramod Pandit\AIML\docker\StormEvents_locations-ftp_v1.0_d2024_c20250401.csv")
OUT_XLSX = Path(r"C:\Users\PramodPandit\Box\C ED Pramod Pandit\AIML\docker\StormEvents_locations-ftp_v1.0_d2024_c20250401.xlsx")

# Florida bbox (same as repository scripts)
LAT_MIN = 24.5
LAT_MAX = 31.1
LON_MIN = -87.63
LON_MAX = -80.0

def load_data(csv_path: Path) -> pd.DataFrame:
    print('Loading CSV:', csv_path)
    df = pd.read_csv(csv_path)
    return df


def make_label(df: pd.DataFrame) -> pd.Series:
    lat = df.get('LATITUDE')
    lon = df.get('LONGITUDE')
    if lat is None or lon is None:
        raise ValueError('Expected LATITUDE and LONGITUDE columns in CSV')
    is_fl = ((lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)).astype(int)
    return is_fl


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use a small set of sensible features available in the CSV
    feats = pd.DataFrame()
    # Numeric features
    for c in ['LATITUDE', 'LONGITUDE', 'RANGE', 'LOCATION_INDEX', 'YEARMONTH']:
        if c in df.columns:
            feats[c] = pd.to_numeric(df[c], errors='coerce')
    # Categorical: AZIMUTH (compass directions) - one-hot encode top values, group others to 'OTHER'
    if 'AZIMUTH' in df.columns:
        az = df['AZIMUTH'].fillna('')
        top = az.value_counts().nlargest(12).index.tolist()
        az_cat = az.where(az.isin(top), other='OTHER')
        dummies = pd.get_dummies(az_cat, prefix='AZ')
        feats = pd.concat([feats, dummies], axis=1)
    # Fill missing numeric
    for col in feats.columns:
        if feats[col].dtype.kind in 'biufc':
            feats[col] = feats[col].fillna(feats[col].median())
        else:
            feats[col] = feats[col].fillna(0)
    return feats


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    print('\nClassification report (test set):')
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print('ROC AUC:', auc)
        except Exception:
            pass
    return clf


def main():
    df = load_data(CSV_PATH)
    print('Rows loaded:', len(df))
    df['is_florida'] = make_label(df)
    print('Florida positive rows:', int(df['is_florida'].sum()))

    X = prepare_features(df)
    print('Feature columns used:', list(X.columns))

    if X.empty:
        raise RuntimeError('No features were prepared — check CSV columns')

    model = train_and_evaluate(X, df['is_florida'])

    # Predict on full dataset and save
    full_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(X))
    full_pred = model.predict(X)
    df['is_florida_pred'] = full_pred
    df['is_florida_proba'] = full_proba

    # Ensure output dir exists
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    print('Writing results to Excel:', OUT_XLSX)
    # Write a single sheet with predictions
    df.to_excel(OUT_XLSX, index=False)
    print('Done.')

if __name__ == '__main__':
    main()
