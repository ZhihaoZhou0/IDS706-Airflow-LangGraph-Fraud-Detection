"""IEEE-CIS fraud detection: two-source scheduled ETL and database-backed analysis."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from airflow.decorators import dag, task
from psycopg2.extras import execute_values
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path('/repo')
SOURCE = ROOT / 'datasets' / 'ieee_fraud' / 'sampled'
OUTPUT = ROOT / 'artifacts' / 'ieee_fraud'
TX = SOURCE / 'transactions_sample.csv'
IDENT = SOURCE / 'identity_sample.csv'
KEY = 'TransactionID'
LABEL = 'isFraud'


def db_connect():
    return psycopg2.connect(
        host=os.getenv('ANALYTICS_DB_HOST', 'analytics-db'),
        dbname=os.getenv('ANALYTICS_DB_NAME', 'fraud_analytics'),
        user=os.getenv('ANALYTICS_DB_USER', 'analytics'),
        password=os.getenv('ANALYTICS_DB_PASSWORD', 'analytics_dev'),
    )


@dag(
    dag_id='ieee_fraud_pipeline',
    description='IEEE-CIS independent-source ETL, Postgres storage and fraud analysis',
    start_date=datetime(2026, 11, 1),
    schedule='@daily',
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(seconds=15)},
    max_active_runs=1,
    tags=['ids706', 'ieee-cis', 'fraud', 'custom'],
)
def ieee_fraud_pipeline():
    @task
    def ingest_transactions():
        if not TX.is_file():
            raise FileNotFoundError(TX)
        columns = set(pd.read_csv(TX, nrows=0).columns)
        required = {KEY, LABEL, 'TransactionAmt'}
        if not required <= columns:
            raise ValueError(f'Missing transaction columns: {required - columns}')
        return str(TX)

    @task
    def ingest_identity():
        if not IDENT.is_file():
            raise FileNotFoundError(IDENT)
        columns = set(pd.read_csv(IDENT, nrows=0).columns)
        required = {KEY, 'DeviceType'}
        if not required <= columns:
            raise ValueError(f'Missing identity columns: {required - columns}')
        return str(IDENT)

    @task
    def clean_transactions(path: str, **context):
        out = OUTPUT / context['ts_nodash'] / 'transactions_clean.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(path, low_memory=False)
        before = len(df)
        df[KEY] = pd.to_numeric(df[KEY], errors='coerce')
        df[LABEL] = pd.to_numeric(df[LABEL], errors='coerce')
        df['TransactionAmt'] = pd.to_numeric(df['TransactionAmt'], errors='coerce')
        df = df.dropna(subset=[KEY, LABEL, 'TransactionAmt'])
        df = df[df[LABEL].isin([0, 1]) & (df['TransactionAmt'] >= 0)]
        df = df.drop_duplicates(subset=[KEY]).copy()
        df[KEY] = df[KEY].astype('int64')
        df[LABEL] = df[LABEL].astype('int64')
        df.to_csv(out, index=False)
        return {'path': str(out), 'before': before, 'after': len(df)}

    @task
    def clean_identity(path: str, **context):
        out = OUTPUT / context['ts_nodash'] / 'identity_clean.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(path, low_memory=False)
        before = len(df)
        df[KEY] = pd.to_numeric(df[KEY], errors='coerce')
        df = df.dropna(subset=[KEY]).drop_duplicates(subset=[KEY]).copy()
        df[KEY] = df[KEY].astype('int64')
        for col in ('DeviceType', 'DeviceInfo', 'id_12', 'id_15', 'id_16', 'id_30', 'id_31'):
            if col in df:
                df[col] = df[col].fillna('Unknown').astype(str)
        df.to_csv(out, index=False)
        return {'path': str(out), 'before': before, 'after': len(df)}

    @task
    def merge_sources(tx_meta: dict, id_meta: dict, **context):
        out = OUTPUT / context['ts_nodash'] / 'merged.csv'
        tx = pd.read_csv(tx_meta['path'], low_memory=False)
        ident = pd.read_csv(id_meta['path'], low_memory=False)
        tx[KEY] = tx[KEY].astype('int64')
        ident[KEY] = ident[KEY].astype('int64')
        ident['has_identity'] = 1
        merged = tx.merge(ident, on=KEY, how='left', validate='one_to_one')
        merged['has_identity'] = merged['has_identity'].fillna(0).astype('int64')
        if len(merged) != len(tx):
            raise ValueError('LEFT JOIN changed the transaction row count')
        merged.to_csv(out, index=False)
        return {'path': str(out), 'rows': len(merged), 'identity_matches': int(merged['has_identity'].sum())}

    @task
    def engineer_features(meta: dict, **context):
        out = OUTPUT / context['ts_nodash'] / 'features.csv'
        df = pd.read_csv(meta['path'], low_memory=False)
        df['log_transaction_amt'] = df['TransactionAmt'].clip(lower=0).map(lambda x: __import__('math').log1p(x))
        df['transaction_day'] = (pd.to_numeric(df['TransactionDT'], errors='coerce') // 86400).fillna(-1).astype(int)
        # Missingness is informative; keep the original feature nulls in the database.
        df.to_csv(out, index=False)
        return {'path': str(out), 'rows': len(df)}

    @task
    def load_postgres(meta: dict):
        df = pd.read_csv(meta['path'], low_memory=False)
        records = []
        for row in df.to_dict(orient='records'):
            cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            records.append((int(cleaned[KEY]), int(cleaned[LABEL]), float(cleaned['TransactionAmt']), int(cleaned['has_identity']), json.dumps(cleaned, default=str)))
        conn = db_connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute('''CREATE TABLE IF NOT EXISTS ieee_fraud_transactions (
                        transaction_id BIGINT PRIMARY KEY,
                        is_fraud SMALLINT NOT NULL,
                        transaction_amt DOUBLE PRECISION NOT NULL,
                        has_identity SMALLINT NOT NULL,
                        features JSONB NOT NULL
                    )''')
                    cur.execute('TRUNCATE TABLE ieee_fraud_transactions')
                    execute_values(cur, '''INSERT INTO ieee_fraud_transactions
                        (transaction_id, is_fraud, transaction_amt, has_identity, features)
                        VALUES %s''', records, template='(%s,%s,%s,%s,%s::jsonb)', page_size=1000)
        finally:
            conn.close()
        return {'rows': len(records), 'table': 'ieee_fraud_transactions'}

    @task
    def read_postgres(meta: dict, **context):
        out = OUTPUT / context['ts_nodash'] / 'database_cleaned.csv'
        conn = db_connect()
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT features FROM ieee_fraud_transactions ORDER BY transaction_id')
                rows = cur.fetchall()
        finally:
            conn.close()
        if len(rows) != meta['rows']:
            raise ValueError(f'DB row mismatch: {len(rows)} vs {meta["rows"]}')
        pd.DataFrame([r[0] for r in rows]).to_csv(out, index=False)
        return {'path': str(out), 'rows': len(rows)}

    @task
    def analyze_fraud(meta: dict, **context):
        df = pd.read_csv(meta['path'], low_memory=False)
        y = df[LABEL].astype(int)
        cols = ['TransactionAmt', 'log_transaction_amt', 'transaction_day', 'has_identity',
                'ProductCD', 'card4', 'card6', 'DeviceType', 'id_12', 'id_15']
        cols = [c for c in cols if c in df.columns]
        X = df[cols].copy()
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in cols if c not in numeric]
        transformers = []
        if numeric:
            transformers.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric))
        if categorical:
            transformers.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical))
        pipe = Pipeline([('preprocess', ColumnTransformer(transformers)),
                         ('classifier', LogisticRegression(max_iter=500, class_weight='balanced'))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results = {
            'source': 'PostgreSQL', 'rows': len(df), 'fraud_rate': round(float(y.mean()), 5),
            'identity_match_rate': round(float(df['has_identity'].mean()), 5),
            'train_rows': len(X_train), 'test_rows': len(X_test),
            'accuracy': round(float(accuracy_score(y_test, pred)), 5),
            'precision': round(float(precision_score(y_test, pred, zero_division=0)), 5),
            'recall': round(float(recall_score(y_test, pred, zero_division=0)), 5),
            'f1': round(float(f1_score(y_test, pred, zero_division=0)), 5),
            'model': 'LogisticRegression(class_weight=balanced)',
        }
        out = OUTPUT / context['ts_nodash'] / 'fraud_metrics.json'
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')
        return {'metrics_path': str(out), 'metrics': results}

    @task
    def publish_metrics(meta: dict):
        latest = OUTPUT / 'latest_metrics.json'
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(meta['metrics'], indent=2), encoding='utf-8')
        print(json.dumps(meta['metrics'], indent=2))
        return str(latest)

    tx_clean = clean_transactions(ingest_transactions())
    id_clean = clean_identity(ingest_identity())
    merged = merge_sources(tx_clean, id_clean)
    engineered = engineer_features(merged)
    loaded = load_postgres(engineered)
    from_db = read_postgres(loaded)
    analyzed = analyze_fraud(from_db)
    publish_metrics(analyzed)


ieee_fraud_pipeline()
