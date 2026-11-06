"""Run independent IEEE-CIS fraud graph against cleaned PostgreSQL records."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2

from lab.langgraph.ieee_fraud_graph import FEATURE_CANDIDATES, build_graph

ROOT = Path('/repo') if Path('/repo').exists() else Path(__file__).resolve().parent
DATA_PATH = ROOT / 'artifacts/ieee_fraud/agent_input.csv'
RESULT_PATH = ROOT / 'artifacts/ieee_fraud/agent_results.json'


def export_postgres_data():
    conn = psycopg2.connect(
        host=os.getenv('ANALYTICS_DB_HOST', 'analytics-db'),
        dbname=os.getenv('ANALYTICS_DB_NAME', 'fraud_analytics'),
        user=os.getenv('ANALYTICS_DB_USER', 'analytics'),
        password=os.getenv('ANALYTICS_DB_PASSWORD', 'analytics_dev'),
    )
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM ieee_fraud_transactions LIMIT 1')
            columns = [d[0] for d in cur.description]
            json_col = next((c for c in ('features', 'payload', 'data') if c in columns), None)
            if json_col is None:
                raise ValueError(f'No JSONB features column found; actual columns: {columns}')
            cur.execute(f'SELECT {json_col} FROM ieee_fraud_transactions')
            rows = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    target = 'isFraud' if 'isFraud' in df else 'is_fraud'
    if target not in df:
        raise ValueError(f'Fraud target missing from features JSONB. Columns: {list(df.columns)}')
    available = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not available:
        raise ValueError(f'No recognized features. Actual columns: {list(df.columns)}')
    df = df[[target, *available]]
    # Stratified cap reduces RAM pressure without changing the holdout policy.
    if len(df) > 12000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=12000, random_state=42, stratify=df[target])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f'PostgreSQL export: {len(df)} rows, {len(available)} features, fraud rate {df[target].mean():.2%}')
    return DATA_PATH


def main():
    path = export_postgres_data()
    app = build_graph()
    state = app.invoke({'data_path': str(path), 'max_cycles': 3, 'target_f1': 0.3,
                        'seed': 42, 'output_path': str(RESULT_PATH), 'steps': []},
                       config={'recursion_limit': 25})
    print('\nIEEE-CIS AGENT EXPERIMENTS')
    print(f"{'Experiment':<16} {'Model':<26} {'F1':>9} {'Accuracy':>10}")
    for r in state['history']:
        print(f"{r['experiment_id']:<16} {r['model_family']:<26} {r['f1']:>9.4f} {r['accuracy']:>10.4f}")
    print('Best model:', state['best_spec']['model_family'])
    print('Best validation F1:', round(state['best_val_f1'], 4))
    print('Holdout metrics:', state['holdout_metrics'])
    print('Stop reason:', state['stop_reason'])
    print('Results saved:', RESULT_PATH)
    mermaid = ROOT / 'artifacts/ieee_fraud/ieee_agent_graph.mmd'
    mermaid.write_text(app.get_graph().draw_mermaid(), encoding='utf-8')
    print('Graph saved:', mermaid)


if __name__ == '__main__':
    main()
