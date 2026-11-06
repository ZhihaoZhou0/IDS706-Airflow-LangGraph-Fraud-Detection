"""LAB — identical to solution_baseline_prep.py; fill empty task bodies.

Copy each @task body + the wiring block from /guide → Airflow (or solution_baseline_prep.py).
Only differences vs the reference: this docstring, dag_id/tags, empty bodies, commented wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil

import pandas as pd
from airflow.decorators import dag, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Paths inside the Airflow containers (/repo is the mounted project root)
# ---------------------------------------------------------------------------
REPO = Path("/repo")
CUSTOMERS = REPO / "datasets" / "airflow" / "customers.csv"
TRANSACTIONS = REPO / "datasets" / "airflow" / "transactions.csv"
RUNS = REPO / "artifacts" / "airflow" / "runs"
PUBLISH = REPO / "datasets" / "airflow" / "merged_latest.csv"
TARGET = "is_fraud"
JOIN_KEY = "customer_id"
SUBSET_ROWS = 10000

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
}


# @dag turns the function below into an Airflow DAG (a named pipeline).
# - dag_id: unique name you see in the Airflow UI
# - start_date: earliest logical date Airflow will consider (required)
# - schedule=None: manual Trigger only (no cron) — perfect for class
# - catchup=False: do not backfill old dates when the DAG is unpaused
# - tags: filters in the UI
# - default_args: retries=2 → failed tasks retry twice (5 sec apart)
#
# Calling lab_baseline_prep() at the bottom REGISTERS this DAG
# with the scheduler. Without that call, Airflow never sees it.
@dag(
    dag_id="lab_baseline_prep",
    description="Scheduled ETL pipeline with PostgreSQL storage and ML analysis",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["student", "baseline", "prep", "classroom"],
)
def lab_baseline_prep():
    """Define tasks, then wire them. Airflow builds the Graph from the wiring."""

    # @task turns a normal Python function into one node on the Graph.
    # - Airflow runs each task in its own worker process
    # - Whatever you return is stored in XCom (Airflow's small data bus)
    # - Downstream tasks receive that return value as a normal argument
    # - If the function raises, the task fails (red) in the UI
    @task
    def validate_customers():
        """Check the customers CSV exists and has the join key."""
        if not CUSTOMERS.is_file():
            raise FileNotFoundError(CUSTOMERS)
        # nrows=0 → headers only (fast)
        cols = list(pd.read_csv(CUSTOMERS, nrows=0).columns)
        if JOIN_KEY not in cols:
            raise ValueError(f"missing {JOIN_KEY}")
        # Returned dict becomes XCom → next task's left_meta
        return {"path": str(CUSTOMERS)}

    @task
    def validate_transactions():
        """Check the transactions CSV exists and has join key + target label."""
        if not TRANSACTIONS.is_file():
            raise FileNotFoundError(TRANSACTIONS)
        cols = list(pd.read_csv(TRANSACTIONS, nrows=0).columns)
        for need in (JOIN_KEY, TARGET):
            if need not in cols:
                raise ValueError(f"missing {need}")
        return {"path": str(TRANSACTIONS)}

    @task
    def clean_customers(meta: dict, **context):
        """Clean customer records and preserve data quality metrics."""
        run_id = context["ts_nodash"]
        out = RUNS / run_id / "customers_clean.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(meta["path"])
        original_rows = len(df)

        df = df.dropna(subset=[JOIN_KEY])
        df = df.drop_duplicates(subset=[JOIN_KEY])

        df["loan_status"] = df["loan_status"].fillna("Unknown")

        dates = pd.to_datetime(df["customer_since"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid customer_since dates")
        df["customer_since"] = dates.dt.strftime("%Y-%m-%d")

        df.to_csv(out, index=False)

        return {
            "path": str(out),
            "rows_before": original_rows,
            "rows_after": len(df),
            "removed_rows": original_rows - len(df),
        }

    @task
    def clean_transactions(meta: dict, **context):
        """Clean transactions and validate fraud labels."""
        run_id = context["ts_nodash"]
        out = RUNS / run_id / "transactions_clean.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(meta["path"])
        original_rows = len(df)

        df = df.dropna(subset=["transaction_id", JOIN_KEY, TARGET])
        df = df.drop_duplicates(subset=["transaction_id"])

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError("Invalid fraud labels")

        dates = pd.to_datetime(df["timestamp"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid transaction timestamps")
        df["timestamp"] = dates.dt.strftime("%Y-%m-%d %H:%M:%S")

        df.to_csv(out, index=False)

        return {
            "path": str(out),
            "rows_before": original_rows,
            "rows_after": len(df),
            "removed_rows": original_rows - len(df),
        }

    @task
    def merge_csvs(left_meta: dict, right_meta: dict, **context):
        """Join transactions ⟕ customers. One output row per transaction.

        left_meta / right_meta come from upstream @task return values (XCom).
        **context is Airflow runtime info — we use ts_nodash as a unique run folder.
        """
        run_id = context["ts_nodash"]  # e.g. 20261104T145012
        out = RUNS / run_id / "merged.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        left = pd.read_csv(left_meta["path"])
        right = pd.read_csv(right_meta["path"])
        # fact table on the left of the merge → keep one row per transaction
        merged = right.merge(left, on=JOIN_KEY, how="left", suffixes=("", "_customer"))
        merged.to_csv(out, index=False)
        return {"merged_path": str(out), "n_merged": len(merged)}

    @task
    def load_to_postgres(merge_meta: dict):
        """Load the complete cleaned, merged dataset into PostgreSQL."""
        df = pd.read_csv(merge_meta["merged_path"])

        # Keep a complete snapshot of the latest cleaned dataset.
        # JSONB preserves numeric, categorical and date fields.
        records = [
            (
                str(row["transaction_id"]),
                str(row[JOIN_KEY]),
                int(row[TARGET]),
                row.to_json(date_format="iso"),
            )
            for _, row in df.iterrows()
        ]

        conn = psycopg2.connect(
            host="analytics-db",
            dbname="fraud_analytics",
            user="analytics",
            password="analytics_dev",
        )

        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS cleaned_transactions (
                            transaction_id TEXT PRIMARY KEY,
                            customer_id TEXT NOT NULL,
                            is_fraud INTEGER NOT NULL,
                            features JSONB NOT NULL
                        )
                    """)

                    # Reload the latest snapshot to avoid duplicate records
                    # when Airflow retries or reruns the DAG.
                    cur.execute("TRUNCATE TABLE cleaned_transactions")

                    execute_values(
                        cur,
                        """
                        INSERT INTO cleaned_transactions
                        (transaction_id, customer_id, is_fraud, features)
                        VALUES %s
                        """,
                        records,
                        template="(%s, %s, %s, %s::jsonb)",
                        page_size=1000,
                    )
        finally:
            conn.close()

        return {"table": "cleaned_transactions", "rows_loaded": len(df)}

    @task
    def read_from_postgres(load_meta: dict, **context):
        """Read cleaned data from PostgreSQL for downstream analysis."""
        run_id = context["ts_nodash"]
        out = RUNS / run_id / "database_cleaned.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        conn = psycopg2.connect(
            host="analytics-db",
            dbname="fraud_analytics",
            user="analytics",
            password="analytics_dev",
        )

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT features
                    FROM cleaned_transactions
                    ORDER BY transaction_id
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            raise ValueError("No cleaned transactions found in PostgreSQL")

        df = pd.DataFrame([row[0] for row in rows])

        if len(df) != load_meta["rows_loaded"]:
            raise ValueError(
                f"Row count mismatch: expected {load_meta['rows_loaded']}, "
                f"got {len(df)}"
            )

        df.to_csv(out, index=False)

        return {
            "merged_path": str(out),
            "n_rows": len(df),
            "source": "postgresql",
        }

    @task
    def make_subset(merge_meta: dict, **context):
        """Downsample so baseline + the later lab/solution-loop demo stay fast in class."""
        run_id = context["ts_nodash"]
        out = RUNS / run_id / "subset.csv"
        df = pd.read_csv(merge_meta["merged_path"])
        n = min(SUBSET_ROWS, len(df))
        subset = df.sample(n=n, random_state=42) if n < len(df) else df
        subset.to_csv(out, index=False)
        # Keep key name merged_path so later tasks stay uniform
        return {"merged_path": str(out), "n_subset": len(subset)}

    @task
    def profile_light(subset_meta: dict):
        """List columns that look leaky / id-like so training can drop them."""
        df = pd.read_csv(subset_meta["merged_path"])
        drops = [
            c
            for c in df.columns
            if c != TARGET
            and (c.endswith("_id") or c in {"timestamp", "fraud_probability", "fraud_history"})
        ]
        return {"recommended_drops": drops, "n_rows": len(df)}

    @task
    def train_baseline(subset_meta: dict, profile_meta: dict):
        """Tiny numeric-only LogisticRegression — a classroom A-baseline, not production ML."""
        df = pd.read_csv(subset_meta["merged_path"])
        drops = list(profile_meta.get("recommended_drops") or [])
        y = df[TARGET]
        X = (
            df.drop(columns=[TARGET, *drops], errors="ignore")
            .select_dtypes("number")
            .fillna(0)
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        return {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

    @task
    def publish_merged(subset_meta: dict, baseline: dict):
        """Copy the subset to a stable path the lab/solution-loop UI can pick up."""
        PUBLISH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(subset_meta["merged_path"], PUBLISH)
        return {
            "published_latest": str(PUBLISH),
            "merged_dataset_path": subset_meta["merged_path"],
            "baseline_val_f1": baseline["f1"],
            "target_column": TARGET,
        }

    # -----------------------------------------------------------------------
    # WIRING = the dependency graph
    #
    # Calling a @task function here does NOT run the Python immediately.
    # It creates a node and an edge: "this task needs that task's XCom output".
    # Airflow's Graph tab is literally this block drawn as boxes + arrows.
    #
    #   validate_customers ─┐
    #                       ├─→ merge_csvs → make_subset → profile_light ─┐
    #   validate_transactions ┘                    └──────────────────────┼─→ train → publish
    # -----------------------------------------------------------------------
    # TODO: paste wiring from /guide → Airflow (identical to solution_baseline_prep.py):
    left = clean_customers(validate_customers())
    right = clean_transactions(validate_transactions())

    merged = merge_csvs(left, right)

    loaded = load_to_postgres(merged)
    database_data = read_from_postgres(loaded)

    demo = make_subset(database_data)
    prof = profile_light(demo)
    base = train_baseline(demo, prof)

    publish_merged(demo, base)


# Actually register the DAG with Airflow (decorators alone are not enough).
lab_baseline_prep()
