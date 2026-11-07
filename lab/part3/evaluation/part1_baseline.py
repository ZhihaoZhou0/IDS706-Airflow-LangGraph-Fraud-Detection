"""Controlled reproduction of the Part 1 fixed Logistic Regression task.

Uses the shared 12,000-row dataset and sealed 20% holdout.
Does not run the Airflow DAG or access PostgreSQL.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("artifacts/ieee_fraud/agent_input.csv")
OUTPUT_PATH = Path("lab/part3/results/part1_baseline.json")

# Available original Part 1 DAG features, without recreating unavailable
# engineered features from the original ETL pipeline.
PART1_FEATURES = [
    "TransactionAmt",
    "has_identity",
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
]


def main():
    df = pd.read_csv(DATA_PATH)
    target = "isFraud" if "isFraud" in df.columns else "is_fraud"
    features = [col for col in PART1_FEATURES if col in df.columns]

    if not features:
        raise ValueError("No Part 1 features available")

    X = df[features]
    y = df[target].astype(int)

    numeric = [
        col for col in features
        if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical = [col for col in features if col not in numeric]

    transformers = []

    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    pipeline = Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
        )),
    ])

    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_hold)

    metrics = {
        "f1": float(f1_score(y_hold, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_hold, predictions)),
        "precision": float(precision_score(y_hold, predictions, zero_division=0)),
        "recall": float(recall_score(y_hold, predictions, zero_division=0)),
        "n_holdout": int(len(y_hold)),
    }

    result = {
        "system": "part1_fixed_airflow_task_reproduction",
        "model_family": "logistic_regression",
        "dataset": str(DATA_PATH),
        "rows": int(len(df)),
        "features": features,
        "seed": 42,
        "holdout_frac": 0.2,
        "holdout_metrics": metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=== PART 1 CONTROLLED BASELINE ===")
    print("Model: Logistic Regression")
    print("Features:", features)
    print("Holdout metrics:", metrics)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
