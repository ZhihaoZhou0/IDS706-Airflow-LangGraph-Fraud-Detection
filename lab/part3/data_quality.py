"""Data quality checks for IEEE-CIS fraud experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_fraud_dataset(data_path: str) -> dict:
    """Validate dataset before model training."""

    path = Path(data_path)

    if not path.is_file():
        return {
            "passed": False,
            "error": f"Dataset not found: {path}",
        }

    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            "passed": False,
            "error": f"Cannot read dataset: {exc}",
        }

    target = (
        "isFraud" if "isFraud" in df.columns
        else "is_fraud" if "is_fraud" in df.columns
        else None
    )

    if target is None:
        return {
            "passed": False,
            "error": "Missing fraud target column",
        }

    counts = df[target].value_counts(dropna=False).to_dict()
    valid_labels = df[target].isin([0, 1]).all()

    if not valid_labels:
        return {
            "passed": False,
            "error": "Fraud labels must be 0 or 1",
        }

    if not all((df[target] == label).sum() >= 10 for label in (0, 1)):
        return {
            "passed": False,
            "error": "Insufficient samples for stratified splitting",
        }

    if df.drop(columns=[target]).shape[1] == 0:
        return {
            "passed": False,
            "error": "No feature columns available",
        }

    return {
        "passed": True,
        "rows": len(df),
        "columns": len(df.columns),
        "target": target,
        "class_distribution": counts,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
