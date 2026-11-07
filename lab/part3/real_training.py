"""Real IEEE-CIS model training adapter for Part 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from loop_runtime import _fit_score


SUPPORTED_MODELS = {
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
}

DEFAULT_DATA_PATH = (
    "artifacts/ieee_fraud/agent_input.csv"
)


def train_real_model(
    model_family: str,
    *,
    data_path: str = DEFAULT_DATA_PATH,
    seed: int = 42,
    holdout_frac: float = 0.2,
    hyperparameters: dict[str, Any] | None = None,
    on_holdout: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a real model using the shared runtime."""

    if model_family not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model family: {model_family}"
        )

    path = Path(data_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if "isFraud" in df.columns:
        target = "isFraud"
    elif "is_fraud" in df.columns:
        target = "is_fraud"
    else:
        raise ValueError(
            "Missing fraud target column."
        )

    spec = {
        "model_family": model_family,
        "hyperparameters": hyperparameters or {},
        "drop_columns": [],
    }

    return _fit_score(
        df,
        target,
        spec,
        seed,
        holdout_frac,
        on_holdout=on_holdout,
    )
