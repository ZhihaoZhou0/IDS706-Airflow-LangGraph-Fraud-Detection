
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

DATA_DIR = Path("datasets/ieee_fraud")
OUTPUT_DIR = DATA_DIR / "sampled"

TRANSACTION_COLS = [
    "TransactionID", "isFraud", "TransactionDT",
    "TransactionAmt", "ProductCD",
    "card1", "card4", "card6",
    "addr1", "dist1",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "D1", "D2",
]

IDENTITY_COLS = [
    "TransactionID", "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16",
    "id_30", "id_31",
]


def sample_transactions(path, sample_size, seed, chunksize):
    # First pass: count records by fraud label.
    counts = {0: 0, 1: 0}
    for chunk in pd.read_csv(
        path, usecols=["isFraud"], chunksize=chunksize
    ):
        for label, count in chunk["isFraud"].value_counts().items():
            counts[int(label)] += int(count)

    total = sum(counts.values())
    if sample_size > total:
        raise ValueError(
            f"Requested {sample_size} rows, but only {total} exist."
        )

    # Allocate exact sample sizes across labels.
    fraud_target = round(sample_size * counts[1] / total)
    targets = {
        1: fraud_target,
        0: sample_size - fraud_target,
    }

    rng = np.random.default_rng(seed)
    selected = {}

    for label in (0, 1):
        selected[label] = np.sort(
            rng.choice(
                counts[label],
                size=targets[label],
                replace=False,
            )
        )

    # Second pass: retrieve only selected rows.
    offsets = {0: 0, 1: 0}
    positions = {0: 0, 1: 0}
    samples = []

    for chunk in pd.read_csv(
        path,
        usecols=TRANSACTION_COLS,
        chunksize=chunksize,
    ):
        for label in (0, 1):
            group = chunk[chunk["isFraud"] == label]
            start = offsets[label]
            end = start + len(group)

            indices = selected[label]
            left = positions[label]
            right = np.searchsorted(indices, end, side="left")
            local_indices = indices[left:right] - start

            if len(local_indices):
                samples.append(group.iloc[local_indices].copy())

            positions[label] = right
            offsets[label] = end

    return (
        pd.concat(samples, ignore_index=True)
        .sort_values("TransactionID")
        .reset_index(drop=True)
    )


def sample_identities(path, transaction_ids, chunksize):
    samples = []

    for chunk in pd.read_csv(
        path,
        usecols=IDENTITY_COLS,
        chunksize=chunksize,
    ):
        matched = chunk[
            chunk["TransactionID"].isin(transaction_ids)
        ]
        if not matched.empty:
            samples.append(matched.copy())

    if not samples:
        return pd.DataFrame(columns=IDENTITY_COLS)

    return pd.concat(samples, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient IEEE-CIS stratified sampling"
    )
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=10000)
    args = parser.parse_args()

    if args.sample_size <= 0 or args.chunksize <= 0:
        parser.error("sample-size and chunksize must be positive")

    transactions_path = DATA_DIR / "train_transaction.csv"
    identities_path = DATA_DIR / "train_identity.csv"

    for path in (transactions_path, identities_path):
        if not path.is_file():
            parser.error(f"Missing dataset: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transactions = sample_transactions(
        transactions_path,
        args.sample_size,
        args.seed,
        args.chunksize,
    )

    identities = sample_identities(
        identities_path,
        set(transactions["TransactionID"]),
        args.chunksize,
    )

    transactions.to_csv(
        OUTPUT_DIR / "transactions_sample.csv", index=False
    )
    identities.to_csv(
        OUTPUT_DIR / "identity_sample.csv", index=False
    )

    print("=== SAMPLING RESULTS ===")
    print(f"Transactions: {len(transactions):,}")
    print(f"Identities: {len(identities):,}")
    print(f"Fraud rate: {transactions['isFraud'].mean():.2%}")
    print(f"Identity match rate: {len(identities) / len(transactions):.2%}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
