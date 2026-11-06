
from pathlib import Path
import pandas as pd

DATA_DIR = Path("datasets/ieee_fraud")
OUTPUT_DIR = DATA_DIR / "sampled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# Read selected columns only, reducing memory consumption.
transaction_cols = [
    "TransactionID", "isFraud", "TransactionDT",
    "TransactionAmt", "ProductCD",
    "card1", "card4", "card6",
    "addr1", "dist1",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "D1", "D2",
]

identity_cols = [
    "TransactionID", "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16",
    "id_30", "id_31",
]

transactions = pd.read_csv(
    DATA_DIR / "train_transaction.csv",
    usecols=transaction_cols,
    low_memory=False,
)

# Stratified sample to preserve fraud distribution.
sampled_transactions = (
    transactions.groupby("isFraud", group_keys=False)
    .sample(
        frac=SAMPLE_SIZE / len(transactions),
        random_state=RANDOM_STATE,
    )
    .sort_values("TransactionID")
    .reset_index(drop=True)
)

identities = pd.read_csv(
    DATA_DIR / "train_identity.csv",
    usecols=identity_cols,
    low_memory=False,
)

# Keep only identity records associated with sampled transactions.
sampled_identities = identities[
    identities["TransactionID"].isin(
        sampled_transactions["TransactionID"]
    )
].copy()

sampled_transactions.to_csv(
    OUTPUT_DIR / "transactions_sample.csv",
    index=False,
)

sampled_identities.to_csv(
    OUTPUT_DIR / "identity_sample.csv",
    index=False,
)

print("=== SAMPLING RESULTS ===")
print(f"Transactions: {len(sampled_transactions):,}")
print(f"Identities: {len(sampled_identities):,}")

print("\nFraud distribution:")
print(sampled_transactions["isFraud"].value_counts())

print("\nFraud rate:")
print(f"{sampled_transactions['isFraud'].mean():.2%}")

print("\nIdentity match rate:")
print(f"{len(sampled_identities) / len(sampled_transactions):.2%}")

print("\nSaved to:", OUTPUT_DIR)
