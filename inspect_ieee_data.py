
import pandas as pd
from pathlib import Path

DATA_DIR = Path("datasets/ieee_fraud")

transaction_path = DATA_DIR / "train_transaction.csv"
identity_path = DATA_DIR / "train_identity.csv"

# Read only the columns needed for initial inspection
transactions = pd.read_csv(
    transaction_path,
    usecols=["TransactionID", "isFraud", "TransactionAmt", "ProductCD"],
)

identities = pd.read_csv(
    identity_path,
    usecols=["TransactionID", "DeviceType", "DeviceInfo"],
)

print("=== DATASET INSPECTION ===")

print(f"Transactions: {len(transactions):,} rows")
print(f"Identities: {len(identities):,} rows")

print(f"Transaction ID duplicates: {transactions['TransactionID'].duplicated().sum()}")
print(f"Identity ID duplicates: {identities['TransactionID'].duplicated().sum()}")

matched = transactions["TransactionID"].isin(identities["TransactionID"])
print(f"Matched transactions: {matched.sum():,}")
print(f"Match rate: {matched.mean():.2%}")

print("\nFraud distribution:")
print(transactions["isFraud"].value_counts())

print("\nIdentity missing values:")
print(identities.isna().sum())

print("\n=== SAMPLE DATA ===")
print(transactions.head(3))
print(identities.head(3))
