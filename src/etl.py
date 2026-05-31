# src/etl.py

import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# 1. Load raw Kaggle CSVs
train = pd.read_csv(f"{RAW_DATA_PATH}train.csv")
stores = pd.read_csv(f"{RAW_DATA_PATH}stores.csv")
transactions = pd.read_csv(f"{RAW_DATA_PATH}transactions.csv")

# 2. Fix date types
train["date"] = pd.to_datetime(train["date"])
transactions["date"] = pd.to_datetime(transactions["date"])

# 3. Build dim tables
dim_store = stores.rename(columns={"store_nbr": "store_id"})
dim_store.to_csv(f"{PROCESSED_DATA_PATH}dim_store.csv", index=False)

dim_family = (
    train[["family"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
dim_family["family_id"] = dim_family.index + 1
dim_family.to_csv(f"{PROCESSED_DATA_PATH}dim_family.csv", index=False)

# 4. Map families → IDs
train = train.merge(dim_family, on="family", how="left")

# 5. Build fact tables
fact_sales = train.rename(
    columns={
        "date": "sales_date",
        "store_nbr": "store_id"
    }
)[["sales_date", "store_id", "family_id", "sales", "onpromotion"]]
fact_sales.to_csv(f"{PROCESSED_DATA_PATH}fact_sales.csv", index=False)

fact_transactions = transactions.rename(
    columns={
        "date": "trans_date",
        "store_nbr": "store_id"
    }
)[["trans_date", "store_id", "transactions"]]
fact_transactions.to_csv(f"{PROCESSED_DATA_PATH}fact_transactions.csv", index=False)