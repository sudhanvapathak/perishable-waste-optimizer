# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from config import PROCESSED_DATA_PATH

# 1. Load processed tables
fact_sales        = pd.read_csv(f"{PROCESSED_DATA_PATH}fact_sales.csv")
fact_transactions = pd.read_csv(f"{PROCESSED_DATA_PATH}fact_transactions.csv")
perishable_config = pd.read_csv(f"{PROCESSED_DATA_PATH}perishable_config.csv")

fact_sales["sales_date"]        = pd.to_datetime(fact_sales["sales_date"])
fact_transactions["trans_date"] = pd.to_datetime(fact_transactions["trans_date"])

# 2. Join sales + transactions + perishable config
df = fact_sales.merge(
    fact_transactions,
    left_on=["sales_date", "store_id"],
    right_on=["trans_date", "store_id"],
    how="left"
).merge(
    perishable_config,
    on="family_id",
    how="left"
)

df = df[df["is_perishable"] == True].copy()
df = df.sort_values(["store_id", "family_id", "sales_date"])

# 3. Lag & rolling features
df["lag_1_sales"] = df.groupby(["store_id", "family_id"])["sales"].shift(1)
df["lag_7_sales"] = df.groupby(["store_id", "family_id"])["sales"].shift(7)
df["rolling_7d_mean"] = df.groupby(["store_id", "family_id"])["sales"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

df = df.dropna(subset=["lag_1_sales", "lag_7_sales"])

# 4. Train / test split
#
# FIX: we split the full df index so that after splitting we can look up
# store_id, family_id, and sales_date for every test row.  Previously only
# the 5 feature columns were kept, so those identifiers were lost and the
# recommendations downstream had no way to join back to inventory.
#
feature_cols = ["transactions", "shelf_life_days", "lag_1_sales", "lag_7_sales", "rolling_7d_mean"]
X = df[feature_cols]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Save predictions  — carry identifiers alongside features
#
# X_test still has the original df row indices, so we can use .loc to pull
# store_id, family_id, and sales_date for exactly those rows.
#
df_test = X_test.copy()
df_test["store_id"]        = df.loc[X_test.index, "store_id"]
df_test["family_id"]       = df.loc[X_test.index, "family_id"]
df_test["sales_date"]      = df.loc[X_test.index, "sales_date"]
df_test["actual_sales"]    = y_test
df_test["predicted_sales"] = model.predict(X_test)

df_test.to_csv(f"{PROCESSED_DATA_PATH}perishable_demand_predictions.csv", index=False)