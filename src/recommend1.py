# src/recommend.py

import pandas as pd
from config import PROCESSED_DATA_PATH

# 1. Load predictions, inventory, and sales
preds             = pd.read_csv(f"{PROCESSED_DATA_PATH}perishable_demand_predictions.csv")
inventory_batches = pd.read_csv(f"{PROCESSED_DATA_PATH}inventory_batches.csv")
fact_sales        = pd.read_csv(f"{PROCESSED_DATA_PATH}fact_sales.csv")

fact_sales["sales_date"]           = pd.to_datetime(fact_sales["sales_date"])
inventory_batches["received_date"] = pd.to_datetime(inventory_batches["received_date"])
inventory_batches["expiry_date"]   = pd.to_datetime(inventory_batches["expiry_date"])

# =============================================================================
# 2. ON-HAND QUANTITY — 30-DAY ROLLING WINDOW
#
# WHAT CHANGED AND WHY
# -----------------------------------------------------------------------------
# OLD: used cumulative totals across ALL years (2013-2017).
#      Total ever received minus total ever sold = always deeply negative.
#
# FIX: compare only the last 30 days of receipts against the last 30 days
#      of sales. Same time window on both sides = a fair, meaningful number.
# =============================================================================

WINDOW_DAYS = 30
cutoff = fact_sales["sales_date"].max() - pd.Timedelta(days=WINDOW_DAYS)

# Total units SOLD in the last 30 days, per store + family
recent_sales = (
    fact_sales[fact_sales["sales_date"] >= cutoff]
    .groupby(["store_id", "family_id"])["sales"]
    .sum()
    .reset_index()
    .rename(columns={"sales": "recent_sales_30d"})
)

# Total units RECEIVED in the last 30 days, per store + family
recent_received = (
    inventory_batches[inventory_batches["received_date"] >= cutoff]
    .groupby(["store_id", "family_id"])["received_qty"]
    .sum()
    .reset_index()
    .rename(columns={"received_qty": "recent_received_30d"})
)

# on_hand = stock delivered this month - stock sold this month
inventory = recent_received.merge(
    recent_sales, on=["store_id", "family_id"], how="left"
)
inventory["on_hand_qty"]    = (
    inventory["recent_received_30d"] - inventory["recent_sales_30d"].fillna(0)
)
# Average daily sales over the 30-day window — used for days-of-supply calc
inventory["daily_avg_sales"] = inventory["recent_sales_30d"].fillna(0) / WINDOW_DAYS

# =============================================================================
# 3. AVERAGE PREDICTED DEMAND PER STORE + FAMILY
#    (unchanged from previous version — merge is already correct)
# =============================================================================

avg_preds = (
    preds.groupby(["store_id", "family_id"])["predicted_sales"]
         .mean()
         .reset_index()
         .rename(columns={"predicted_sales": "avg_predicted_sales"})
)

combined = inventory.merge(avg_preds, on=["store_id", "family_id"], how="left")
combined["avg_predicted_sales"] = combined["avg_predicted_sales"].fillna(0)

# =============================================================================
# 4. RULE-BASED RECOMMENDATION
#
# WHAT CHANGED AND WHY
# -----------------------------------------------------------------------------
# OLD: compared raw on_hand_qty units against daily predicted_sales × factor.
#      Incompatible units — one is a 30-day stock figure, the other is daily.
#
# FIX: convert on_hand_qty into DAYS OF SUPPLY:
#
#        days_of_supply = on_hand_qty / avg_daily_sales
#
#      "Days of supply" is the standard retail metric for inventory health.
#      It answers: "If no new stock arrives, how many days until we run out?"
#      It is unit-agnostic — it works the same whether you sell 100 or 10,000
#      units per day, because it normalises by the store's own sales rate.
#
# THRESHOLDS (in days):
#   URGENT_RESTOCK    → on_hand ≤ 0           (already out of stock)
#   PRIORITY_RESTOCK  → days_of_supply < 7    (less than one week of cover)
#   CONSIDER_MARKDOWN → days_of_supply > 30   (more than a month — waste risk)
#   OK                → 7 ≤ days ≤ 30         (healthy operating range)
#
# These thresholds reflect real retail practice:
#   <7 days  = danger zone for perishables (shelf life is 3-7 days)
#   >30 days = you ordered 4-10× what you need — mark it down before it expires
# =============================================================================

PRIORITY_THRESHOLD_DAYS  = 7    # below this → need stock urgently
MARKDOWN_THRESHOLD_DAYS  = 30   # above this → overstock, expiry risk

def decide_action(row):
    on_hand     = row["on_hand_qty"]
    daily_sales = row["daily_avg_sales"]

    # Step 1: truly out of stock
    if on_hand <= 0:
        return "URGENT_RESTOCK"

    # Step 2: guard against zero-sales edge case to avoid divide-by-zero
    if daily_sales <= 0:
        return "OK"

    # Step 3: convert stock to days of supply
    days_of_supply = on_hand / daily_sales

    # Step 4: apply thresholds
    if days_of_supply < PRIORITY_THRESHOLD_DAYS:
        return "PRIORITY_RESTOCK"
    if days_of_supply > MARKDOWN_THRESHOLD_DAYS:
        return "CONSIDER_MARKDOWN"
    return "OK"

combined["recommended_action"] = combined.apply(decide_action, axis=1)

# Rename for consistency with downstream consumers (visualize.py)
combined = combined.rename(columns={"avg_predicted_sales": "predicted_sales"})

combined.to_csv(f"{PROCESSED_DATA_PATH}perishable_recommendations.csv", index=False)

# Print a summary so you can confirm the distribution looks healthy
print("\n  Recommendation distribution:")
print(combined["recommended_action"].value_counts().to_string())
print(f"\n  Total rows: {len(combined)}")