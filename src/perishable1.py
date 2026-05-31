# src/perishable.py

import numpy as np
import pandas as pd
from config import PROCESSED_DATA_PATH

# Fix random seed so results are reproducible — running the script twice
# gives identical inventory_batches.csv every time.
np.random.seed(42)

# 1. Load processed dim tables and sales
dim_family = pd.read_csv(f"{PROCESSED_DATA_PATH}dim_family.csv")
fact_sales = pd.read_csv(f"{PROCESSED_DATA_PATH}fact_sales.csv")
fact_sales["sales_date"] = pd.to_datetime(fact_sales["sales_date"])

# 2. Define perishable families & shelf life
perishable_config = pd.DataFrame([
    {"family_name": "DAIRY",   "is_perishable": True, "shelf_life_days": 7},
    {"family_name": "MEATS",   "is_perishable": True, "shelf_life_days": 5},
    {"family_name": "PRODUCE", "is_perishable": True, "shelf_life_days": 3},
])

perishable_cfg_joined = perishable_config.merge(
    dim_family, left_on="family_name", right_on="family", how="inner"
)[["family_id", "is_perishable", "shelf_life_days"]]

perishable_cfg_joined.to_csv(f"{PROCESSED_DATA_PATH}perishable_config.csv", index=False)

# 3. Filter sales to perishables only
perishable_ids = perishable_cfg_joined["family_id"].unique()
sales = fact_sales[fact_sales["family_id"].isin(perishable_ids)].copy()
sales = sales.sort_values(["store_id", "family_id", "sales_date"])

# 4. Create synthetic inventory batches
#
# =============================================================================
# WHAT CHANGED AND WHY
# =============================================================================
#
# OLD CODE (broken):
#   received_qty = int(window["sales"].mean() * 1.2)
#   shelf_life_days = 5  (hardcoded the same for every family)
#
# WHY IT BROKE:
#   Each batch was sized to cover only ONE day of demand (daily_sales × 1.2).
#   But deliveries arrive every TWO days (delivery_interval_days=2).
#   This means the store always receives LESS stock than it sells between
#   deliveries — like filling a bucket with a cup while it drains with a jug.
#   The result: on_hand_qty is permanently negative for every store × family,
#   so the rule engine fires URGENT_RESTOCK on all 162 rows.
#
# FIX 1 — Batch size covers the full delivery interval:
#   received_qty = daily_sales × delivery_interval_days × ordering_factor
#   This ensures each batch covers the demand UNTIL the next delivery arrives,
#   plus the ordering_factor buffer. This is how real retail replenishment works.
#
# FIX 2 — Per-store ordering factor from a realistic distribution:
#   Real stores don't all order the same amount. Some over-order (wasteful),
#   some under-order (stockout risk), most are in the middle. We model this
#   with np.random.uniform(0.6, 2.5) drawn once per store × family group.
#   This is what creates the natural spread of all four recommendation types.
#
# FIX 3 — Use the actual per-family shelf life:
#   OLD: shelf_life_days=5 hardcoded for DAIRY (7), MEATS (5), and PRODUCE (3)
#   NEW: looked up from perishable_cfg_joined for each family_id in the loop
# =============================================================================

def create_batches_for_group(
    group,
    shelf_life_days=5,
    delivery_interval_days=2,
    ordering_factor=1.2,       # ← NEW parameter: per-group ordering behaviour
):
    group = group.copy()
    batches = []
    current_date = group["sales_date"].min()
    last_date    = group["sales_date"].max()

    while current_date <= last_date:
        window = group[group["sales_date"] == current_date]
        if not window.empty:
            # FIX 1: multiply by delivery_interval_days so each batch covers
            # demand until the NEXT delivery, not just one single day.
            # FIX 2: multiply by ordering_factor (varies per store) so different
            # stores naturally end up over/under/correctly stocked.
            daily_sales  = window["sales"].mean()
            received_qty = max(0, int(daily_sales * delivery_interval_days * ordering_factor))

            expiry_date = current_date + pd.Timedelta(days=shelf_life_days)
            batches.append({
                "store_id":     window["store_id"].iloc[0],
                "family_id":    window["family_id"].iloc[0],
                "received_date": current_date,
                "received_qty": received_qty,
                "expiry_date":  expiry_date,
            })
        current_date += pd.Timedelta(days=delivery_interval_days)

    return pd.DataFrame(batches)


# Build a quick shelf_life lookup dict so the loop below can find the right
# shelf life for each family_id without repeatedly filtering the DataFrame.
shelf_life_lookup = dict(
    zip(perishable_cfg_joined["family_id"], perishable_cfg_joined["shelf_life_days"])
)

batch_frames = []
for (store_id, family_id), grp in sales.groupby(["store_id", "family_id"]):

    # FIX 3: use the actual shelf life for this product family.
    # DAIRY=7 days, MEATS=5 days, PRODUCE=3 days.
    # Fall back to 5 if somehow the family_id is not in the lookup.
    actual_shelf_life = shelf_life_lookup.get(family_id, 5)

    # FIX 2: draw a unique ordering factor for every store × family combination.
    # uniform(0.6, 2.5) means:
    #   ~21% of combos will have factor < 1.0  → under-ordered → URGENT/PRIORITY
    #   ~53% will have factor 1.0–2.0          → near-balanced  → PRIORITY/OK
    #   ~26% will have factor > 2.0            → over-ordered   → MARKDOWN
    ordering_factor = np.random.uniform(0.6, 2.5)

    batch_df = create_batches_for_group(
        grp,
        shelf_life_days=actual_shelf_life,
        delivery_interval_days=2,
        ordering_factor=ordering_factor,
    )
    batch_frames.append(batch_df)

inventory_batches = pd.concat(batch_frames, ignore_index=True)
inventory_batches.to_csv(f"{PROCESSED_DATA_PATH}inventory_batches.csv", index=False)

print(f"  inventory_batches.csv written — {len(inventory_batches):,} rows")