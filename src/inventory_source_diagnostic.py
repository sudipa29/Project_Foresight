import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PROJECT FORESIGHT
# INVENTORY SOURCE DATA DIAGNOSTIC
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - INVENTORY SOURCE DATA DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INVENTORY_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "inventory_clean.csv"
)

DEMAND_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "daily_demand.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "inventory_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading inventory data...")

inventory = pd.read_csv(INVENTORY_PATH)

print("Inventory shape:", inventory.shape)

print("\nLoading daily demand data...")

demand = pd.read_csv(DEMAND_PATH)

print("Daily demand shape:", demand.shape)


# ------------------------------------------------------------
# BASIC STRUCTURE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INVENTORY COLUMNS")
print("=" * 70)

print(inventory.columns.tolist())

print("\nFirst 5 inventory rows:")
print(inventory.head())


# ------------------------------------------------------------
# DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(inventory.dtypes)


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing = inventory.isna().sum()

print(missing)

print("\nTotal missing values:", missing.sum())


# ------------------------------------------------------------
# DUPLICATE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_rows = inventory.duplicated().sum()

duplicate_keys = inventory.duplicated(
    subset=["store_id", "sku_id"]
).sum()

print("Duplicate complete rows:", duplicate_rows)

print("Duplicate store-SKU combinations:", duplicate_keys)


# ------------------------------------------------------------
# UNIQUE COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE ENTITY COUNTS")
print("=" * 70)

print("Unique stores:", inventory["store_id"].nunique())

print("Unique SKUs:", inventory["sku_id"].nunique())

print(
    "Unique store-SKU combinations:",
    inventory[["store_id", "sku_id"]].drop_duplicates().shape[0]
)


# ------------------------------------------------------------
# INVENTORY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INVENTORY STATISTICS")
print("=" * 70)

inventory_cols = [
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

print(
    inventory[inventory_cols].describe()
)


# ------------------------------------------------------------
# RELATIONSHIP CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INVENTORY BUSINESS RELATIONSHIPS")
print("=" * 70)

print(
    "Stock <= reorder point:",
    (inventory["stock_on_hand"] <= inventory["reorder_point"]).sum()
)

print(
    "Stock <= safety stock:",
    (inventory["stock_on_hand"] <= inventory["safety_stock"]).sum()
)

print(
    "Reorder point <= safety stock:",
    (
        inventory["reorder_point"]
        <= inventory["safety_stock"]
    ).sum()
)

print(
    "Safety stock <= reorder point:",
    (
        inventory["safety_stock"]
        <= inventory["reorder_point"]
    ).sum()
)


# ------------------------------------------------------------
# INVENTORY RATIOS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INVENTORY RATIOS")
print("=" * 70)

inventory["stock_to_reorder_ratio"] = (
    inventory["stock_on_hand"]
    / inventory["reorder_point"].replace(0, np.nan)
)

inventory["stock_to_safety_ratio"] = (
    inventory["stock_on_hand"]
    / inventory["safety_stock"].replace(0, np.nan)
)

inventory["reorder_to_stock_ratio"] = (
    inventory["reorder_point"]
    / inventory["stock_on_hand"].replace(0, np.nan)
)

print(
    inventory[
        [
            "stock_to_reorder_ratio",
            "stock_to_safety_ratio",
            "reorder_to_stock_ratio"
        ]
    ].describe()
)


# ------------------------------------------------------------
# STOCK VS REORDER DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STOCK VS REORDER DISTRIBUTION")
print("=" * 70)

inventory["stock_status"] = np.select(
    [
        inventory["stock_on_hand"] <= inventory["safety_stock"],
        inventory["stock_on_hand"] <= inventory["reorder_point"],
        inventory["stock_on_hand"]
        <= inventory["reorder_point"] * 1.25,
        inventory["stock_on_hand"]
        <= inventory["reorder_point"] * 1.50
    ],
    [
        "Below Safety Stock",
        "Below Reorder Point",
        "Near Reorder Point",
        "Moderately Above Reorder Point"
    ],
    default="Well Above Reorder Point"
)

print(
    inventory["stock_status"]
    .value_counts()
)


# ------------------------------------------------------------
# DEMAND AGGREGATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEMAND ANALYSIS")
print("=" * 70)

demand["date"] = pd.to_datetime(demand["date"])

latest_date = demand["date"].max()

print("Latest demand date:", latest_date)

period_30_start = latest_date - pd.Timedelta(days=29)

recent_demand = demand[
    demand["date"] >= period_30_start
]

demand_30 = (
    recent_demand
    .groupby(["store_id", "sku_id"], as_index=False)
    .agg(
        units_30d=("units_sold", "sum"),
        avg_daily_demand=("units_sold", "mean")
    )
)

print(
    "30-day demand combinations:",
    len(demand_30)
)


# ------------------------------------------------------------
# MERGE INVENTORY + DEMAND
# ------------------------------------------------------------

analysis = inventory.merge(
    demand_30,
    on=["store_id", "sku_id"],
    how="left"
)

analysis["units_30d"] = (
    analysis["units_30d"]
    .fillna(0)
)

analysis["avg_daily_demand"] = (
    analysis["avg_daily_demand"]
    .fillna(0)
)


# ------------------------------------------------------------
# DEMAND VS INVENTORY
# ------------------------------------------------------------

analysis["days_of_inventory"] = np.where(
    analysis["avg_daily_demand"] > 0,
    analysis["stock_on_hand"]
    / analysis["avg_daily_demand"],
    np.inf
)

analysis["demand_to_reorder_ratio"] = np.where(
    analysis["reorder_point"] > 0,
    analysis["units_30d"]
    / analysis["reorder_point"],
    np.nan
)

analysis["demand_to_stock_ratio"] = np.where(
    analysis["stock_on_hand"] > 0,
    analysis["units_30d"]
    / analysis["stock_on_hand"],
    np.nan
)


# ------------------------------------------------------------
# DEMAND VS INVENTORY SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEMAND VS INVENTORY SUMMARY")
print("=" * 70)

print(
    analysis[
        [
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "units_30d",
            "avg_daily_demand",
            "days_of_inventory",
            "demand_to_reorder_ratio",
            "demand_to_stock_ratio"
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).describe()
)


# ------------------------------------------------------------
# DEMAND COVERAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEMAND COVERAGE CHECK")
print("=" * 70)

positive_demand = analysis[
    analysis["avg_daily_demand"] > 0
].copy()

print(
    "Items with positive demand:",
    len(positive_demand)
)

print(
    "Items with zero demand:",
    len(analysis) - len(positive_demand)
)

if len(positive_demand) > 0:

    print(
        "\nDays of inventory statistics:"
    )

    print(
        positive_demand["days_of_inventory"]
        .describe()
    )


# ------------------------------------------------------------
# LOWEST STOCK COVERAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("20 ITEMS WITH LOWEST DAYS OF INVENTORY")
print("=" * 70)

if len(positive_demand) > 0:

    lowest_coverage = (
        positive_demand
        .sort_values("days_of_inventory")
        [
            [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "reorder_point",
                "safety_stock",
                "units_30d",
                "avg_daily_demand",
                "days_of_inventory"
            ]
        ]
        .head(20)
    )

    print(lowest_coverage.to_string(index=False))


# ------------------------------------------------------------
# HIGHEST DEMAND ITEMS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("20 HIGHEST DEMAND ITEMS")
print("=" * 70)

top_demand = (
    analysis
    .sort_values(
        "units_30d",
        ascending=False
    )
    [
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "units_30d",
            "avg_daily_demand",
            "days_of_inventory"
        ]
    ]
    .head(20)
)

print(top_demand.to_string(index=False))


# ------------------------------------------------------------
# STORES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STORE-LEVEL INVENTORY SUMMARY")
print("=" * 70)

store_summary = (
    analysis
    .groupby("store_id")
    .agg(
        inventory_items=("sku_id", "count"),
        total_stock=("stock_on_hand", "sum"),
        total_reorder_point=("reorder_point", "sum"),
        total_safety_stock=("safety_stock", "sum"),
        total_30d_demand=("units_30d", "sum")
    )
    .reset_index()
)

store_summary["stock_to_reorder_ratio"] = (
    store_summary["total_stock"]
    / store_summary["total_reorder_point"]
)

print(
    store_summary
    .sort_values(
        "stock_to_reorder_ratio"
    )
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# SKU LEVEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SKU-LEVEL INVENTORY SUMMARY")
print("=" * 70)

sku_summary = (
    analysis
    .groupby("sku_id")
    .agg(
        store_count=("store_id", "nunique"),
        total_stock=("stock_on_hand", "sum"),
        total_reorder_point=("reorder_point", "sum"),
        total_safety_stock=("safety_stock", "sum"),
        total_30d_demand=("units_30d", "sum")
    )
    .reset_index()
)

sku_summary["stock_to_reorder_ratio"] = (
    sku_summary["total_stock"]
    / sku_summary["total_reorder_point"]
)

print(
    sku_summary
    .sort_values(
        "stock_to_reorder_ratio"
    )
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# FINAL DIAGNOSTIC FLAGS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC FLAGS")
print("=" * 70)

flags = []

if (
    inventory["stock_on_hand"]
    <= inventory["reorder_point"]
).sum() == 0:

    flags.append(
        "No inventory items are below or equal to reorder point."
    )

if (
    inventory["stock_on_hand"]
    <= inventory["safety_stock"]
).sum() == 0:

    flags.append(
        "No inventory items are below or equal to safety stock."
    )

if (
    inventory["stock_on_hand"]
    / inventory["reorder_point"]
).median() > 2:

    flags.append(
        "Median stock is more than 2x reorder point."
    )

if (
    positive_demand["days_of_inventory"].median()
    if len(positive_demand) > 0
    else 0
) > 90:

    flags.append(
        "Median inventory coverage exceeds 90 days."
    )

if (
    analysis["units_30d"] == 0
).mean() > 0.30:

    flags.append(
        "More than 30% of inventory items had zero demand in the last 30 days."
    )


if flags:

    for i, flag in enumerate(flags, 1):
        print(f"{i}. WARNING: {flag}")

else:

    print(
        "No major source inventory anomalies detected."
    )


# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "inventory_source_diagnostic.csv"
)

analysis.to_csv(
    output_path,
    index=False
)

store_output = (
    OUTPUT_DIR
    / "inventory_store_diagnostic.csv"
)

store_summary.to_csv(
    store_output,
    index=False
)

sku_output = (
    OUTPUT_DIR
    / "inventory_sku_diagnostic.csv"
)

sku_summary.to_csv(
    sku_output,
    index=False
)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INVENTORY SOURCE DATA DIAGNOSTIC COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(output_path)

print(store_output)

print(sku_output)

print("\n" + "=" * 70)