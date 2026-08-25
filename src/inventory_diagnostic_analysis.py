import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================
# PROJECT FORESIGHT - INVENTORY DIAGNOSTIC ANALYSIS
# ==============================================================

print("=" * 70)
print("PROJECT FORESIGHT - INVENTORY–DEMAND–FORECAST DIAGNOSTIC")
print("=" * 70)


# ==============================================================
# PATHS
# ==============================================================

BASE_PATH = Path(__file__).resolve().parents[1]

INVENTORY_PATH = BASE_PATH / "data" / "processed" / "inventory_clean.csv"

DAILY_DEMAND_PATH = BASE_PATH / "data" / "processed" / "daily_demand.csv"

FORECAST_RISK_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "inventory_analysis"
    / "inventory_forecast_risk_analysis.csv"
)

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "inventory_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# LOAD DATA
# ==============================================================

print("\nLoading inventory data...")
inventory = pd.read_csv(INVENTORY_PATH)

print("Inventory shape:", inventory.shape)

print("\nLoading daily demand data...")
daily_demand = pd.read_csv(DAILY_DEMAND_PATH)

print("Daily demand shape:", daily_demand.shape)

print("\nLoading forecast-based inventory risk data...")
risk = pd.read_csv(FORECAST_RISK_PATH)

print("Risk analysis shape:", risk.shape)


# ==============================================================
# DATE CONVERSION
# ==============================================================

inventory["snapshot_date"] = pd.to_datetime(
    inventory["snapshot_date"],
    errors="coerce"
)

daily_demand["date"] = pd.to_datetime(
    daily_demand["date"],
    errors="coerce"
)


# ==============================================================
# LATEST INVENTORY
# ==============================================================

latest_snapshot = inventory["snapshot_date"].max()

current_inventory = inventory[
    inventory["snapshot_date"] == latest_snapshot
].copy()

print("\nLatest inventory snapshot:", latest_snapshot)
print("Current inventory rows:", len(current_inventory))


# ==============================================================
# DEMAND PERIODS
# ==============================================================

max_demand_date = daily_demand["date"].max()

date_30 = max_demand_date - pd.Timedelta(days=29)
date_90 = max_demand_date - pd.Timedelta(days=89)

demand_30 = daily_demand[
    daily_demand["date"] >= date_30
].copy()

demand_90 = daily_demand[
    daily_demand["date"] >= date_90
].copy()

print("\nDemand periods:")
print("30 days:", date_30.date(), "to", max_demand_date.date())
print("90 days:", date_90.date(), "to", max_demand_date.date())


# ==============================================================
# HISTORICAL DEMAND BY STORE + SKU
# ==============================================================

demand_30_summary = (
    demand_30
    .groupby(["store_id", "sku_id"], as_index=False)
    .agg(
        units_30d=("units_sold", "sum")
    )
)

demand_90_summary = (
    demand_90
    .groupby(["store_id", "sku_id"], as_index=False)
    .agg(
        units_90d=("units_sold", "sum")
    )
)


# ==============================================================
# MERGE DEMAND WITH INVENTORY
# ==============================================================

diagnostic = current_inventory.merge(
    demand_30_summary,
    on=["store_id", "sku_id"],
    how="left"
)

diagnostic = diagnostic.merge(
    demand_90_summary,
    on=["store_id", "sku_id"],
    how="left"
)

diagnostic["units_30d"] = diagnostic["units_30d"].fillna(0)

diagnostic["units_90d"] = diagnostic["units_90d"].fillna(0)


# ==============================================================
# HISTORICAL DAILY DEMAND
# ==============================================================

diagnostic["avg_daily_demand_30d"] = (
    diagnostic["units_30d"] / 30
)

diagnostic["avg_daily_demand_90d"] = (
    diagnostic["units_90d"] / 90
)


# ==============================================================
# DEMAND TREND
# ==============================================================

diagnostic["demand_trend"] = np.where(
    diagnostic["avg_daily_demand_90d"] > 0,
    diagnostic["avg_daily_demand_30d"]
    / diagnostic["avg_daily_demand_90d"],
    0
)


# ==============================================================
# MERGE FORECAST INFORMATION
# ==============================================================

forecast_columns = [
    "store_id",
    "sku_id",
    "forecast_30d_units",
    "forecast_daily_demand",
    "planning_daily_demand",
    "days_of_inventory",
    "forecast_coverage_ratio",
    "inventory_risk",
    "stockout_risk",
    "target_stock",
    "suggested_reorder_qty",
    "reorder_status",
    "priority"
]

forecast_columns = [
    c for c in forecast_columns
    if c in risk.columns
]

diagnostic = diagnostic.merge(
    risk[forecast_columns],
    on=["store_id", "sku_id"],
    how="left"
)


# ==============================================================
# FORECAST / HISTORICAL COMPARISON
# ==============================================================

diagnostic["historical_30d_forecast_30d_ratio"] = np.where(
    diagnostic["forecast_30d_units"] > 0,
    diagnostic["units_30d"]
    / diagnostic["forecast_30d_units"],
    np.nan
)

diagnostic["forecast_vs_90d_ratio"] = np.where(
    diagnostic["avg_daily_demand_90d"] > 0,
    diagnostic["forecast_daily_demand"]
    / diagnostic["avg_daily_demand_90d"],
    np.nan
)


# ==============================================================
# STOCK COVERAGE CALCULATIONS
# ==============================================================

diagnostic["stock_to_30d_demand"] = np.where(
    diagnostic["units_30d"] > 0,
    diagnostic["stock_on_hand"]
    / diagnostic["units_30d"],
    np.inf
)

diagnostic["stock_to_90d_demand"] = np.where(
    diagnostic["units_90d"] > 0,
    diagnostic["stock_on_hand"]
    / diagnostic["units_90d"],
    np.inf
)

diagnostic["stock_to_forecast"] = np.where(
    diagnostic["forecast_30d_units"] > 0,
    diagnostic["stock_on_hand"]
    / diagnostic["forecast_30d_units"],
    np.inf
)


# ==============================================================
# BASIC DISTRIBUTION ANALYSIS
# ==============================================================

print("\n" + "=" * 70)
print("INVENTORY DISTRIBUTION")
print("=" * 70)

print(
    diagnostic[
        [
            "stock_on_hand",
            "reorder_point",
            "safety_stock"
        ]
    ].describe()
)


# ==============================================================
# HISTORICAL DEMAND DISTRIBUTION
# ==============================================================

print("\n" + "=" * 70)
print("HISTORICAL DEMAND DISTRIBUTION")
print("=" * 70)

print(
    diagnostic[
        [
            "units_30d",
            "units_90d",
            "avg_daily_demand_30d",
            "avg_daily_demand_90d"
        ]
    ].describe()
)


# ==============================================================
# FORECAST DISTRIBUTION
# ==============================================================

print("\n" + "=" * 70)
print("FORECAST DISTRIBUTION")
print("=" * 70)

print(
    diagnostic[
        [
            "forecast_30d_units",
            "forecast_daily_demand",
            "planning_daily_demand"
        ]
    ].describe()
)


# ==============================================================
# ZERO-DEMAND ANALYSIS
# ==============================================================

print("\n" + "=" * 70)
print("ZERO-DEMAND ANALYSIS")
print("=" * 70)

zero_30d = (
    diagnostic["units_30d"] == 0
).sum()

zero_90d = (
    diagnostic["units_90d"] == 0
).sum()

zero_forecast = (
    diagnostic["forecast_30d_units"] == 0
).sum()

print("Items with zero 30-day demand:", zero_30d)

print("Items with zero 90-day demand:", zero_90d)

print("Items with zero 30-day forecast:", zero_forecast)

print(
    "Percentage with zero forecast:",
    round(zero_forecast / len(diagnostic) * 100, 2),
    "%"
)


# ==============================================================
# STOCK VS REORDER POINT
# ==============================================================

print("\n" + "=" * 70)
print("STOCK VS REORDER POINT")
print("=" * 70)

below_reorder = (
    diagnostic["stock_on_hand"]
    < diagnostic["reorder_point"]
).sum()

below_safety = (
    diagnostic["stock_on_hand"]
    < diagnostic["safety_stock"]
).sum()

print("Items below reorder point:", below_reorder)

print("Items below safety stock:", below_safety)


# ==============================================================
# STOCK VS FORECAST
# ==============================================================

print("\n" + "=" * 70)
print("STOCK VS FORECAST DEMAND")
print("=" * 70)

valid_forecast = diagnostic[
    diagnostic["forecast_30d_units"] > 0
].copy()

print(
    "Items with positive forecast:",
    len(valid_forecast)
)

if len(valid_forecast) > 0:

    print(
        "\nStock / Forecast Demand Ratio:"
    )

    print(
        valid_forecast[
            "stock_to_forecast"
        ].describe()
    )

    print(
        "\nItems with stock below forecast demand:",
        (
            valid_forecast["stock_on_hand"]
            < valid_forecast["forecast_30d_units"]
        ).sum()
    )

    print(
        "Items with stock below 2x forecast demand:",
        (
            valid_forecast["stock_on_hand"]
            < 2 * valid_forecast["forecast_30d_units"]
        ).sum()
    )

    print(
        "Items with stock below 5x forecast demand:",
        (
            valid_forecast["stock_on_hand"]
            < 5 * valid_forecast["forecast_30d_units"]
        ).sum()
    )


# ==============================================================
# FORECAST VS HISTORICAL DEMAND
# ==============================================================

print("\n" + "=" * 70)
print("FORECAST VS HISTORICAL DEMAND")
print("=" * 70)

valid_comparison = diagnostic[
    (diagnostic["forecast_30d_units"] > 0)
    &
    (diagnostic["units_30d"] > 0)
].copy()

print(
    "Items with both historical and forecast demand:",
    len(valid_comparison)
)

if len(valid_comparison) > 0:

    print(
        "\nHistorical 30-day demand / Forecast 30-day demand:"
    )

    print(
        valid_comparison[
            "historical_30d_forecast_30d_ratio"
        ].describe()
    )


# ==============================================================
# DEMAND TREND ANALYSIS
# ==============================================================

print("\n" + "=" * 70)
print("DEMAND TREND ANALYSIS")
print("=" * 70)

print(
    diagnostic[
        "demand_trend"
    ].describe()
)

increasing = (
    diagnostic["demand_trend"] > 1.2
).sum()

stable = (
    (diagnostic["demand_trend"] >= 0.8)
    &
    (diagnostic["demand_trend"] <= 1.2)
).sum()

decreasing = (
    diagnostic["demand_trend"] < 0.8
).sum()

print("\nIncreasing demand:", increasing)

print("Stable demand:", stable)

print("Decreasing demand:", decreasing)


# ==============================================================
# RISK DISTRIBUTION
# ==============================================================

print("\n" + "=" * 70)
print("CURRENT RISK DISTRIBUTION")
print("=" * 70)

if "inventory_risk" in diagnostic.columns:

    print(
        diagnostic["inventory_risk"]
        .value_counts(dropna=False)
    )

if "stockout_risk" in diagnostic.columns:

    print(
        "\nStockout risk:"
    )

    print(
        diagnostic["stockout_risk"]
        .value_counts(dropna=False)
    )

if "priority" in diagnostic.columns:

    print(
        "\nPriority:"
    )

    print(
        diagnostic["priority"]
        .value_counts(dropna=False)
    )


# ==============================================================
# TOP ITEMS BY FORECAST DEMAND
# ==============================================================

print("\n" + "=" * 70)
print("TOP 20 ITEMS BY FORECAST DEMAND")
print("=" * 70)

top_forecast = (
    diagnostic
    .sort_values(
        "forecast_30d_units",
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
            "units_90d",
            "forecast_30d_units",
            "forecast_daily_demand",
            "inventory_risk"
        ]
    ]
    .head(20)
)

print(top_forecast.to_string(index=False))


# ==============================================================
# TOP ITEMS WITH LOWEST STOCK COVERAGE
# ==============================================================

print("\n" + "=" * 70)
print("20 ITEMS WITH LOWEST FORECAST STOCK COVERAGE")
print("=" * 70)

lowest_coverage = (
    diagnostic[
        diagnostic["forecast_30d_units"] > 0
    ]
    .sort_values(
        "stock_to_forecast",
        ascending=True
    )
    [
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "forecast_30d_units",
            "stock_to_forecast",
            "inventory_risk",
            "stockout_risk",
            "reorder_status"
        ]
    ]
    .head(20)
)

print(
    lowest_coverage.to_string(index=False)
)


# ==============================================================
# POSSIBLE MODEL / DATA ISSUES
# ==============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC FLAGS")
print("=" * 70)

flags = []

if zero_forecast > 0:
    flags.append(
        f"{zero_forecast} items have zero forecast demand"
    )

if (
    diagnostic["inventory_risk"]
    .nunique(dropna=True) == 1
):
    flags.append(
        "All inventory items have the same inventory risk category"
    )

if (
    diagnostic["stockout_risk"]
    .nunique(dropna=True) == 1
):
    flags.append(
        "All inventory items have the same stockout risk category"
    )

if (
    diagnostic["priority"]
    .nunique(dropna=True) == 1
):
    flags.append(
        "All inventory items have the same priority"
    )

if below_reorder == 0:
    flags.append(
        "No items are below the reorder point"
    )

if below_safety == 0:
    flags.append(
        "No items are below safety stock"
    )

if len(flags) == 0:

    print("No major diagnostic flags found.")

else:

    for i, flag in enumerate(flags, 1):

        print(
            f"{i}. {flag}"
        )


# ==============================================================
# SAVE DIAGNOSTIC FILE
# ==============================================================

OUTPUT_PATH = (
    OUTPUT_DIR
    / "inventory_diagnostic_analysis.csv"
)

diagnostic.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==============================================================
# SAVE SUMMARY
# ==============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "inventory_items",
            "zero_30d_demand",
            "zero_90d_demand",
            "zero_forecast",
            "below_reorder_point",
            "below_safety_stock",
            "positive_forecast_items",
            "increasing_demand_items",
            "stable_demand_items",
            "decreasing_demand_items"
        ],
        "value": [
            len(diagnostic),
            zero_30d,
            zero_90d,
            zero_forecast,
            below_reorder,
            below_safety,
            len(valid_forecast),
            increasing,
            stable,
            decreasing
        ]
    }
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "inventory_diagnostic_summary.csv"
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ==============================================================
# COMPLETION
# ==============================================================

print("\n" + "=" * 70)
print("INVENTORY DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(OUTPUT_PATH)

print(SUMMARY_PATH)

print("\n" + "=" * 70)