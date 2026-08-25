import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT FORESIGHT
# FORECAST-BASED INVENTORY RISK ANALYSIS
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FORECAST-BASED INVENTORY RISK ANALYSIS")
print("=" * 70)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


INVENTORY_PATH = os.path.join(
    PROCESSED_DIR,
    "inventory_clean.csv"
)


DAILY_DEMAND_PATH = os.path.join(
    PROCESSED_DIR,
    "daily_demand.csv"
)


FUTURE_FORECAST_PATH = os.path.join(
    PROCESSED_DIR,
    "forecasting",
    "future",
    "future_30_day_forecast.csv"
)


OUTPUT_DIR = os.path.join(
    PROCESSED_DIR,
    "inventory_analysis"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading inventory data...")


inventory = pd.read_csv(
    INVENTORY_PATH
)


daily_demand = pd.read_csv(
    DAILY_DEMAND_PATH
)


future_forecast = pd.read_csv(
    FUTURE_FORECAST_PATH
)


print(
    f"Inventory shape: {inventory.shape}"
)


print(
    f"Daily demand shape: {daily_demand.shape}"
)


print(
    f"Future forecast shape: {future_forecast.shape}"
)


# ============================================================
# DATE CONVERSION
# ============================================================

inventory["snapshot_date"] = pd.to_datetime(
    inventory["snapshot_date"],
    errors="coerce"
)


inventory["last_restock_date"] = pd.to_datetime(
    inventory["last_restock_date"],
    errors="coerce"
)


daily_demand["date"] = pd.to_datetime(
    daily_demand["date"],
    errors="coerce"
)


future_forecast["date"] = pd.to_datetime(
    future_forecast["date"],
    errors="coerce"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "snapshot_date"
]


required_demand_columns = [
    "date",
    "store_id",
    "sku_id",
    "units_sold"
]


required_forecast_columns = [
    "date",
    "forecast_units"
]


missing_inventory = [
    col for col in required_inventory_columns
    if col not in inventory.columns
]


missing_demand = [
    col for col in required_demand_columns
    if col not in daily_demand.columns
]


missing_forecast = [
    col for col in required_forecast_columns
    if col not in future_forecast.columns
]


if missing_inventory:

    raise ValueError(
        f"Missing inventory columns: {missing_inventory}"
    )


if missing_demand:

    raise ValueError(
        f"Missing demand columns: {missing_demand}"
    )


if missing_forecast:

    raise ValueError(
        f"Missing forecast columns: {missing_forecast}"
    )


# ============================================================
# LATEST INVENTORY SNAPSHOT
# ============================================================

latest_snapshot = inventory[
    "snapshot_date"
].max()


current_inventory = inventory[
    inventory["snapshot_date"] == latest_snapshot
].copy()


print(
    f"\nLatest inventory snapshot: "
    f"{latest_snapshot.date()}"
)


print(
    f"Current inventory rows: "
    f"{len(current_inventory)}"
)


# ============================================================
# FORECAST SUMMARY
# ============================================================

future_forecast = future_forecast.sort_values(
    "date"
).copy()


forecast_days = len(
    future_forecast
)


total_forecast_demand = future_forecast[
    "forecast_units"
].sum()


avg_forecast_daily_demand = future_forecast[
    "forecast_units"
].mean()


max_forecast_daily_demand = future_forecast[
    "forecast_units"
].max()


min_forecast_daily_demand = future_forecast[
    "forecast_units"
].min()


print("\n" + "=" * 70)
print("FUTURE FORECAST SUMMARY")
print("=" * 70)


print(
    f"Forecast period: "
    f"{future_forecast['date'].min().date()} "
    f"to "
    f"{future_forecast['date'].max().date()}"
)


print(
    f"Forecast days: {forecast_days}"
)


print(
    f"Total forecast demand: "
    f"{total_forecast_demand:,.0f}"
)


print(
    f"Average daily forecast demand: "
    f"{avg_forecast_daily_demand:,.2f}"
)


# ============================================================
# HISTORICAL DEMAND
# ============================================================

demand_max_date = daily_demand[
    "date"
].max()


demand_30_start = (
    demand_max_date
    - pd.Timedelta(days=29)
)


demand_90_start = (
    demand_max_date
    - pd.Timedelta(days=89)
)


print("\n" + "=" * 70)
print("HISTORICAL DEMAND PERIODS")
print("=" * 70)


print(
    f"30 days: "
    f"{demand_30_start.date()} "
    f"to "
    f"{demand_max_date.date()}"
)


print(
    f"90 days: "
    f"{demand_90_start.date()} "
    f"to "
    f"{demand_max_date.date()}"
)


# ============================================================
# 30-DAY DEMAND BY STORE / SKU
# ============================================================

demand_30 = daily_demand[
    (
        daily_demand["date"]
        >= demand_30_start
    )
    &
    (
        daily_demand["date"]
        <= demand_max_date
    )
].copy()


demand_30_grouped = (
    demand_30
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )[
        "units_sold"
    ]
    .sum()
    .rename(
        columns={
            "units_sold": "units_30d"
        }
    )
)


# ============================================================
# 90-DAY DEMAND BY STORE / SKU
# ============================================================

demand_90 = daily_demand[
    (
        daily_demand["date"]
        >= demand_90_start
    )
    &
    (
        daily_demand["date"]
        <= demand_max_date
    )
].copy()


demand_90_grouped = (
    demand_90
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )[
        "units_sold"
    ]
    .sum()
    .rename(
        columns={
            "units_sold": "units_90d"
        }
    )
)


# ============================================================
# MERGE HISTORICAL DEMAND
# ============================================================

risk = current_inventory.merge(
    demand_30_grouped,
    on=[
        "store_id",
        "sku_id"
    ],
    how="left"
)


risk = risk.merge(
    demand_90_grouped,
    on=[
        "store_id",
        "sku_id"
    ],
    how="left"
)


risk["units_30d"] = risk[
    "units_30d"
].fillna(0)


risk["units_90d"] = risk[
    "units_90d"
].fillna(0)


# ============================================================
# HISTORICAL DAILY DEMAND
# ============================================================

risk["avg_daily_demand_30d"] = (
    risk["units_30d"] / 30
)


risk["avg_daily_demand_90d"] = (
    risk["units_90d"] / 90
)


# ============================================================
# DEMAND TREND
# ============================================================

risk["demand_trend"] = np.where(
    risk["avg_daily_demand_90d"] > 0,
    risk["avg_daily_demand_30d"]
    /
    risk["avg_daily_demand_90d"],
    0
)


# ============================================================
# FORECAST ALLOCATION
# ============================================================
#
# The future forecast is an overall business-level forecast.
# We allocate the forecast to each store-SKU using the recent
# 90-day demand share.
#
# ============================================================

total_historical_90d = risk[
    "units_90d"
].sum()


if total_historical_90d > 0:

    risk["demand_share"] = (
        risk["units_90d"]
        /
        total_historical_90d
    )

else:

    risk["demand_share"] = 0


risk["forecast_30d_units"] = (
    risk["demand_share"]
    *
    total_forecast_demand
)


# ============================================================
# FORECAST DAILY DEMAND
# ============================================================

risk["forecast_daily_demand"] = (
    risk["forecast_30d_units"]
    /
    forecast_days
)


# ============================================================
# COMBINED DEMAND ESTIMATE
# ============================================================
#
# We use the forecast as the primary forward-looking signal.
#
# For SKUs with zero forecast allocation but historical demand,
# the recent 30-day demand is used as a fallback.
#
# ============================================================

risk["planning_daily_demand"] = np.where(
    risk["forecast_daily_demand"] > 0,
    risk["forecast_daily_demand"],
    risk["avg_daily_demand_30d"]
)


# ============================================================
# DAYS OF INVENTORY
# ============================================================

risk["days_of_inventory"] = np.where(
    risk["planning_daily_demand"] > 0,
    risk["stock_on_hand"]
    /
    risk["planning_daily_demand"],
    np.inf
)


# ============================================================
# STOCK VS REORDER POINT
# ============================================================

risk["stock_vs_reorder_point"] = (
    risk["stock_on_hand"]
    -
    risk["reorder_point"]
)


# ============================================================
# STOCK VS SAFETY STOCK
# ============================================================

risk["stock_vs_safety_stock"] = (
    risk["stock_on_hand"]
    -
    risk["safety_stock"]
)


# ============================================================
# FORECAST COVERAGE
# ============================================================

risk["forecast_coverage_ratio"] = np.where(
    risk["forecast_30d_units"] > 0,
    risk["stock_on_hand"]
    /
    risk["forecast_30d_units"],
    np.inf
)


# ============================================================
# INVENTORY RISK CLASSIFICATION
# ============================================================

def calculate_inventory_risk(row):

    stock = row["stock_on_hand"]

    reorder_point = row["reorder_point"]

    safety_stock = row["safety_stock"]

    days = row["days_of_inventory"]

    forecast_coverage = row[
        "forecast_coverage_ratio"
    ]


    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if stock <= safety_stock:

        return "Critical"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if stock <= reorder_point:

        return "High"


    if forecast_coverage < 0.50:

        return "High"


    if days < 14:

        return "High"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if forecast_coverage < 1:

        return "Medium"


    if days < 30:

        return "Medium"


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if days < 60:

        return "Low"


    # --------------------------------------------------------
    # VERY LOW
    # --------------------------------------------------------

    return "Very Low"


risk["inventory_risk"] = risk.apply(
    calculate_inventory_risk,
    axis=1
)


# ============================================================
# STOCKOUT RISK
# ============================================================

def calculate_stockout_risk(row):

    stock = row["stock_on_hand"]

    safety_stock = row["safety_stock"]

    days = row["days_of_inventory"]


    if stock <= safety_stock:

        return "Very High"


    if days < 14:

        return "High"


    if days < 30:

        return "Medium"


    if days < 60:

        return "Low"


    return "Very Low"


risk["stockout_risk"] = risk.apply(
    calculate_stockout_risk,
    axis=1
)


# ============================================================
# REORDER TARGET
# ============================================================
#
# Target stock =
# forecast demand for 30 days + safety stock
#
# ============================================================

risk["target_stock"] = (
    risk["forecast_30d_units"]
    +
    risk["safety_stock"]
)


# ============================================================
# SUGGESTED REORDER QUANTITY
# ============================================================

risk["suggested_reorder_qty"] = (
    risk["target_stock"]
    -
    risk["stock_on_hand"]
)


risk["suggested_reorder_qty"] = (
    risk["suggested_reorder_qty"]
    .clip(lower=0)
    .round()
)


# ============================================================
# REORDER STATUS
# ============================================================

risk["reorder_status"] = np.where(
    risk["suggested_reorder_qty"] > 0,
    "Reorder Required",
    "Sufficient Stock"
)


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(row):

    if row["inventory_risk"] == "Critical":

        return "Critical"


    if row["inventory_risk"] == "High":

        return "High"


    if row["inventory_risk"] == "Medium":

        return "Medium"


    if row["inventory_risk"] == "Low":

        return "Low"


    return "Normal"


risk["priority"] = risk.apply(
    calculate_priority,
    axis=1
)


# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

risk["days_of_inventory"] = risk[
    "days_of_inventory"
].replace(
    [np.inf, -np.inf],
    np.nan
)


risk["forecast_coverage_ratio"] = risk[
    "forecast_coverage_ratio"
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SORT BY RISK
# ============================================================

risk_order = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4,
    "Very Low": 5
}


risk["risk_rank"] = risk[
    "inventory_risk"
].map(
    risk_order
)


risk = risk.sort_values(
    [
        "risk_rank",
        "days_of_inventory"
    ],
    ascending=[
        True,
        True
    ]
)


# ============================================================
# DISPLAY COLUMNS
# ============================================================

output_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "units_30d",
    "units_90d",
    "avg_daily_demand_30d",
    "avg_daily_demand_90d",
    "demand_trend",
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


risk_output = risk[
    output_columns
].copy()


# ============================================================
# RISK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FORECAST-BASED INVENTORY RISK SUMMARY")
print("=" * 70)


print(
    f"\nTotal inventory items: "
    f"{len(risk_output):,}"
)


risk_distribution = (
    risk_output[
        "inventory_risk"
    ]
    .value_counts()
)


for level in [
    "Critical",
    "High",
    "Medium",
    "Low",
    "Very Low"
]:

    print(
        f"{level}: "
        f"{risk_distribution.get(level, 0):,}"
    )


reorder_count = (
    risk_output[
        "reorder_status"
    ]
    .eq("Reorder Required")
    .sum()
)


total_reorder_qty = (
    risk_output[
        "suggested_reorder_qty"
    ]
    .sum()
)


print(
    f"Reorder required: "
    f"{reorder_count:,}"
)


print(
    f"Total suggested reorder quantity: "
    f"{total_reorder_qty:,.0f}"
)


# ============================================================
# TOP 20 RISKS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FORECAST-BASED INVENTORY RISKS")
print("=" * 70)


top_20 = risk_output.head(20)


print(
    top_20.to_string(
        index=False
    )
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)


print(
    risk_output[
        "inventory_risk"
    ].value_counts()
)


# ============================================================
# REORDER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REORDER SUMMARY")
print("=" * 70)


reorder_recommendations = (
    risk_output[
        risk_output[
            "reorder_status"
        ]
        ==
        "Reorder Required"
    ]
    .copy()
)


if reorder_recommendations.empty:

    print(
        "No items currently require reorder "
        "under the forecast-based rules."
    )

else:

    print(
        f"Items requiring reorder: "
        f"{len(reorder_recommendations):,}"
    )


    print(
        f"Total reorder quantity: "
        f"{reorder_recommendations['suggested_reorder_qty'].sum():,.0f}"
    )


    print("\nTop reorder recommendations:")


    print(
        reorder_recommendations[
            [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "forecast_30d_units",
                "safety_stock",
                "suggested_reorder_qty",
                "inventory_risk",
                "priority"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE MAIN ANALYSIS
# ============================================================

main_output_path = os.path.join(
    OUTPUT_DIR,
    "inventory_forecast_risk_analysis.csv"
)


risk_output.to_csv(
    main_output_path,
    index=False
)


# ============================================================
# SAVE HIGH RISK ITEMS
# ============================================================

high_risk = risk_output[
    risk_output[
        "inventory_risk"
    ].isin(
        [
            "Critical",
            "High",
            "Medium"
        ]
    )
].copy()


high_risk_path = os.path.join(
    OUTPUT_DIR,
    "high_risk_inventory_forecast.csv"
)


high_risk.to_csv(
    high_risk_path,
    index=False
)


# ============================================================
# SAVE REORDER RECOMMENDATIONS
# ============================================================

reorder_path = os.path.join(
    OUTPUT_DIR,
    "forecast_reorder_recommendations.csv"
)


reorder_recommendations.to_csv(
    reorder_path,
    index=False
)


# ============================================================
# SAVE RISK SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Total Inventory Items",
            "Critical Items",
            "High Risk Items",
            "Medium Risk Items",
            "Low Risk Items",
            "Very Low Risk Items",
            "Reorder Required",
            "Total Suggested Reorder Quantity",
            "Total Forecast Demand 30 Days",
            "Average Forecast Daily Demand"
        ],
        "value": [
            len(risk_output),
            risk_distribution.get(
                "Critical",
                0
            ),
            risk_distribution.get(
                "High",
                0
            ),
            risk_distribution.get(
                "Medium",
                0
            ),
            risk_distribution.get(
                "Low",
                0
            ),
            risk_distribution.get(
                "Very Low",
                0
            ),
            reorder_count,
            total_reorder_qty,
            total_forecast_demand,
            avg_forecast_daily_demand
        ]
    }
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "forecast_inventory_risk_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("FORECAST-BASED INVENTORY RISK ANALYSIS COMPLETED")
print("=" * 70)


print("\nFiles saved:")


print(
    main_output_path
)


print(
    high_risk_path
)


print(
    reorder_path
)


print(
    summary_path
)


print("\n" + "=" * 70)