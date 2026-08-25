# ============================================================
# PROJECT FORESIGHT
# Phase 5 - Demand Forecasting
# Step 5.1 - Forecast Dataset Preparation
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

PROCESSED_PATH = BASE_PATH / "data" / "processed"

DEMAND_PATH = (
    PROCESSED_PATH /
    "daily_demand.csv"
)

FORECAST_PATH = (
    PROCESSED_PATH /
    "forecasting"
)

FORECAST_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

print("=" * 70)
print("PROJECT FORESIGHT - DEMAND FORECASTING ENGINE")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading daily demand data...")

demand = pd.read_csv(
    DEMAND_PATH,
    low_memory=False
)

print(
    "Daily demand shape:",
    demand.shape
)


# ------------------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------------------

demand["date"] = pd.to_datetime(
    demand["date"],
    errors="coerce"
)


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)

print(
    "\nDate range:"
)

print(
    demand["date"].min(),
    "to",
    demand["date"].max()
)

print(
    "\nStores:",
    demand["store_id"].nunique()
)

print(
    "SKUs:",
    demand["sku_id"].nunique()
)

print(
    "Store-SKU combinations:",
    demand[
        ["store_id", "sku_id"]
    ].drop_duplicates().shape[0]
)

print(
    "Rows:",
    len(demand)
)


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "date",
    "store_id",
    "sku_id",
    "units_sold"
]

missing_columns = [
    col
    for col in required_columns
    if col not in demand.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

demand["units_sold"] = pd.to_numeric(
    demand["units_sold"],
    errors="coerce"
)

demand["units_sold"] = (
    demand["units_sold"]
    .fillna(0)
)


# ------------------------------------------------------------
# REMOVE INVALID DATES
# ------------------------------------------------------------

invalid_dates = (
    demand["date"].isna().sum()
)

print(
    "\nInvalid dates:",
    invalid_dates
)

demand = demand[
    demand["date"].notna()
].copy()


# ------------------------------------------------------------
# CHECK NEGATIVE DEMAND
# ------------------------------------------------------------

negative_demand = (
    demand["units_sold"] < 0
).sum()

print(
    "Negative demand rows:",
    negative_demand
)

if negative_demand > 0:

    print(
        "Warning: negative demand detected."
    )


# ------------------------------------------------------------
# SORT DATA
# ------------------------------------------------------------

demand = demand.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# CREATE COMPLETE DAILY STORE-SKU GRID
# ------------------------------------------------------------
#
# This is important for forecasting.
#
# If a store-SKU has no sales on a particular day,
# that day must appear as zero demand instead of
# being completely absent from the time series.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING COMPLETE DAILY DEMAND GRID")
print("=" * 70)

min_date = demand["date"].min()
max_date = demand["date"].max()

print(
    "\nForecasting date range:",
    min_date.date(),
    "to",
    max_date.date()
)

date_range = pd.date_range(
    start=min_date,
    end=max_date,
    freq="D"
)

store_sku = (
    demand[
        [
            "store_id",
            "sku_id"
        ]
    ]
    .drop_duplicates()
)

print(
    "\nUnique store-SKU combinations:",
    len(store_sku)
)

print(
    "Number of dates:",
    len(date_range)
)


# ------------------------------------------------------------
# CREATE GRID
# ------------------------------------------------------------

store_sku["key"] = 1

dates = pd.DataFrame(
    {
        "date": date_range,
        "key": 1
    }
)

forecast_grid = store_sku.merge(
    dates,
    on="key",
    how="outer"
)

forecast_grid = forecast_grid.drop(
    columns=["key"]
)


# ------------------------------------------------------------
# MERGE ACTUAL DEMAND
# ------------------------------------------------------------

actual_demand = (
    demand[
        [
            "date",
            "store_id",
            "sku_id",
            "units_sold"
        ]
    ]
    .groupby(
        [
            "date",
            "store_id",
            "sku_id"
        ],
        as_index=False
    )
    .agg(
        units_sold=("units_sold", "sum")
    )
)

forecast_grid = forecast_grid.merge(
    actual_demand,
    on=[
        "date",
        "store_id",
        "sku_id"
    ],
    how="left"
)


# ------------------------------------------------------------
# FILL MISSING DEMAND WITH ZERO
# ------------------------------------------------------------

forecast_grid["units_sold"] = (
    forecast_grid["units_sold"]
    .fillna(0)
)


# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

forecast_grid = forecast_grid.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# ADD TIME FEATURES
# ------------------------------------------------------------

forecast_grid["day_of_week"] = (
    forecast_grid["date"]
    .dt.dayofweek
)

forecast_grid["day_name"] = (
    forecast_grid["date"]
    .dt.day_name()
)

forecast_grid["week_of_year"] = (
    forecast_grid["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

forecast_grid["month"] = (
    forecast_grid["date"]
    .dt.month
)

forecast_grid["quarter"] = (
    forecast_grid["date"]
    .dt.quarter
)

forecast_grid["year"] = (
    forecast_grid["date"]
    .dt.year
)

forecast_grid["is_weekend"] = (
    forecast_grid["day_of_week"] >= 5
)


# ------------------------------------------------------------
# STORE-SKU DEMAND SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STORE-SKU DEMAND SUMMARY")
print("=" * 70)

summary = (
    forecast_grid
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )
    .agg(
        total_units=(
            "units_sold",
            "sum"
        ),
        avg_daily_units=(
            "units_sold",
            "mean"
        ),
        max_daily_units=(
            "units_sold",
            "max"
        ),
        active_days=(
            "units_sold",
            lambda x: (x > 0).sum()
        ),
        zero_demand_days=(
            "units_sold",
            lambda x: (x == 0).sum()
        ),
        demand_std=(
            "units_sold",
            "std"
        )
    )
)


# ------------------------------------------------------------
# DEMAND ACTIVITY RATE
# ------------------------------------------------------------

total_days = (
    forecast_grid["date"]
    .nunique()
)

summary["demand_activity_rate"] = (
    summary["active_days"] /
    total_days
)


# ------------------------------------------------------------
# DEMAND CLASSIFICATION
# ------------------------------------------------------------

def classify_demand_activity(rate):

    if rate == 0:
        return "No Demand"

    elif rate < 0.10:
        return "Very Sparse"

    elif rate < 0.30:
        return "Sparse"

    elif rate < 0.60:
        return "Moderate"

    else:
        return "Frequent"


summary["demand_activity_class"] = (
    summary["demand_activity_rate"]
    .apply(classify_demand_activity)
)


# ------------------------------------------------------------
# PRINT DEMAND ACTIVITY DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nDemand activity distribution:"
)

print(
    summary[
        "demand_activity_class"
    ].value_counts()
)


# ------------------------------------------------------------
# PRINT DEMAND STATISTICS
# ------------------------------------------------------------

print(
    "\nTotal store-SKU combinations:",
    len(summary)
)

print(
    "\nTotal units sold:",
    summary["total_units"].sum()
)

print(
    "\nAverage daily units:",
    forecast_grid[
        "units_sold"
    ].mean()
)

print(
    "\nMedian daily units:",
    forecast_grid[
        "units_sold"
    ].median()
)

print(
    "\nZero-demand observations:",
    (
        forecast_grid[
            "units_sold"
        ] == 0
    ).sum()
)

print(
    "\nTotal observations:",
    len(forecast_grid)
)


# ------------------------------------------------------------
# SAVE FORECAST DAILY DATA
# ------------------------------------------------------------

forecast_daily_file = (
    FORECAST_PATH /
    "forecast_demand_daily.csv"
)

forecast_grid.to_csv(
    forecast_daily_file,
    index=False
)


# ------------------------------------------------------------
# SAVE STORE-SKU SUMMARY
# ------------------------------------------------------------

summary_file = (
    FORECAST_PATH /
    "forecast_store_sku_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    "\nForecast daily shape:",
    forecast_grid.shape
)

print(
    "Forecast summary shape:",
    summary.shape
)

print(
    "Missing dates:",
    forecast_grid["date"].isna().sum()
)

print(
    "Missing store IDs:",
    forecast_grid["store_id"].isna().sum()
)

print(
    "Missing SKU IDs:",
    forecast_grid["sku_id"].isna().sum()
)

print(
    "Missing demand values:",
    forecast_grid["units_sold"].isna().sum()
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 5.1 COMPLETED")
print("=" * 70)

print(
    "\nForecast daily dataset saved to:"
)

print(
    forecast_daily_file
)

print(
    "\nStore-SKU summary saved to:"
)

print(
    summary_file
)