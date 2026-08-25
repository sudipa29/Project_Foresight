# ============================================================
# PROJECT FORESIGHT
# Phase 5.3 - Demand Forecast Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

FORECAST_PATH = (
    BASE_PATH /
    "data" /
    "processed" /
    "forecasting"
)

INPUT_PATH = (
    FORECAST_PATH /
    "forecast_demand_daily.csv"
)

OUTPUT_PATH = (
    FORECAST_PATH /
    "daily_forecasting_dataset.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading forecast demand dataset...")

demand = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(
    "Input shape:",
    demand.shape
)


# ============================================================
# DATE CONVERSION
# ============================================================

demand["date"] = pd.to_datetime(
    demand["date"],
    errors="coerce"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print(
    "\nDate range:",
    demand["date"].min(),
    "to",
    demand["date"].max()
)

print(
    "Rows:",
    len(demand)
)

print(
    "Stores:",
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
    "Missing demand:",
    demand["units_sold"].isna().sum()
)


# ============================================================
# SORT
# ============================================================

print("\nSorting data...")

demand = demand.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(
    drop=True
)


# ============================================================
# TIME FEATURES
# ============================================================

print("\nCreating time features...")


demand["year"] = (
    demand["date"].dt.year
)

demand["month"] = (
    demand["date"].dt.month
)

demand["quarter"] = (
    demand["date"].dt.quarter
)

demand["week_of_year"] = (
    demand["date"].dt.isocalendar().week.astype(int)
)

demand["day_of_week"] = (
    demand["date"].dt.dayofweek
)

demand["day_of_month"] = (
    demand["date"].dt.day
)

demand["is_weekend"] = (
    demand["day_of_week"] >= 5
).astype(int)

demand["is_month_start"] = (
    demand["date"].dt.is_month_start
).astype(int)

demand["is_month_end"] = (
    demand["date"].dt.is_month_end
).astype(int)

demand["is_quarter_start"] = (
    demand["date"].dt.is_quarter_start
).astype(int)

demand["is_quarter_end"] = (
    demand["date"].dt.is_quarter_end
).astype(int)


# ============================================================
# LAG FEATURES
# ============================================================

print("\nCreating lag features...")


grouped = demand.groupby(
    [
        "store_id",
        "sku_id"
    ],
    sort=False
)["units_sold"]


demand["lag_1"] = (
    grouped.shift(1)
)

demand["lag_7"] = (
    grouped.shift(7)
)

demand["lag_14"] = (
    grouped.shift(14)
)

demand["lag_30"] = (
    grouped.shift(30)
)


# ============================================================
# ROLLING FEATURES
# ============================================================

print("\nCreating rolling demand features...")


demand["rolling_mean_7"] = (
    grouped
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


demand["rolling_mean_14"] = (
    grouped
    .rolling(
        window=14,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


demand["rolling_mean_30"] = (
    grouped
    .rolling(
        window=30,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


# ============================================================
# ROLLING VOLATILITY
# ============================================================

print("\nCreating volatility features...")


demand["rolling_std_7"] = (
    grouped
    .rolling(
        window=7,
        min_periods=2
    )
    .std()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


demand["rolling_std_30"] = (
    grouped
    .rolling(
        window=30,
        min_periods=2
    )
    .std()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


# ============================================================
# ACTIVE DAYS
# ============================================================

print("\nCreating demand activity features...")


active_flag = (
    demand["units_sold"] > 0
).astype(int)


demand["active_days_7"] = (
    active_flag
    .groupby(
        [
            demand["store_id"],
            demand["sku_id"]
        ]
    )
    .rolling(
        window=7,
        min_periods=1
    )
    .sum()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


demand["active_days_30"] = (
    active_flag
    .groupby(
        [
            demand["store_id"],
            demand["sku_id"]
        ]
    )
    .rolling(
        window=30,
        min_periods=1
    )
    .sum()
    .reset_index(
        level=[
            0,
            1
        ],
        drop=True
    )
)


# ============================================================
# RECENT DEMAND TREND
# ============================================================

print("\nCreating demand trend features...")


demand["trend_7_vs_30"] = np.where(
    demand["rolling_mean_30"] > 0,

    demand["rolling_mean_7"] /
    demand["rolling_mean_30"],

    np.nan
)


# ============================================================
# INTERMITTENCY FEATURES
# ============================================================

print("\nCreating intermittency features...")


demand["demand_occurrence"] = (
    demand["units_sold"] > 0
).astype(int)


# Days since last positive demand
def days_since_demand(group):

    dates = group["date"]
    positive = (
        group["units_sold"] > 0
    )

    last_positive = (
        dates.where(positive)
        .ffill()
    )

    return (
        dates -
        last_positive
    ).dt.days


demand["days_since_demand"] = (
    demand
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        group_keys=False
    )
    .apply(
        days_since_demand
    )
)


# ============================================================
# FILL INITIAL LAG VALUES
# ============================================================

print("\nHandling initial missing lag values...")


lag_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_30",
    "active_days_7",
    "active_days_30"
]


for col in lag_columns:

    demand[col] = (
        demand[col]
        .fillna(0)
    )


demand["days_since_demand"] = (
    demand["days_since_demand"]
    .fillna(999)
)


# ============================================================
# REMOVE INF VALUES
# ============================================================

demand = demand.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING VALIDATION")
print("=" * 70)


print(
    "\nFinal dataset shape:",
    demand.shape
)

print(
    "Missing dates:",
    demand["date"].isna().sum()
)

print(
    "Missing demand:",
    demand["units_sold"].isna().sum()
)

print(
    "Missing lag_7:",
    demand["lag_7"].isna().sum()
)

print(
    "Missing rolling_mean_7:",
    demand["rolling_mean_7"].isna().sum()
)

print(
    "Missing rolling_mean_30:",
    demand["rolling_mean_30"].isna().sum()
)

print(
    "Missing days_since_demand:",
    demand["days_since_demand"].isna().sum()
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving engineered forecasting dataset...")

demand.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5.3 COMPLETED")
print("=" * 70)