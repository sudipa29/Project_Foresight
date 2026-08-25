# ============================================================
# PROJECT FORESIGHT
# Phase 5 - Demand Forecasting
# Step 5.2 - Demand Pattern Analysis & Baseline Forecast
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

PROCESSED_PATH = (
    BASE_PATH /
    "data" /
    "processed"
)

FORECAST_PATH = (
    PROCESSED_PATH /
    "forecasting"
)

INPUT_PATH = (
    FORECAST_PATH /
    "forecast_demand_daily.csv"
)

SUMMARY_PATH = (
    FORECAST_PATH /
    "forecast_store_sku_summary.csv"
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

FORECAST_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

print("=" * 70)
print("PROJECT FORESIGHT - DEMAND FORECAST BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading forecast dataset...")

demand = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(
    "Forecast dataset shape:",
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
# GLOBAL DEMAND STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GLOBAL DEMAND STATISTICS")
print("=" * 70)

total_observations = len(demand)

zero_observations = (
    demand["units_sold"] == 0
).sum()

positive_observations = (
    demand["units_sold"] > 0
).sum()

zero_pct = (
    zero_observations /
    total_observations *
    100
)

positive_pct = (
    positive_observations /
    total_observations *
    100
)

print(
    "\nTotal observations:",
    total_observations
)

print(
    "Zero-demand observations:",
    zero_observations
)

print(
    "Positive-demand observations:",
    positive_observations
)

print(
    f"Zero-demand percentage: {zero_pct:.2f}%"
)

print(
    f"Positive-demand percentage: {positive_pct:.2f}%"
)

print(
    "\nMean daily demand:",
    demand["units_sold"].mean()
)

print(
    "Median daily demand:",
    demand["units_sold"].median()
)

print(
    "Maximum daily demand:",
    demand["units_sold"].max()
)


# ------------------------------------------------------------
# STORE-SKU DEMAND PROFILE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STORE-SKU DEMAND PROFILE")
print("=" * 70)


def calculate_profile(group):

    units = group["units_sold"]

    total_days = len(units)

    active_days = (
        units > 0
    ).sum()

    zero_days = (
        units == 0
    ).sum()

    activity_rate = (
        active_days /
        total_days
    )

    positive_demand = (
        units[units > 0]
    )

    if len(positive_demand) > 0:

        avg_positive_demand = (
            positive_demand.mean()
        )

        std_positive_demand = (
            positive_demand.std()
        )

    else:

        avg_positive_demand = 0

        std_positive_demand = 0

    if activity_rate == 0:

        demand_class = "No Demand"

    elif activity_rate < 0.05:

        demand_class = "Extremely Sparse"

    elif activity_rate < 0.10:

        demand_class = "Very Sparse"

    elif activity_rate < 0.30:

        demand_class = "Sparse"

    elif activity_rate < 0.60:

        demand_class = "Moderate"

    else:

        demand_class = "Frequent"

    return pd.Series(
        {
            "total_units": units.sum(),

            "active_days":
                active_days,

            "zero_days":
                zero_days,

            "activity_rate":
                activity_rate,

            "avg_positive_demand":
                avg_positive_demand,

            "std_positive_demand":
                std_positive_demand,

            "max_daily_demand":
                units.max(),

            "demand_class":
                demand_class
        }
    )


profile = (
    demand
    .groupby(
        [
            "store_id",
            "sku_id"
        ]
    )
    .apply(
        calculate_profile,
        include_groups=False
    )
    .reset_index()
)


# ------------------------------------------------------------
# DEMAND CLASS DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nDemand class distribution:"
)

print(
    profile[
        "demand_class"
    ].value_counts()
)


# ------------------------------------------------------------
# ACTIVITY RATE SUMMARY
# ------------------------------------------------------------

print(
    "\nActivity rate statistics:"
)

print(
    profile[
        "activity_rate"
    ].describe()
)


# ------------------------------------------------------------
# RECENT DEMAND WINDOWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RECENT DEMAND WINDOWS")
print("=" * 70)


reference_date = demand["date"].max()


print(
    "\nReference date:",
    reference_date.date()
)


# ------------------------------------------------------------
# 7-DAY DEMAND
# ------------------------------------------------------------

recent_7d = demand[
    (
        demand["date"]
        >=
        reference_date -
        pd.Timedelta(days=6)
    )
].copy()


# ------------------------------------------------------------
# 14-DAY DEMAND
# ------------------------------------------------------------

recent_14d = demand[
    (
        demand["date"]
        >=
        reference_date -
        pd.Timedelta(days=13)
    )
].copy()


# ------------------------------------------------------------
# 30-DAY DEMAND
# ------------------------------------------------------------

recent_30d = demand[
    (
        demand["date"]
        >=
        reference_date -
        pd.Timedelta(days=29)
    )
].copy()


# ------------------------------------------------------------
# 90-DAY DEMAND
# ------------------------------------------------------------

recent_90d = demand[
    (
        demand["date"]
        >=
        reference_date -
        pd.Timedelta(days=89)
    )
].copy()


def aggregate_window(
    df,
    name
):

    return (
        df
        .groupby(
            [
                "store_id",
                "sku_id"
            ],
            as_index=False
        )
        ["units_sold"]
        .sum()
        .rename(
            columns={
                "units_sold": name
            }
        )
    )


window_7d = aggregate_window(
    recent_7d,
    "units_7d"
)

window_14d = aggregate_window(
    recent_14d,
    "units_14d"
)

window_30d = aggregate_window(
    recent_30d,
    "units_30d"
)

window_90d = aggregate_window(
    recent_90d,
    "units_90d"
)


# ------------------------------------------------------------
# MERGE WINDOWS
# ------------------------------------------------------------

baseline = profile.copy()

for window in [
    window_7d,
    window_14d,
    window_30d,
    window_90d
]:

    baseline = baseline.merge(
        window,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )


baseline[
    [
        "units_7d",
        "units_14d",
        "units_30d",
        "units_90d"
    ]
] = (
    baseline[
        [
            "units_7d",
            "units_14d",
            "units_30d",
            "units_90d"
        ]
    ]
    .fillna(0)
)


# ------------------------------------------------------------
# DAILY BASELINE DEMAND
# ------------------------------------------------------------

baseline["avg_daily_7d"] = (
    baseline["units_7d"] / 7
)

baseline["avg_daily_14d"] = (
    baseline["units_14d"] / 14
)

baseline["avg_daily_30d"] = (
    baseline["units_30d"] / 30
)

baseline["avg_daily_90d"] = (
    baseline["units_90d"] / 90
)


# ------------------------------------------------------------
# BASELINE FORECASTS
# ------------------------------------------------------------
#
# Forecast is expressed as expected units per day.
#
# These are NOT ML forecasts yet.
# They are benchmark models that future ML models
# must outperform.
# ------------------------------------------------------------

baseline["forecast_naive"] = (
    baseline["units_7d"] / 7
)


baseline["forecast_ma_7"] = (
    baseline["units_7d"] / 7
)


baseline["forecast_ma_14"] = (
    baseline["units_14d"] / 14
)


baseline["forecast_ma_30"] = (
    baseline["units_30d"] / 30
)


baseline["forecast_ma_90"] = (
    baseline["units_90d"] / 90
)


# ------------------------------------------------------------
# WEIGHTED MOVING AVERAGE
# ------------------------------------------------------------
#
# Recent demand receives more weight.
#
# 7-day  = 50%
# 14-day = 30%
# 30-day = 20%
# ------------------------------------------------------------

baseline["forecast_weighted"] = (
    0.50 *
    baseline["avg_daily_7d"]
    +
    0.30 *
    baseline["avg_daily_14d"]
    +
    0.20 *
    baseline["avg_daily_30d"]
)


# ------------------------------------------------------------
# DEMAND TREND
# ------------------------------------------------------------

baseline["recent_vs_90d_ratio"] = np.where(
    baseline["avg_daily_90d"] > 0,

    baseline["avg_daily_7d"] /
    baseline["avg_daily_90d"],

    np.nan
)


def classify_trend(ratio):

    if pd.isna(ratio):

        return "No Historical Demand"

    if ratio >= 1.50:

        return "Strongly Increasing"

    elif ratio >= 1.15:

        return "Increasing"

    elif ratio <= 0.67:

        return "Strongly Decreasing"

    elif ratio <= 0.85:

        return "Decreasing"

    else:

        return "Stable"


baseline["baseline_trend"] = (
    baseline[
        "recent_vs_90d_ratio"
    ]
    .apply(classify_trend)
)


# ------------------------------------------------------------
# FORECASTABILITY
# ------------------------------------------------------------

def classify_forecastability(row):

    activity = row[
        "activity_rate"
    ]

    if activity == 0:

        return "Not Forecastable"

    elif activity < 0.05:

        return "Very Difficult"

    elif activity < 0.10:

        return "Difficult"

    elif activity < 0.30:

        return "Moderate"

    else:

        return "Forecastable"


baseline[
    "forecastability"
] = baseline.apply(
    classify_forecastability,
    axis=1
)


# ------------------------------------------------------------
# EXPECTED 30-DAY DEMAND
# ------------------------------------------------------------

baseline[
    "expected_30d_demand"
] = (
    baseline[
        "forecast_weighted"
    ] * 30
)


# ------------------------------------------------------------
# EXPECTED 7-DAY DEMAND
# ------------------------------------------------------------

baseline[
    "expected_7d_demand"
] = (
    baseline[
        "forecast_weighted"
    ] * 7
)


# ------------------------------------------------------------
# PRINT BASELINE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE FORECAST SUMMARY")
print("=" * 70)


print(
    "\nAverage daily demand by method:"
)

print(
    "7-day:",
    baseline[
        "forecast_ma_7"
    ].mean()
)

print(
    "14-day:",
    baseline[
        "forecast_ma_14"
    ].mean()
)

print(
    "30-day:",
    baseline[
        "forecast_ma_30"
    ].mean()
)

print(
    "90-day:",
    baseline[
        "forecast_ma_90"
    ].mean()
)

print(
    "Weighted:",
    baseline[
        "forecast_weighted"
    ].mean()
)


# ------------------------------------------------------------
# TREND DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nBaseline trend distribution:"
)

print(
    baseline[
        "baseline_trend"
    ].value_counts()
)


# ------------------------------------------------------------
# FORECASTABILITY DISTRIBUTION
# ------------------------------------------------------------

print(
    "\nForecastability distribution:"
)

print(
    baseline[
        "forecastability"
    ].value_counts()
)


# ------------------------------------------------------------
# TOP DEMAND ITEMS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 STORE-SKU COMBINATIONS BY 30-DAY DEMAND")
print("=" * 70)


top_demand = (
    baseline
    .sort_values(
        "units_30d",
        ascending=False
    )
    .head(20)
)


print(
    top_demand[
        [
            "store_id",
            "sku_id",
            "units_7d",
            "units_14d",
            "units_30d",
            "units_90d",
            "avg_daily_30d",
            "forecast_weighted",
            "baseline_trend",
            "forecastability"
        ]
    ]
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# MOST INTERMITTENT ACTIVE ITEMS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MOST INTERMITTENT ACTIVE ITEMS")
print("=" * 70)


intermittent = (
    baseline[
        baseline["active_days"] > 0
    ]
    .sort_values(
        [
            "activity_rate",
            "total_units"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(20)
)


print(
    intermittent[
        [
            "store_id",
            "sku_id",
            "total_units",
            "active_days",
            "zero_days",
            "activity_rate",
            "avg_positive_demand",
            "max_daily_demand",
            "forecastability"
        ]
    ]
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE BASELINE DATASET
# ------------------------------------------------------------

output_file = (
    FORECAST_PATH /
    "demand_forecast_baseline.csv"
)

baseline.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    "\nBaseline shape:",
    baseline.shape
)

print(
    "Missing forecast values:",
    baseline[
        "forecast_weighted"
    ].isna().sum()
)

print(
    "Missing trend:",
    baseline[
        "baseline_trend"
    ].isna().sum()
)

print(
    "Missing forecastability:",
    baseline[
        "forecastability"
    ].isna().sum()
)

print(
    "Negative forecast values:",
    (
        baseline[
            "forecast_weighted"
        ] < 0
    ).sum()
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 5.2 COMPLETED")
print("=" * 70)

print(
    "\nBaseline forecast saved to:"
)

print(
    output_file
)