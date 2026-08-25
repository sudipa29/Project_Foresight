# ============================================================
# PROJECT FORESIGHT
# Phase 6.1 - Future Forecast Validation & Comparison
#
# Purpose:
# Compare:
# 1. LightGBM future forecast
# 2. Corrected intermittent forecast
# 3. Historical demand
#
# IMPORTANT:
# This phase validates forecast scale BEFORE inventory integration.
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
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
)

INTERMITTENT_PATH = (
    FORECAST_PATH
    / "intermittent_corrected"
)

HISTORICAL_FILE = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "forecast_demand_daily.csv"
)


# ============================================================
# FILE DISCOVERY
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.1 - FUTURE FORECAST VALIDATION")
print("=" * 70)


print()
print("=" * 70)
print("CHECKING FORECAST FILES")
print("=" * 70)


# ------------------------------------------------------------
# Search for LightGBM forecast files
# ------------------------------------------------------------

lightgbm_files = list(
    FORECAST_PATH.glob("*.csv")
)

print()
print("Forecast directory:")
print(FORECAST_PATH)

print()
print("CSV files found:")

for f in lightgbm_files:
    print(" -", f.name)


# ============================================================
# FIND LIGHTGBM FILES
# ============================================================

candidate_30 = [
    f for f in lightgbm_files
    if "30" in f.name.lower()
    and "forecast" in f.name.lower()
]

candidate_60 = [
    f for f in lightgbm_files
    if "60" in f.name.lower()
    and "forecast" in f.name.lower()
]

candidate_90 = [
    f for f in lightgbm_files
    if "90" in f.name.lower()
    and "forecast" in f.name.lower()
]


print()
print("Possible 30-day LightGBM files:")

for f in candidate_30:
    print(" -", f.name)

print()
print("Possible 60-day LightGBM files:")

for f in candidate_60:
    print(" -", f.name)

print()
print("Possible 90-day LightGBM files:")

for f in candidate_90:
    print(" -", f.name)


# ============================================================
# INTERMITTENT FILES
# ============================================================

intermittent_30 = (
    INTERMITTENT_PATH
    / "intermittent_future_30_day_forecast.csv"
)

intermittent_60 = (
    INTERMITTENT_PATH
    / "intermittent_future_60_day_forecast.csv"
)

intermittent_90 = (
    INTERMITTENT_PATH
    / "intermittent_future_90_day_forecast.csv"
)


print()
print("=" * 70)
print("CHECKING INTERMITTENT FORECASTS")
print("=" * 70)

for f in [
    intermittent_30,
    intermittent_60,
    intermittent_90
]:

    print()

    if f.exists():
        print("FOUND:", f)
    else:
        print("MISSING:", f)


# ============================================================
# LOAD INTERMITTENT FORECASTS
# ============================================================

print()
print("=" * 70)
print("LOADING CORRECTED INTERMITTENT FORECASTS")
print("=" * 70)


int30 = pd.read_csv(
    intermittent_30
)

int60 = pd.read_csv(
    intermittent_60
)

int90 = pd.read_csv(
    intermittent_90
)


print()
print("30-day shape:", int30.shape)
print("60-day shape:", int60.shape)
print("90-day shape:", int90.shape)


print()
print("30-day total:",
      int30["forecast_units"].sum())

print("60-day total:",
      int60["forecast_units"].sum())

print("90-day total:",
      int90["forecast_units"].sum())


# ============================================================
# CHECK DAILY TOTALS
# ============================================================

print()
print("=" * 70)
print("INTERMITTENT DAILY FORECAST PROFILE")
print("=" * 70)


daily_int30 = (
    int30
    .groupby("date")["forecast_units"]
    .sum()
)


print()
print("30-day average daily forecast:",
      daily_int30.mean())

print("30-day minimum daily forecast:",
      daily_int30.min())

print("30-day maximum daily forecast:",
      daily_int30.max())


# ============================================================
# CHECK STORE-SKU DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("STORE-SKU FORECAST DISTRIBUTION")
print("=" * 70)


item_int30 = (
    int30
    .groupby(
        ["store_id", "sku_id"],
        as_index=False
    )
    .agg(
        forecast_30d_units=(
            "forecast_units",
            "sum"
        ),

        avg_daily_forecast=(
            "forecast_units",
            "mean"
        ),

        avg_occurrence_probability=(
            "occurrence_probability",
            "mean"
        ),

        avg_positive_quantity=(
            "positive_demand_quantity",
            "mean"
        )
    )
)


print()
print(
    "Store-SKU combinations:",
    len(item_int30)
)


print()
print(
    "Forecast 30-day statistics:"
)

print(
    item_int30[
        "forecast_30d_units"
    ].describe()
)


# ============================================================
# ZERO FORECAST CHECK
# ============================================================

zero_forecast = (
    item_int30[
        "forecast_30d_units"
    ] <= 0
).sum()


positive_forecast = (
    item_int30[
        "forecast_30d_units"
    ] > 0
).sum()


print()
print("Zero-forecast Store-SKU:",
      zero_forecast)

print("Positive-forecast Store-SKU:",
      positive_forecast)


# ============================================================
# HISTORICAL DEMAND
# ============================================================

print()
print("=" * 70)
print("HISTORICAL DEMAND VALIDATION")
print("=" * 70)


print()
print("Loading historical demand...")

historical = pd.read_csv(
    HISTORICAL_FILE,
    usecols=[
        "store_id",
        "sku_id",
        "date",
        "units_sold"
    ],
    dtype={
        "store_id": "int16",
        "sku_id": "int16",
        "date": "string",
        "units_sold": "float32"
    }
)


historical["date"] = pd.to_datetime(
    historical["date"]
)


print()
print("Historical shape:",
      historical.shape)

print(
    "Historical date range:",
    historical["date"].min(),
    "to",
    historical["date"].max()
)


# ============================================================
# LAST 30 DAYS ACTUAL DEMAND
# ============================================================

latest_date = historical["date"].max()

historical_30 = historical.loc[
    historical["date"]
    >=
    latest_date - pd.Timedelta(days=29)
].copy()


historical_30_total = (
    historical_30["units_sold"]
    .sum()
)


historical_30_daily = (
    historical_30
    .groupby("date")["units_sold"]
    .sum()
)


print()
print(
    "Historical last 30-day demand:",
    historical_30_total
)

print(
    "Historical average daily demand:",
    historical_30_daily.mean()
)


# ============================================================
# LAST 60 DAYS
# ============================================================

historical_60 = historical.loc[
    historical["date"]
    >=
    latest_date - pd.Timedelta(days=59)
].copy()


historical_60_total = (
    historical_60["units_sold"]
    .sum()
)


# ============================================================
# LAST 90 DAYS
# ============================================================

historical_90 = historical.loc[
    historical["date"]
    >=
    latest_date - pd.Timedelta(days=89)
].copy()


historical_90_total = (
    historical_90["units_sold"]
    .sum()
)


print(
    "Historical last 60-day demand:",
    historical_60_total
)

print(
    "Historical last 90-day demand:",
    historical_90_total
)


# ============================================================
# FORECAST VS HISTORICAL
# ============================================================

print()
print("=" * 70)
print("FORECAST VS HISTORICAL")
print("=" * 70)


comparison = pd.DataFrame({

    "Horizon": [
        "30 Days",
        "60 Days",
        "90 Days"
    ],

    "Historical_Demand": [
        historical_30_total,
        historical_60_total,
        historical_90_total
    ],

    "Intermittent_Forecast": [
        int30["forecast_units"].sum(),
        int60["forecast_units"].sum(),
        int90["forecast_units"].sum()
    ]
})


comparison[
    "Forecast_vs_Historical_Ratio"
] = (
    comparison[
        "Intermittent_Forecast"
    ]
    /
    comparison[
        "Historical_Demand"
    ]
)


comparison[
    "Forecast_Difference"
] = (
    comparison[
        "Intermittent_Forecast"
    ]
    -
    comparison[
        "Historical_Demand"
    ]
)


comparison[
    "Forecast_Difference_Pct"
] = (
    comparison[
        "Forecast_Difference"
    ]
    /
    comparison[
        "Historical_Demand"
    ]
    * 100
)


print()
print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# RECENT DEMAND BY STORE-SKU
# ============================================================

print()
print("=" * 70)
print("STORE-SKU HISTORICAL VS FORECAST")
print("=" * 70)


historical_item_30 = (
    historical_30
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )
    .agg(
        historical_30d_units=(
            "units_sold",
            "sum"
        )
    )
)


item_comparison = (
    item_int30
    .merge(
        historical_item_30,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )
)


item_comparison[
    "historical_30d_units"
] = (
    item_comparison[
        "historical_30d_units"
    ]
    .fillna(0)
)


item_comparison[
    "forecast_vs_historical_ratio"
] = np.where(
    item_comparison[
        "historical_30d_units"
    ] > 0,

    item_comparison[
        "forecast_30d_units"
    ]
    /
    item_comparison[
        "historical_30d_units"
    ],

    np.nan
)


# ============================================================
# SAVE COMPARISON
# ============================================================

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "validation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


comparison_file = (
    OUTPUT_DIR
    / "future_forecast_validation_summary.csv"
)


item_file = (
    OUTPUT_DIR
    / "future_forecast_store_sku_comparison.csv"
)


comparison.to_csv(
    comparison_file,
    index=False
)


item_comparison.to_csv(
    item_file,
    index=False
)


print()
print("=" * 70)
print("VALIDATION FILES SAVED")
print("=" * 70)

print()
print(comparison_file)

print()
print(item_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PHASE 6.1 VALIDATION COMPLETED")
print("=" * 70)

print()
print(
    "Intermittent 30-day forecast:",
    round(
        int30["forecast_units"].sum(),
        2
    )
)

print(
    "Historical 30-day demand:",
    round(
        historical_30_total,
        2
    )
)

print()
print(
    "Next step:"
)

print(
    "Compare this forecast against the previous LightGBM forecast."
)

print(
    "DO NOT proceed to inventory recommendations until "
    "forecast scale has been validated."
)