# ============================================================
# PROJECT FORESIGHT
# Phase 5.3 - Intermittent Demand Forecasting
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

PROCESSED_PATH = BASE_PATH / "data" / "processed"

FORECASTING_PATH = (
    PROCESSED_PATH / "forecasting"
)

INPUT_PATH = (
    FORECASTING_PATH /
    "forecast_demand_daily.csv"
)

OUTPUT_PATH = (
    FORECASTING_PATH /
    "intermittent_demand_forecast.csv"
)

SUMMARY_PATH = (
    FORECASTING_PATH /
    "intermittent_demand_summary.csv"
)

FORECASTING_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - INTERMITTENT DEMAND FORECASTING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading forecast demand dataset...")

daily = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print("Dataset shape:", daily.shape)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)

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
    if col not in daily.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns found successfully.")

# ------------------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------------------

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce"
)

print(
    "\nDate range:",
    daily["date"].min(),
    "to",
    daily["date"].max()
)

# ------------------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------------------

print(
    "Stores:",
    daily["store_id"].nunique()
)

print(
    "SKUs:",
    daily["sku_id"].nunique()
)

print(
    "Store-SKU combinations:",
    daily[
        ["store_id", "sku_id"]
    ].drop_duplicates().shape[0]
)

print(
    "Missing dates:",
    daily["date"].isna().sum()
)

print(
    "Missing demand values:",
    daily["units_sold"].isna().sum()
)

print(
    "Negative demand rows:",
    (
        daily["units_sold"] < 0
    ).sum()
)


# ============================================================
# STANDARDIZE DEMAND
# ============================================================

daily["units_sold"] = pd.to_numeric(
    daily["units_sold"],
    errors="coerce"
)

daily["units_sold"] = (
    daily["units_sold"]
    .fillna(0)
)

daily["units_sold"] = (
    daily["units_sold"]
    .clip(lower=0)
)


# ============================================================
# GLOBAL DEMAND PROFILE
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL INTERMITTENT DEMAND PROFILE")
print("=" * 70)

total_rows = len(daily)

zero_demand_rows = (
    daily["units_sold"] == 0
).sum()

positive_demand_rows = (
    daily["units_sold"] > 0
).sum()

zero_percentage = (
    zero_demand_rows /
    total_rows *
    100
)

positive_percentage = (
    positive_demand_rows /
    total_rows *
    100
)

print(
    "\nTotal observations:",
    total_rows
)

print(
    "Zero-demand observations:",
    zero_demand_rows
)

print(
    "Positive-demand observations:",
    positive_demand_rows
)

print(
    f"Zero-demand percentage: "
    f"{zero_percentage:.2f}%"
)

print(
    f"Positive-demand percentage: "
    f"{positive_percentage:.2f}%"
)


# ============================================================
# STORE-SKU LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STORE-SKU INTERMITTENT DEMAND PROFILE")
print("=" * 70)


summary = (
    daily
    .groupby(
        ["store_id", "sku_id"],
        as_index=False
    )
    .agg(
        total_units=(
            "units_sold",
            "sum"
        ),

        active_days=(
            "units_sold",
            lambda x: (x > 0).sum()
        ),

        total_days=(
            "units_sold",
            "count"
        ),

        max_daily_demand=(
            "units_sold",
            "max"
        ),

        avg_daily_demand=(
            "units_sold",
            "mean"
        )
    )
)


# ============================================================
# ZERO DAYS
# ============================================================

summary["zero_days"] = (
    summary["total_days"] -
    summary["active_days"]
)


# ============================================================
# ACTIVITY RATE
# ============================================================

summary["activity_rate"] = np.where(
    summary["total_days"] > 0,

    summary["active_days"] /
    summary["total_days"],

    0
)


# ============================================================
# AVERAGE POSITIVE DEMAND
# ============================================================

summary["avg_positive_demand"] = np.where(
    summary["active_days"] > 0,

    summary["total_units"] /
    summary["active_days"],

    0
)


# ============================================================
# INTERMITTENCY MEASURES
# ============================================================

# ------------------------------------------------------------
# INTER-DEMAND INTERVAL
# ------------------------------------------------------------

summary["average_demand_interval"] = np.where(
    summary["active_days"] > 0,

    summary["total_days"] /
    summary["active_days"],

    np.inf
)


# ------------------------------------------------------------
# COEFFICIENT OF VARIATION
# ------------------------------------------------------------

positive_cv = (
    daily[
        daily["units_sold"] > 0
    ]
    .groupby(
        ["store_id", "sku_id"]
    )["units_sold"]
    .agg(
        positive_std="std",
        positive_mean="mean"
    )
    .reset_index()
)

summary = summary.merge(
    positive_cv,
    on=[
        "store_id",
        "sku_id"
    ],
    how="left"
)

summary["positive_std"] = (
    summary["positive_std"]
    .fillna(0)
)

summary["positive_mean"] = (
    summary["positive_mean"]
    .fillna(0)
)

summary["cv_positive"] = np.where(
    summary["positive_mean"] > 0,

    summary["positive_std"] /
    summary["positive_mean"],

    0
)


# ============================================================
# DEMAND CLASSIFICATION
# ============================================================

def classify_intermitency(row):

    activity = row["activity_rate"]

    cv = row["cv_positive"]

    if activity == 0:

        return "No Demand"

    elif activity < 0.05:

        if cv > 1.0:
            return "Highly Intermittent"

        return "Intermittent"

    elif activity < 0.20:

        if cv > 1.0:
            return "Intermittent Variable"

        return "Intermittent"

    else:

        if cv > 1.0:
            return "Variable"

        return "Regular"


summary["intermittency_class"] = (
    summary.apply(
        classify_intermitency,
        axis=1
    )
)


# ============================================================
# FORECASTABILITY
# ============================================================

def classify_forecastability(row):

    activity = row["activity_rate"]

    cv = row["cv_positive"]

    if activity == 0:

        return "No Demand"

    if activity < 0.03:

        return "Very Difficult"

    if activity < 0.05:

        if cv > 1.0:
            return "Very Difficult"

        return "Difficult"

    if activity < 0.10:

        if cv > 1.5:
            return "Very Difficult"

        return "Difficult"

    if cv > 1.5:

        return "Difficult"

    return "Moderate"


summary["forecastability"] = (
    summary.apply(
        classify_forecastability,
        axis=1
    )
)


# ============================================================
# RECENT DEMAND WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING INTERMITTENT DEMAND FORECASTS")
print("=" * 70)


reference_date = daily["date"].max()

print(
    "\nReference date:",
    reference_date.date()
)


# ------------------------------------------------------------
# CREATE DAYS FROM REFERENCE
# ------------------------------------------------------------

daily["days_from_reference"] = (
    reference_date -
    daily["date"]
).dt.days


# ============================================================
# FORECAST FUNCTION
# ============================================================

def calculate_window_forecast(
    df,
    days
):

    window = df[
        (
            df["days_from_reference"] >= 0
        )
        &
        (
            df["days_from_reference"] < days
        )
    ]

    result = (
        window
        .groupby(
            ["store_id", "sku_id"],
            as_index=False
        )["units_sold"]
        .sum()
    )

    result[
        f"units_{days}d"
    ] = result["units_sold"]

    result[
        f"avg_daily_{days}d"
    ] = (
        result["units_sold"] /
        days
    )

    return result[
        [
            "store_id",
            "sku_id",
            f"units_{days}d",
            f"avg_daily_{days}d"
        ]
    ]


# ============================================================
# WINDOW FORECASTS
# ============================================================

f7 = calculate_window_forecast(
    daily,
    7
)

f14 = calculate_window_forecast(
    daily,
    14
)

f30 = calculate_window_forecast(
    daily,
    30
)

f60 = calculate_window_forecast(
    daily,
    60
)

f90 = calculate_window_forecast(
    daily,
    90
)


# ============================================================
# MERGE FORECAST WINDOWS
# ============================================================

forecast = summary.copy()


for window_df in [
    f7,
    f14,
    f30,
    f60,
    f90
]:

    forecast = forecast.merge(
        window_df,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )


# ============================================================
# FILL MISSING VALUES
# ============================================================

window_columns = [
    "units_7d",
    "avg_daily_7d",

    "units_14d",
    "avg_daily_14d",

    "units_30d",
    "avg_daily_30d",

    "units_60d",
    "avg_daily_60d",

    "units_90d",
    "avg_daily_90d"
]

forecast[
    window_columns
] = forecast[
    window_columns
].fillna(0)


# ============================================================
# CROSTON-STYLE BASELINE
# ============================================================

# For intermittent demand, a simple Croston-style
# estimate is based on:
#
# Average positive demand
# divided by
# Average interval between demand events

forecast["croston_forecast"] = np.where(

    forecast["average_demand_interval"] > 0,

    forecast["avg_positive_demand"] /
    forecast["average_demand_interval"],

    0
)


# ============================================================
# RECENT WEIGHTED FORECAST
# ============================================================

forecast["recent_weighted_forecast"] = (

    forecast["avg_daily_7d"] * 0.40

    +

    forecast["avg_daily_14d"] * 0.25

    +

    forecast["avg_daily_30d"] * 0.20

    +

    forecast["avg_daily_60d"] * 0.10

    +

    forecast["avg_daily_90d"] * 0.05
)


# ============================================================
# INTERMITTENT DEMAND FORECAST
# ============================================================

forecast["intermittent_forecast"] = np.where(

    forecast["intermittency_class"].isin(
        [
            "Highly Intermittent",
            "Intermittent",
            "Intermittent Variable"
        ]
    ),

    (
        forecast["croston_forecast"] * 0.60
        +
        forecast["recent_weighted_forecast"] * 0.40
    ),

    forecast["recent_weighted_forecast"]
)


# ============================================================
# RECENT DEMAND SIGNAL
# ============================================================

forecast["recent_demand_signal"] = np.select(

    [

        forecast["units_7d"] > 0,

        forecast["units_14d"] > 0,

        forecast["units_30d"] > 0,

        forecast["units_90d"] > 0

    ],

    [

        "Very Recent Demand",

        "Recent Demand",

        "Historical Recent Demand",

        "Old Demand"

    ],

    default="No Observed Demand"
)


# ============================================================
# FORECAST CONFIDENCE
# ============================================================

def forecast_confidence(row):

    forecastability = (
        row["forecastability"]
    )

    activity = (
        row["activity_rate"]
    )

    if forecastability == "No Demand":

        return "None"

    if forecastability == "Very Difficult":

        return "Low"

    if forecastability == "Difficult":

        return "Low-Medium"

    if activity >= 0.10:

        return "Medium"

    return "Low-Medium"


forecast["forecast_confidence"] = (
    forecast.apply(
        forecast_confidence,
        axis=1
    )
)


# ============================================================
# FORECAST HORIZONS
# ============================================================

forecast["forecast_7d_units"] = (
    forecast["intermittent_forecast"] *
    7
)

forecast["forecast_30d_units"] = (
    forecast["intermittent_forecast"] *
    30
)

forecast["forecast_90d_units"] = (
    forecast["intermittent_forecast"] *
    90
)


# ============================================================
# ROUNDING
# ============================================================

numeric_columns = [

    "activity_rate",

    "avg_daily_demand",

    "avg_positive_demand",

    "average_demand_interval",

    "cv_positive",

    "croston_forecast",

    "recent_weighted_forecast",

    "intermittent_forecast",

    "forecast_7d_units",

    "forecast_30d_units",

    "forecast_90d_units"

]

for column in numeric_columns:

    if column in forecast.columns:

        forecast[column] = (
            forecast[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .round(4)
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INTERMITTENCY CLASS DISTRIBUTION")
print("=" * 70)

print(
    forecast[
        "intermittency_class"
    ].value_counts()
)


print("\n" + "=" * 70)
print("FORECASTABILITY DISTRIBUTION")
print("=" * 70)

print(
    forecast[
        "forecastability"
    ].value_counts()
)


print("\n" + "=" * 70)
print("RECENT DEMAND SIGNAL DISTRIBUTION")
print("=" * 70)

print(
    forecast[
        "recent_demand_signal"
    ].value_counts()
)


# ============================================================
# FORECAST SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("INTERMITTENT FORECAST SUMMARY")
print("=" * 70)

print(
    forecast[
        "intermittent_forecast"
    ].describe()
)


# ============================================================
# TOP FORECAST ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 STORE-SKU ITEMS BY INTERMITTENT FORECAST")
print("=" * 70)

top_forecast = (
    forecast
    .sort_values(
        "intermittent_forecast",
        ascending=False
    )
    .head(20)
)

print(
    top_forecast[
        [
            "store_id",
            "sku_id",
            "total_units",
            "active_days",
            "activity_rate",
            "avg_positive_demand",
            "average_demand_interval",
            "units_7d",
            "units_30d",
            "units_90d",
            "croston_forecast",
            "recent_weighted_forecast",
            "intermittent_forecast",
            "forecastability",
            "forecast_confidence"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# MOST INTERMITTENT ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 MOST INTERMITTENT ACTIVE ITEMS")
print("=" * 70)

intermittent_items = (
    forecast[
        forecast["active_days"] > 0
    ]
    .sort_values(
        [
            "activity_rate",
            "average_demand_interval"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(20)
)

print(
    intermittent_items[
        [
            "store_id",
            "sku_id",
            "total_units",
            "active_days",
            "zero_days",
            "activity_rate",
            "avg_positive_demand",
            "average_demand_interval",
            "cv_positive",
            "intermittency_class",
            "forecastability",
            "intermittent_forecast"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# NO-DEMAND ITEMS
# ============================================================

print("\n" + "=" * 70)
print("NO-DEMAND STORE-SKU ITEMS")
print("=" * 70)

no_demand = forecast[
    forecast["active_days"] == 0
]

print(
    "Number of no-demand items:",
    len(no_demand)
)

print(
    no_demand[
        [
            "store_id",
            "sku_id",
            "total_units",
            "active_days",
            "zero_days",
            "activity_rate",
            "intermittency_class",
            "forecastability",
            "intermittent_forecast"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    "Forecast shape:",
    forecast.shape
)

print(
    "Missing intermittent forecasts:",
    forecast[
        "intermittent_forecast"
    ].isna().sum()
)

print(
    "Negative forecasts:",
    (
        forecast[
            "intermittent_forecast"
        ] < 0
    ).sum()
)

print(
    "Missing forecastability:",
    forecast[
        "forecastability"
    ].isna().sum()
)

print(
    "Missing intermittency class:",
    forecast[
        "intermittency_class"
    ].isna().sum()
)

print(
    "Missing forecast confidence:",
    forecast[
        "forecast_confidence"
    ].isna().sum()
)


# ============================================================
# SAVE OUTPUT
# ============================================================

forecast.to_csv(
    OUTPUT_PATH,
    index=False
)

forecast[
    [
        "store_id",
        "sku_id",
        "total_units",
        "active_days",
        "zero_days",
        "activity_rate",
        "avg_positive_demand",
        "average_demand_interval",
        "cv_positive",
        "intermittency_class",
        "forecastability",
        "units_7d",
        "units_14d",
        "units_30d",
        "units_60d",
        "units_90d",
        "croston_forecast",
        "recent_weighted_forecast",
        "intermittent_forecast",
        "forecast_7d_units",
        "forecast_30d_units",
        "forecast_90d_units",
        "recent_demand_signal",
        "forecast_confidence"
    ]
].to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5.3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nIntermittent forecast saved to:")
print(OUTPUT_PATH)

print("\nIntermittent forecast summary saved to:")
print(SUMMARY_PATH)

print("\n" + "=" * 70)
print("NEXT PHASE: FORECAST MODEL EVALUATION")
print("=" * 70)