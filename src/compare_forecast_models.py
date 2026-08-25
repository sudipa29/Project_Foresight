# ============================================================
# PROJECT FORESIGHT
# PHASE 6.2 - FORECAST MODEL COMPARISON
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

VALIDATION_PATH = (
    FORECAST_PATH
    / "validation"
)

VALIDATION_PATH.mkdir(
    parents=True,
    exist_ok=True
)

HISTORICAL_FILE = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "forecast_demand_daily.csv"
)


# ============================================================
# FILES
# ============================================================

LIGHTGBM_FILES = {
    30: FORECAST_PATH / "future_30_day_forecast.csv",
    60: FORECAST_PATH / "future_60_day_forecast.csv",
    90: FORECAST_PATH / "future_90_day_forecast.csv"
}

INTERMITTENT_FILES = {
    30: INTERMITTENT_PATH
    / "intermittent_future_30_day_forecast.csv",

    60: INTERMITTENT_PATH
    / "intermittent_future_60_day_forecast.csv",

    90: INTERMITTENT_PATH
    / "intermittent_future_90_day_forecast.csv"
}


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_COMPARISON_FILE = (
    VALIDATION_PATH
    / "forecast_model_comparison.csv"
)

STORE_SKU_COMPARISON_FILE = (
    VALIDATION_PATH
    / "forecast_model_store_sku_comparison.csv"
)

DAILY_COMPARISON_FILE = (
    VALIDATION_PATH
    / "forecast_model_daily_comparison.csv"
)

MODEL_DECISION_FILE = (
    VALIDATION_PATH
    / "forecast_model_decision.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.2 - FORECAST MODEL COMPARISON")
print("=" * 70)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print()
print("=" * 70)
print("LOADING HISTORICAL DEMAND")
print("=" * 70)

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

historical_max_date = historical["date"].max()

print(
    "Historical rows:",
    len(historical)
)

print(
    "Historical last date:",
    historical_max_date.date()
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print()
print("=" * 70)
print("COMPARING FORECAST MODELS")
print("=" * 70)


comparison_rows = []


for horizon in [30, 60, 90]:

    print()
    print("-" * 70)
    print(f"HORIZON: {horizon} DAYS")
    print("-" * 70)

    # --------------------------------------------------------
    # Load LightGBM
    # --------------------------------------------------------

    lightgbm = pd.read_csv(
        LIGHTGBM_FILES[horizon],
        usecols=[
            "store_id",
            "sku_id",
            "date",
            "forecast_units"
        ]
    )

    lightgbm["date"] = pd.to_datetime(
        lightgbm["date"]
    )

    # --------------------------------------------------------
    # Load intermittent
    # --------------------------------------------------------

    intermittent = pd.read_csv(
        INTERMITTENT_FILES[horizon],
        usecols=[
            "store_id",
            "sku_id",
            "date",
            "forecast_units"
        ]
    )

    intermittent["date"] = pd.to_datetime(
        intermittent["date"]
    )

    # --------------------------------------------------------
    # Historical period
    # --------------------------------------------------------

    historical_start = (
        historical_max_date
        - pd.Timedelta(days=horizon - 1)
    )

    historical_period = historical[
        historical["date"] >= historical_start
    ]

    historical_total = (
        historical_period["units_sold"]
        .sum()
    )

    # --------------------------------------------------------
    # Forecast totals
    # --------------------------------------------------------

    lightgbm_total = (
        lightgbm["forecast_units"]
        .sum()
    )

    intermittent_total = (
        intermittent["forecast_units"]
        .sum()
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    lightgbm_difference = (
        lightgbm_total
        - historical_total
    )

    intermittent_difference = (
        intermittent_total
        - historical_total
    )

    lightgbm_pct = (
        lightgbm_difference
        / historical_total
        * 100
    )

    intermittent_pct = (
        intermittent_difference
        / historical_total
        * 100
    )

    lightgbm_ratio = (
        lightgbm_total
        / historical_total
    )

    intermittent_ratio = (
        intermittent_total
        / historical_total
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Historical demand:       {historical_total:,.2f}"
    )

    print(
        f"LightGBM forecast:       {lightgbm_total:,.2f}"
    )

    print(
        f"Intermittent forecast:   {intermittent_total:,.2f}"
    )

    print()

    print(
        f"LightGBM vs historical:  "
        f"{lightgbm_pct:+.2f}%"
    )

    print(
        f"Intermittent vs historical:"
        f" {intermittent_pct:+.2f}%"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    comparison_rows.append(
        {
            "horizon_days": horizon,
            "historical_demand": historical_total,
            "lightgbm_forecast": lightgbm_total,
            "intermittent_forecast": intermittent_total,
            "lightgbm_difference": lightgbm_difference,
            "intermittent_difference": intermittent_difference,
            "lightgbm_difference_pct": lightgbm_pct,
            "intermittent_difference_pct": intermittent_pct,
            "lightgbm_ratio": lightgbm_ratio,
            "intermittent_ratio": intermittent_ratio
        }
    )

    del lightgbm
    del intermittent
    del historical_period


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame(
    comparison_rows
)

model_comparison.to_csv(
    MODEL_COMPARISON_FILE,
    index=False
)


# ============================================================
# DISPLAY MODEL COMPARISON
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

print(
    model_comparison.to_string(
        index=False
    )
)

print()
print(
    "Saved:"
)

print(
    MODEL_COMPARISON_FILE
)


# ============================================================
# STORE-SKU COMPARISON - 30 DAYS
# ============================================================

print()
print("=" * 70)
print("STORE-SKU MODEL COMPARISON - 30 DAYS")
print("=" * 70)


lightgbm_30 = pd.read_csv(
    LIGHTGBM_FILES[30],
    usecols=[
        "store_id",
        "sku_id",
        "forecast_units"
    ]
)

intermittent_30 = pd.read_csv(
    INTERMITTENT_FILES[30],
    usecols=[
        "store_id",
        "sku_id",
        "forecast_units"
    ]
)


# ------------------------------------------------------------
# Historical 30-day demand
# ------------------------------------------------------------

historical_start = (
    historical_max_date
    - pd.Timedelta(days=29)
)

historical_30 = historical[
    historical["date"] >= historical_start
]

historical_30 = (
    historical_30
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )["units_sold"]
    .sum()
    .rename(
        columns={
            "units_sold":
                "historical_30d"
        }
    )
)


# ------------------------------------------------------------
# Aggregate forecasts
# ------------------------------------------------------------

lightgbm_30 = (
    lightgbm_30
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
                "lightgbm_30d"
        }
    )
)


intermittent_30 = (
    intermittent_30
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
                "intermittent_30d"
        }
    )
)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

store_sku = (
    historical_30
    .merge(
        lightgbm_30,
        on=[
            "store_id",
            "sku_id"
        ],
        how="outer"
    )
    .merge(
        intermittent_30,
        on=[
            "store_id",
            "sku_id"
        ],
        how="outer"
    )
    .fillna(0)
)


# ------------------------------------------------------------
# Differences
# ------------------------------------------------------------

store_sku[
    "lightgbm_difference"
] = (
    store_sku["lightgbm_30d"]
    - store_sku["historical_30d"]
)


store_sku[
    "intermittent_difference"
] = (
    store_sku["intermittent_30d"]
    - store_sku["historical_30d"]
)


store_sku[
    "lightgbm_abs_error"
] = (
    store_sku["lightgbm_difference"]
    .abs()
)


store_sku[
    "intermittent_abs_error"
] = (
    store_sku["intermittent_difference"]
    .abs()
)


# ------------------------------------------------------------
# Historical activity flags
# ------------------------------------------------------------

store_sku[
    "historical_zero_demand"
] = (
    store_sku["historical_30d"] == 0
)


store_sku[
    "historical_positive_demand"
] = (
    store_sku["historical_30d"] > 0
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

store_sku = (
    store_sku
    .sort_values(
        "historical_30d",
        ascending=False
    )
)


store_sku.to_csv(
    STORE_SKU_COMPARISON_FILE,
    index=False
)


# ============================================================
# STORE-SKU METRICS
# ============================================================

print()
print(
    "Total Store-SKU:",
    len(store_sku)
)

print(
    "Historical zero-demand:",
    (
        store_sku[
            "historical_30d"
        ] == 0
    ).sum()
)

print(
    "Historical positive-demand:",
    (
        store_sku[
            "historical_30d"
        ] > 0
    ).sum()
)

print()

print(
    "LightGBM MAE:",
    store_sku[
        "lightgbm_abs_error"
    ].mean()
)

print(
    "Intermittent MAE:",
    store_sku[
        "intermittent_abs_error"
    ].mean()
)


# ============================================================
# ZERO-DEMAND ANALYSIS
# ============================================================

zero_demand = store_sku[
    store_sku["historical_30d"] == 0
]

print()
print(
    "Historical zero-demand Store-SKU:",
    len(zero_demand)
)

print(
    "Average LightGBM forecast for zero-demand:",
    zero_demand[
        "lightgbm_30d"
    ].mean()
)

print(
    "Average Intermittent forecast for zero-demand:",
    zero_demand[
        "intermittent_30d"
    ].mean()
)

print()
print(
    "Total LightGBM forecast on zero-demand:",
    zero_demand[
        "lightgbm_30d"
    ].sum()
)

print(
    "Total Intermittent forecast on zero-demand:",
    zero_demand[
        "intermittent_30d"
    ].sum()
)


# ============================================================
# DAILY COMPARISON - 30 DAYS
# ============================================================

print()
print("=" * 70)
print("CREATING DAILY MODEL COMPARISON")
print("=" * 70)


lgb_daily = pd.read_csv(
    LIGHTGBM_FILES[30],
    usecols=[
        "date",
        "forecast_units"
    ]
)

int_daily = pd.read_csv(
    INTERMITTENT_FILES[30],
    usecols=[
        "date",
        "forecast_units"
    ]
)


lgb_daily = (
    lgb_daily
    .groupby("date", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
                "lightgbm_daily"
        }
    )
)


int_daily = (
    int_daily
    .groupby("date", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
                "intermittent_daily"
        }
    )
)


daily_comparison = (
    lgb_daily
    .merge(
        int_daily,
        on="date",
        how="outer"
    )
    .sort_values("date")
)


daily_comparison[
    "difference"
] = (
    daily_comparison["lightgbm_daily"]
    - daily_comparison["intermittent_daily"]
)


daily_comparison[
    "lightgbm_to_intermittent_ratio"
] = (
    daily_comparison["lightgbm_daily"]
    /
    daily_comparison["intermittent_daily"]
)


daily_comparison.to_csv(
    DAILY_COMPARISON_FILE,
    index=False
)


print()
print(
    daily_comparison.to_string(
        index=False
    )
)

print()
print(
    "Saved:"
)

print(
    DAILY_COMPARISON_FILE
)


# ============================================================
# MODEL DECISION
# ============================================================

print()
print("=" * 70)
print("MODEL DECISION")
print("=" * 70)


model_comparison[
    "lightgbm_abs_pct_error"
] = (
    model_comparison[
        "lightgbm_difference_pct"
    ].abs()
)


model_comparison[
    "intermittent_abs_pct_error"
] = (
    model_comparison[
        "intermittent_difference_pct"
    ].abs()
)


lightgbm_score = (
    model_comparison[
        "lightgbm_abs_pct_error"
    ].mean()
)


intermittent_score = (
    model_comparison[
        "intermittent_abs_pct_error"
    ].mean()
)


print(
    f"Average LightGBM absolute % difference:"
    f" {lightgbm_score:.2f}%"
)

print(
    f"Average Intermittent absolute % difference:"
    f" {intermittent_score:.2f}%"
)


if intermittent_score < lightgbm_score:

    selected_model = "INTERMITTENT"

else:

    selected_model = "LIGHTGBM"


decision = pd.DataFrame(
    [
        {
            "selected_model":
                selected_model,

            "lightgbm_average_abs_pct_error":
                lightgbm_score,

            "intermittent_average_abs_pct_error":
                intermittent_score,

            "reason":
                "Model selected based on lower average absolute deviation from recent historical demand."
        }
    ]
)


decision.to_csv(
    MODEL_DECISION_FILE,
    index=False
)


print()
print(
    "Preliminary selected model:",
    selected_model
)

print()
print(
    "Decision file saved:"
)

print(
    MODEL_DECISION_FILE
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PHASE 6.2 COMPLETED")
print("=" * 70)