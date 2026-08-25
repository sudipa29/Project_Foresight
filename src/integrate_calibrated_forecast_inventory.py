# ============================================================
# PROJECT FORESIGHT
# PHASE 7.1 - CALIBRATED FORECAST + INVENTORY INTEGRATION
#
# Production Forecast:
# CALIBRATED INTERMITTENT MODEL
#
# Forecast Horizons:
# 30 / 60 / 90 Days
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
    / "calibrated"
)

INVENTORY_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "inventory_analysis"
    / "inventory_diagnostic_analysis.csv"
)

OUTPUT_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "integration"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FORECAST FILES
# ============================================================

FORECAST_FILES = {
    30: FORECAST_PATH
    / "calibrated_intermittent_30_day_forecast.csv",

    60: FORECAST_PATH
    / "calibrated_intermittent_60_day_forecast.csv",

    90: FORECAST_PATH
    / "calibrated_intermittent_90_day_forecast.csv"
}


OUTPUT_FILE = (
    OUTPUT_PATH
    / "calibrated_forecast_inventory_integrated.csv"
)

SUMMARY_FILE = (
    OUTPUT_PATH
    / "calibrated_forecast_inventory_integration_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 7.1 - CALIBRATED FORECAST + INVENTORY INTEGRATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for horizon, path in FORECAST_FILES.items():

    print()
    print(f"Checking {horizon}-day forecast:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {horizon}-day forecast:\n{path}"
        )

    print("FOUND")


print()
print("Checking inventory file:")
print(INVENTORY_PATH)

if not INVENTORY_PATH.exists():

    raise FileNotFoundError(
        f"Inventory file not found:\n{INVENTORY_PATH}"
    )

print("FOUND")


# ============================================================
# LOAD INVENTORY
# ============================================================

print()
print("=" * 70)
print("LOADING INVENTORY DATA")
print("=" * 70)

inventory = pd.read_csv(
    INVENTORY_PATH
)

print()
print("Inventory shape:", inventory.shape)

print()
print("Inventory columns:")
print(inventory.columns.tolist())


# ============================================================
# REQUIRED INVENTORY COLUMNS
# ============================================================

required_inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand"
]

missing_inventory_columns = [
    col
    for col in required_inventory_columns
    if col not in inventory.columns
]

if missing_inventory_columns:

    raise ValueError(
        "Missing required inventory columns:\n"
        f"{missing_inventory_columns}"
    )


# ============================================================
# STANDARDIZE INVENTORY
# ============================================================

inventory["store_id"] = pd.to_numeric(
    inventory["store_id"],
    errors="coerce"
)

inventory["sku_id"] = pd.to_numeric(
    inventory["sku_id"],
    errors="coerce"
)

inventory["stock_on_hand"] = pd.to_numeric(
    inventory["stock_on_hand"],
    errors="coerce"
).fillna(0)


# ============================================================
# CHECK INVENTORY DUPLICATES
# ============================================================

print()
print("=" * 70)
print("CHECKING INVENTORY STORE-SKU KEYS")
print("=" * 70)

duplicate_inventory = inventory.duplicated(
    subset=["store_id", "sku_id"]
).sum()

print(
    "Duplicate Store-SKU inventory rows:",
    duplicate_inventory
)

if duplicate_inventory > 0:

    print()
    print(
        "WARNING: Duplicate Store-SKU inventory records found."
    )

    print(
        "Keeping first record per Store-SKU."
    )

    inventory = (
        inventory
        .drop_duplicates(
            subset=["store_id", "sku_id"],
            keep="first"
        )
    )

print(
    "Final inventory rows:",
    len(inventory)
)


# ============================================================
# LOAD FORECAST FUNCTION
# ============================================================

def load_calibrated_forecast(
    path,
    horizon
):

    print()
    print("-" * 70)
    print(
        f"LOADING CALIBRATED {horizon}-DAY FORECAST"
    )
    print("-" * 70)

    df = pd.read_csv(path)

    print(
        "Rows:",
        len(df)
    )

    required_columns = [
        "store_id",
        "sku_id",
        "date",
        "calibrated_forecast_units"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in {horizon}-day forecast: "
            f"{missing}"
        )

    df["store_id"] = pd.to_numeric(
        df["store_id"],
        errors="coerce"
    )

    df["sku_id"] = pd.to_numeric(
        df["sku_id"],
        errors="coerce"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["calibrated_forecast_units"] = pd.to_numeric(
        df["calibrated_forecast_units"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Forecast validation
    # --------------------------------------------------------

    print(
        "Date range:",
        df["date"].min(),
        "to",
        df["date"].max()
    )

    print(
        "Store-SKU combinations:",
        df.groupby(
            ["store_id", "sku_id"]
        ).ngroups
    )

    print(
        "Negative forecasts:",
        (
            df["calibrated_forecast_units"] < 0
        ).sum()
    )

    # --------------------------------------------------------
    # Aggregate Store-SKU
    # --------------------------------------------------------

    agg = (
        df
        .groupby(
            ["store_id", "sku_id"],
            as_index=False
        )["calibrated_forecast_units"]
        .sum()
    )

    agg = agg.rename(
        columns={
            "calibrated_forecast_units":
                f"calibrated_forecast_{horizon}d"
        }
    )

    print(
        f"Total calibrated {horizon}-day forecast:",
        f"{agg[f'calibrated_forecast_{horizon}d'].sum():,.2f}"
    )

    return agg


# ============================================================
# LOAD ALL FORECASTS
# ============================================================

forecast_30 = load_calibrated_forecast(
    FORECAST_FILES[30],
    30
)

forecast_60 = load_calibrated_forecast(
    FORECAST_FILES[60],
    60
)

forecast_90 = load_calibrated_forecast(
    FORECAST_FILES[90],
    90
)


# ============================================================
# MERGE FORECASTS
# ============================================================

print()
print("=" * 70)
print("MERGING 30 / 60 / 90 DAY FORECASTS")
print("=" * 70)

forecast = forecast_30.merge(
    forecast_60,
    on=["store_id", "sku_id"],
    how="outer"
)

forecast = forecast.merge(
    forecast_90,
    on=["store_id", "sku_id"],
    how="outer"
)

forecast = forecast.fillna(0)

print()
print(
    "Combined forecast Store-SKU rows:",
    len(forecast)
)


# ============================================================
# MERGE INVENTORY + FORECAST
# ============================================================

print()
print("=" * 70)
print("MERGING INVENTORY WITH CALIBRATED FORECAST")
print("=" * 70)

integrated = inventory.merge(
    forecast,
    on=["store_id", "sku_id"],
    how="left",
    indicator=True
)


# ============================================================
# FORECAST MATCH
# ============================================================

integrated["forecast_match"] = np.where(
    integrated["_merge"] == "both",
    "Matched",
    "Inventory Only"
)

integrated.drop(
    columns=["_merge"],
    inplace=True
)


# ============================================================
# FILL FORECAST NULLS
# ============================================================

forecast_columns = [
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d"
]

for col in forecast_columns:

    integrated[col] = pd.to_numeric(
        integrated[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# CREATE FORECAST FEATURES
# ============================================================

print()
print("=" * 70)
print("CREATING FORECAST + INVENTORY FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# Average daily forecast
# ------------------------------------------------------------

integrated["calibrated_avg_daily_forecast_30d"] = (
    integrated["calibrated_forecast_30d"] / 30
)

integrated["calibrated_avg_daily_forecast_60d"] = (
    integrated["calibrated_forecast_60d"] / 60
)

integrated["calibrated_avg_daily_forecast_90d"] = (
    integrated["calibrated_forecast_90d"] / 90
)


# ------------------------------------------------------------
# Inventory coverage based on calibrated forecast
# ------------------------------------------------------------

integrated["calibrated_inventory_coverage_days"] = np.where(

    integrated["calibrated_avg_daily_forecast_30d"] > 0,

    integrated["stock_on_hand"]
    / integrated["calibrated_avg_daily_forecast_30d"],

    np.inf
)


# ------------------------------------------------------------
# Inventory after forecast
# ------------------------------------------------------------

integrated["stock_after_calibrated_30d"] = (
    integrated["stock_on_hand"]
    - integrated["calibrated_forecast_30d"]
)

integrated["stock_after_calibrated_60d"] = (
    integrated["stock_on_hand"]
    - integrated["calibrated_forecast_60d"]
)

integrated["stock_after_calibrated_90d"] = (
    integrated["stock_on_hand"]
    - integrated["calibrated_forecast_90d"]
)


# ------------------------------------------------------------
# Stock-to-forecast ratio
# ------------------------------------------------------------

integrated["stock_to_calibrated_30d_ratio"] = np.where(

    integrated["calibrated_forecast_30d"] > 0,

    integrated["stock_on_hand"]
    / integrated["calibrated_forecast_30d"],

    np.inf
)


# ------------------------------------------------------------
# Replenishment gap
# ------------------------------------------------------------

integrated["calibrated_replenishment_gap_30d"] = (

    integrated["calibrated_forecast_30d"]
    - integrated["stock_on_hand"]

).clip(lower=0)


integrated["calibrated_replenishment_gap_60d"] = (

    integrated["calibrated_forecast_60d"]
    - integrated["stock_on_hand"]

).clip(lower=0)


integrated["calibrated_replenishment_gap_90d"] = (

    integrated["calibrated_forecast_90d"]
    - integrated["stock_on_hand"]

).clip(lower=0)


# ============================================================
# CALIBRATED STOCKOUT RISK
# ============================================================

def calculate_stockout_risk(row):

    stock = row["stock_on_hand"]

    forecast = row[
        "calibrated_forecast_30d"
    ]

    if forecast <= 0:

        return "NO_FORECAST_DEMAND"

    remaining = (
        stock - forecast
    )

    if stock <= 0:

        return "HIGH"

    if remaining < 0:

        return "HIGH"

    if remaining <= (
        0.25 * forecast
    ):

        return "MEDIUM"

    return "LOW"


integrated[
    "calibrated_stockout_risk_30d"
] = integrated.apply(
    calculate_stockout_risk,
    axis=1
)


# ============================================================
# CALIBRATED INVENTORY STATUS
# ============================================================

def calculate_inventory_status(row):

    stock = row["stock_on_hand"]

    daily_forecast = row[
        "calibrated_avg_daily_forecast_30d"
    ]

    if daily_forecast <= 0:

        if stock > 0:
            return "NO_FORECAST_DEMAND"

        return "NO_DEMAND_NO_STOCK"

    coverage = (
        stock / daily_forecast
    )

    if stock <= 0:

        return "OUT_OF_STOCK"

    elif coverage < 7:

        return "CRITICAL"

    elif coverage < 15:

        return "LOW_STOCK"

    elif coverage < 30:

        return "HEALTHY"

    else:

        return "OVERSTOCK"


integrated[
    "calibrated_inventory_status"
] = integrated.apply(
    calculate_inventory_status,
    axis=1
)


# ============================================================
# CALIBRATED FORECAST VS EXISTING FORECAST
# ============================================================

if "forecast_30d_units" in integrated.columns:

    integrated[
        "calibrated_vs_existing_forecast_ratio"
    ] = np.where(

        integrated["forecast_30d_units"] > 0,

        integrated["calibrated_forecast_30d"]
        / integrated["forecast_30d_units"],

        np.nan
    )

    integrated[
        "calibrated_vs_existing_forecast_difference"
    ] = (
        integrated["calibrated_forecast_30d"]
        - integrated["forecast_30d_units"]
    )


# ============================================================
# STORE-SKU VALIDATION
# ============================================================

print()
print("=" * 70)
print("INTEGRATION VALIDATION")
print("=" * 70)

print()
print(
    "Inventory Store-SKU:",
    len(inventory)
)

print(
    "Forecast Store-SKU:",
    len(forecast)
)

print(
    "Integrated Store-SKU:",
    len(integrated)
)

print()
print("Forecast match:")
print(
    integrated[
        "forecast_match"
    ].value_counts()
)


# ============================================================
# DATA QUALITY
# ============================================================

print()
print("=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

print()

for col in forecast_columns:

    print(
        f"Missing {col}:",
        integrated[col].isna().sum()
    )

    print(
        f"Negative {col}:",
        (
            integrated[col] < 0
        ).sum()
    )


print(
    "Negative stock:",
    (
        integrated["stock_on_hand"] < 0
    ).sum()
)

print(
    "Duplicate Store-SKU:",
    integrated.duplicated(
        subset=[
            "store_id",
            "sku_id"
        ]
    ).sum()
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "Inventory Store-SKU combinations",

        "Forecast Store-SKU combinations",

        "Integrated Store-SKU combinations",

        "Total stock on hand",

        "Calibrated 30-day forecast",

        "Calibrated 60-day forecast",

        "Calibrated 90-day forecast",

        "Matched Store-SKU",

        "Inventory-only Store-SKU",

        "Out of stock",

        "Critical",

        "Low stock",

        "Healthy",

        "Overstock",

        "High stockout risk",

        "Medium stockout risk",

        "Low stockout risk"
    ],

    "value": [

        len(inventory),

        len(forecast),

        len(integrated),

        integrated[
            "stock_on_hand"
        ].sum(),

        integrated[
            "calibrated_forecast_30d"
        ].sum(),

        integrated[
            "calibrated_forecast_60d"
        ].sum(),

        integrated[
            "calibrated_forecast_90d"
        ].sum(),

        (
            integrated[
                "forecast_match"
            ] == "Matched"
        ).sum(),

        (
            integrated[
                "forecast_match"
            ] == "Inventory Only"
        ).sum(),

        (
            integrated[
                "calibrated_inventory_status"
            ] == "OUT_OF_STOCK"
        ).sum(),

        (
            integrated[
                "calibrated_inventory_status"
            ] == "CRITICAL"
        ).sum(),

        (
            integrated[
                "calibrated_inventory_status"
            ] == "LOW_STOCK"
        ).sum(),

        (
            integrated[
                "calibrated_inventory_status"
            ] == "HEALTHY"
        ).sum(),

        (
            integrated[
                "calibrated_inventory_status"
            ] == "OVERSTOCK"
        ).sum(),

        (
            integrated[
                "calibrated_stockout_risk_30d"
            ] == "HIGH"
        ).sum(),

        (
            integrated[
                "calibrated_stockout_risk_30d"
            ] == "MEDIUM"
        ).sum(),

        (
            integrated[
                "calibrated_stockout_risk_30d"
            ] == "LOW"
        ).sum()
    ]
})


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 70)
print("SAVING INTEGRATED DATASET")
print("=" * 70)

integrated.to_csv(
    OUTPUT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print()
print("Integrated dataset:")
print(OUTPUT_FILE)

print()
print("Integration summary:")
print(SUMMARY_FILE)


# ============================================================
# PREVIEW
# ============================================================

print()
print("=" * 70)
print("INTEGRATED DATASET PREVIEW")
print("=" * 70)

preview_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "calibrated_avg_daily_forecast_30d",

    "calibrated_inventory_coverage_days",

    "stock_after_calibrated_30d",

    "calibrated_replenishment_gap_30d",

    "calibrated_inventory_status",

    "calibrated_stockout_risk_30d"
]

preview_columns = [
    c
    for c in preview_columns
    if c in integrated.columns
]

print(
    integrated[
        preview_columns
    ].head(10)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PHASE 7.1 FINAL SUMMARY")
print("=" * 70)

print()

print(
    "Integrated rows:",
    len(integrated)
)

print(
    "Integrated columns:",
    len(integrated.columns)
)

print()

print(
    "Total stock on hand:",
    f"{integrated['stock_on_hand'].sum():,.2f}"
)

print(
    "Calibrated 30-day forecast:",
    f"{integrated['calibrated_forecast_30d'].sum():,.2f}"
)

print(
    "Calibrated 60-day forecast:",
    f"{integrated['calibrated_forecast_60d'].sum():,.2f}"
)

print(
    "Calibrated 90-day forecast:",
    f"{integrated['calibrated_forecast_90d'].sum():,.2f}"
)

print()
print("Calibrated inventory status:")
print(
    integrated[
        "calibrated_inventory_status"
    ].value_counts()
)

print()
print("Calibrated 30-day stockout risk:")
print(
    integrated[
        "calibrated_stockout_risk_30d"
    ].value_counts()
)

print()
print("=" * 70)
print("PHASE 7.1 COMPLETED")
print("=" * 70)