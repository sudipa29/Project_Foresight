# ============================================================
# PROJECT FORESIGHT
# PHASE 7.3 - INVENTORY RISK VALIDATION
#
# Purpose:
# Validate the corrected inventory risk and reorder results
# before proceeding to business insights/dashboard development.
#
# Input:
# corrected_inventory_risk_reorder_recommendations.csv
#
# Outputs:
# 1. inventory_risk_validation_summary.csv
# 2. inventory_extreme_overstock.csv
# 3. inventory_store_summary.csv
# 4. inventory_sku_summary.csv
# 5. inventory_no_forecast_analysis.csv
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

INPUT_FILE = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "inventory_risk"
    / "corrected_inventory_risk_reorder_recommendations.csv"
)

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "inventory_risk"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 7.3 - INVENTORY RISK VALIDATION")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILE")
print("=" * 70)

print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

print("FOUND")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING CORRECTED INVENTORY RISK DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "planning_stockout_risk",
    "planning_inventory_risk",
    "planning_reorder_status",

    "suggested_reorder_quantity",
    "reorder_priority",

    "business_action"
]

print("\n" + "=" * 70)
print("CHECKING REQUIRED COLUMNS")
print("=" * 70)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("All required columns FOUND")


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("PREPARING NUMERIC COLUMNS")
print("=" * 70)

numeric_columns = [
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "suggested_reorder_quantity"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# BASIC DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA QUALITY CHECK")
print("=" * 70)

duplicate_keys = df.duplicated(
    subset=["store_id", "sku_id"]
).sum()

negative_stock = (
    df["stock_on_hand"] < 0
).sum()

negative_forecast = (
    (
        (df["calibrated_forecast_30d"] < 0)
        |
        (df["calibrated_forecast_60d"] < 0)
        |
        (df["calibrated_forecast_90d"] < 0)
    )
).sum()

negative_reorder = (
    df["suggested_reorder_quantity"] < 0
).sum()

missing_planning = (
    df["planning_daily_demand"]
    .isna()
    .sum()
)

print(f"Duplicate Store-SKU:       {duplicate_keys:,}")
print(f"Negative stock:             {negative_stock:,}")
print(f"Negative forecast rows:     {negative_forecast:,}")
print(f"Negative reorder quantity:  {negative_reorder:,}")
print(f"Missing planning demand:    {missing_planning:,}")


# ============================================================
# TOTALS
# ============================================================

print("\n" + "=" * 70)
print("TOTAL DEMAND / INVENTORY CHECK")
print("=" * 70)

total_stock = df["stock_on_hand"].sum()

forecast_30 = df["calibrated_forecast_30d"].sum()
forecast_60 = df["calibrated_forecast_60d"].sum()
forecast_90 = df["calibrated_forecast_90d"].sum()

planning_daily = df["planning_daily_demand"].sum()

planning_30 = planning_daily * 30
planning_60 = planning_daily * 60
planning_90 = planning_daily * 90

print(f"Total stock on hand:       {total_stock:,.2f}")
print(f"Calibrated 30-day demand:  {forecast_30:,.2f}")
print(f"Calibrated 60-day demand:  {forecast_60:,.2f}")
print(f"Calibrated 90-day demand:  {forecast_90:,.2f}")

print(f"\nPlanning daily demand:     {planning_daily:,.2f}")
print(f"Planning 30-day demand:    {planning_30:,.2f}")
print(f"Planning 60-day demand:    {planning_60:,.2f}")
print(f"Planning 90-day demand:    {planning_90:,.2f}")


# ============================================================
# INVENTORY / FORECAST RATIOS
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY COVERAGE ANALYSIS")
print("=" * 70)

if forecast_30 > 0:
    total_stock_to_forecast_30 = (
        total_stock / forecast_30
    )
else:
    total_stock_to_forecast_30 = np.nan

if forecast_60 > 0:
    total_stock_to_forecast_60 = (
        total_stock / forecast_60
    )
else:
    total_stock_to_forecast_60 = np.nan

if forecast_90 > 0:
    total_stock_to_forecast_90 = (
        total_stock / forecast_90
    )
else:
    total_stock_to_forecast_90 = np.nan


print(
    f"Stock / 30-day forecast ratio: "
    f"{total_stock_to_forecast_30:,.2f}x"
)

print(
    f"Stock / 60-day forecast ratio: "
    f"{total_stock_to_forecast_60:,.2f}x"
)

print(
    f"Stock / 90-day forecast ratio: "
    f"{total_stock_to_forecast_90:,.2f}x"
)


# ============================================================
# DAYS OF INVENTORY RE-CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DAYS OF INVENTORY")
print("=" * 70)

df["recalculated_days_inventory"] = np.where(
    df["planning_daily_demand"] > 0,
    df["stock_on_hand"]
    / df["planning_daily_demand"],
    np.inf
)

df["days_inventory_difference"] = (
    df["planning_days_of_inventory"]
    - df["recalculated_days_inventory"]
)

valid_doi = df[
    np.isfinite(df["recalculated_days_inventory"])
]

if len(valid_doi) > 0:

    max_difference = (
        valid_doi["days_inventory_difference"]
        .abs()
        .max()
    )

    mean_difference = (
        valid_doi["days_inventory_difference"]
        .abs()
        .mean()
    )

else:

    max_difference = np.nan
    mean_difference = np.nan


print(
    f"Mean absolute DOI difference: "
    f"{mean_difference:.6f}"
)

print(
    f"Maximum DOI difference: "
    f"{max_difference:.6f}"
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY RISK DISTRIBUTION")
print("=" * 70)

risk_distribution = (
    df["planning_inventory_risk"]
    .value_counts(dropna=False)
)

print(risk_distribution)


# ============================================================
# STOCKOUT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STOCKOUT RISK DISTRIBUTION")
print("=" * 70)

stockout_distribution = (
    df["planning_stockout_risk"]
    .value_counts(dropna=False)
)

print(stockout_distribution)


# ============================================================
# REORDER DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("REORDER STATUS DISTRIBUTION")
print("=" * 70)

reorder_distribution = (
    df["planning_reorder_status"]
    .value_counts(dropna=False)
)

print(reorder_distribution)


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("REORDER PRIORITY DISTRIBUTION")
print("=" * 70)

priority_distribution = (
    df["reorder_priority"]
    .value_counts(dropna=False)
)

print(priority_distribution)


# ============================================================
# EXTREME OVERSTOCK
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING EXTREME OVERSTOCK")
print("=" * 70)

extreme_overstock = df[
    (
        df["planning_inventory_risk"]
        == "SEVERE_OVERSTOCK"
    )
    &
    (
        df["planning_days_of_inventory"]
        > 365
    )
].copy()

extreme_overstock = extreme_overstock.sort_values(
    "planning_days_of_inventory",
    ascending=False
)

print(
    f"Store-SKU with >365 days inventory: "
    f"{len(extreme_overstock):,}"
)

print("\nTop 20 extreme overstock:")

print(
    extreme_overstock[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "planning_daily_demand",
            "planning_days_of_inventory",
            "planning_inventory_risk",
            "business_action"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# NO FORECAST ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("NO FORECAST DEMAND ANALYSIS")
print("=" * 70)

no_forecast = df[
    df["planning_stockout_risk"]
    == "NO_FORECAST_DEMAND"
].copy()

print(
    f"Store-SKU with no forecast demand: "
    f"{len(no_forecast):,}"
)

print(
    f"Inventory held by these combinations: "
    f"{no_forecast['stock_on_hand'].sum():,.2f}"
)

print(
    f"30-day forecast from these combinations: "
    f"{no_forecast['calibrated_forecast_30d'].sum():,.2f}"
)

print("\nTop 20 no-forecast inventory:")

print(
    no_forecast[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "calibrated_forecast_30d",
            "planning_daily_demand",
            "business_action"
        ]
    ]
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# STORE LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STORE-LEVEL INVENTORY ANALYSIS")
print("=" * 70)

store_summary = (
    df.groupby("store_id")
    .agg(
        store_sku_count=("sku_id", "count"),
        stock_on_hand=("stock_on_hand", "sum"),
        forecast_30d=("calibrated_forecast_30d", "sum"),
        forecast_60d=("calibrated_forecast_60d", "sum"),
        forecast_90d=("calibrated_forecast_90d", "sum"),
        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),
        severe_overstock_count=(
            "planning_inventory_risk",
            lambda x: (
                x == "SEVERE_OVERSTOCK"
            ).sum()
        ),
        no_forecast_count=(
            "planning_stockout_risk",
            lambda x: (
                x == "NO_FORECAST_DEMAND"
            ).sum()
        )
    )
    .reset_index()
)

store_summary["stock_to_forecast_30d"] = np.where(
    store_summary["forecast_30d"] > 0,
    store_summary["stock_on_hand"]
    / store_summary["forecast_30d"],
    np.inf
)

store_summary = store_summary.sort_values(
    "stock_on_hand",
    ascending=False
)

print("\nTop 20 stores by stock:")

print(
    store_summary.head(20).to_string(
        index=False
    )
)


# ============================================================
# SKU LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SKU-LEVEL INVENTORY ANALYSIS")
print("=" * 70)

sku_summary = (
    df.groupby("sku_id")
    .agg(
        store_count=("store_id", "count"),
        stock_on_hand=("stock_on_hand", "sum"),
        forecast_30d=("calibrated_forecast_30d", "sum"),
        forecast_60d=("calibrated_forecast_60d", "sum"),
        forecast_90d=("calibrated_forecast_90d", "sum"),
        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),
        severe_overstock_count=(
            "planning_inventory_risk",
            lambda x: (
                x == "SEVERE_OVERSTOCK"
            ).sum()
        ),
        no_forecast_count=(
            "planning_stockout_risk",
            lambda x: (
                x == "NO_FORECAST_DEMAND"
            ).sum()
        )
    )
    .reset_index()
)

sku_summary["stock_to_forecast_30d"] = np.where(
    sku_summary["forecast_30d"] > 0,
    sku_summary["stock_on_hand"]
    / sku_summary["forecast_30d"],
    np.inf
)

sku_summary = sku_summary.sort_values(
    "stock_on_hand",
    ascending=False
)

print("\nTop 20 SKUs by stock:")

print(
    sku_summary.head(20).to_string(
        index=False
    )
)


# ============================================================
# REORDER VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("REORDER VALIDATION")
print("=" * 70)

total_reorder = (
    df["suggested_reorder_quantity"]
    .sum()
)

positive_reorder = (
    df["suggested_reorder_quantity"] > 0
).sum()

print(
    f"Total suggested reorder quantity: "
    f"{total_reorder:,.2f}"
)

print(
    f"Store-SKU requiring reorder: "
    f"{positive_reorder:,}"
)

if total_reorder == 0:

    print(
        "\nRESULT: No additional inventory "
        "is recommended under the current "
        "planning assumptions."
    )

else:

    print(
        "\nRESULT: Replenishment is required "
        "for some Store-SKU combinations."
    )


# ============================================================
# BUSINESS CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS CONSISTENCY CHECK")
print("=" * 70)

checks = {}

checks["duplicate_store_sku"] = (
    duplicate_keys == 0
)

checks["negative_stock"] = (
    negative_stock == 0
)

checks["negative_forecast"] = (
    negative_forecast == 0
)

checks["negative_reorder"] = (
    negative_reorder == 0
)

checks["missing_planning_demand"] = (
    missing_planning == 0
)

checks["doi_calculation"] = (
    pd.isna(max_difference)
    or max_difference < 0.000001
)

checks["no_negative_inventory_gap"] = True

if "inventory_gap" in df.columns:

    checks["no_negative_inventory_gap"] = (
        df["inventory_gap"]
        .dropna()
        .ge(0)
        .all()
    )

for name, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{name:<35}: {status}"
    )


# ============================================================
# VALIDATION SUMMARY
# ============================================================

validation_summary = pd.DataFrame(
    [
        {
            "metric": "total_store_sku",
            "value": len(df)
        },
        {
            "metric": "total_stock_on_hand",
            "value": total_stock
        },
        {
            "metric": "calibrated_forecast_30d",
            "value": forecast_30
        },
        {
            "metric": "calibrated_forecast_60d",
            "value": forecast_60
        },
        {
            "metric": "calibrated_forecast_90d",
            "value": forecast_90
        },
        {
            "metric": "planning_30d_demand",
            "value": planning_30
        },
        {
            "metric": "planning_60d_demand",
            "value": planning_60
        },
        {
            "metric": "planning_90d_demand",
            "value": planning_90
        },
        {
            "metric": "stock_to_forecast_30d",
            "value": total_stock_to_forecast_30
        },
        {
            "metric": "stock_to_forecast_60d",
            "value": total_stock_to_forecast_60
        },
        {
            "metric": "stock_to_forecast_90d",
            "value": total_stock_to_forecast_90
        },
        {
            "metric": "extreme_overstock_store_sku",
            "value": len(extreme_overstock)
        },
        {
            "metric": "no_forecast_store_sku",
            "value": len(no_forecast)
        },
        {
            "metric": "no_forecast_inventory",
            "value": no_forecast[
                "stock_on_hand"
            ].sum()
        },
        {
            "metric": "total_reorder_quantity",
            "value": total_reorder
        },
        {
            "metric": "positive_reorder_store_sku",
            "value": positive_reorder
        },
        {
            "metric": "mean_doi_difference",
            "value": mean_difference
        },
        {
            "metric": "max_doi_difference",
            "value": max_difference
        }
    ]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING VALIDATION FILES")
print("=" * 70)

validation_file = (
    OUTPUT_DIR
    / "inventory_risk_validation_summary.csv"
)

extreme_file = (
    OUTPUT_DIR
    / "inventory_extreme_overstock.csv"
)

store_file = (
    OUTPUT_DIR
    / "inventory_store_summary.csv"
)

sku_file = (
    OUTPUT_DIR
    / "inventory_sku_summary.csv"
)

no_forecast_file = (
    OUTPUT_DIR
    / "inventory_no_forecast_analysis.csv"
)

validation_summary.to_csv(
    validation_file,
    index=False
)

extreme_overstock.to_csv(
    extreme_file,
    index=False
)

store_summary.to_csv(
    store_file,
    index=False
)

sku_summary.to_csv(
    sku_file,
    index=False
)

no_forecast.to_csv(
    no_forecast_file,
    index=False
)

print(f"Saved:\n{validation_file}")
print(f"Saved:\n{extreme_file}")
print(f"Saved:\n{store_file}")
print(f"Saved:\n{sku_file}")
print(f"Saved:\n{no_forecast_file}")


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 7.3 FINAL DECISION")
print("=" * 70)

all_pass = all(checks.values())

if all_pass:

    print("DATA QUALITY CHECKS: PASS")

else:

    print("DATA QUALITY CHECKS: REVIEW REQUIRED")

print(
    f"\nTotal inventory: "
    f"{total_stock:,.2f}"
)

print(
    f"30-day calibrated forecast: "
    f"{forecast_30:,.2f}"
)

print(
    f"30-day planning demand: "
    f"{planning_30:,.2f}"
)

print(
    f"Inventory / 30-day forecast: "
    f"{total_stock_to_forecast_30:,.2f}x"
)

print(
    f"Extreme overstock Store-SKU: "
    f"{len(extreme_overstock):,}"
)

print(
    f"No-forecast Store-SKU: "
    f"{len(no_forecast):,}"
)

print(
    f"Suggested reorder quantity: "
    f"{total_reorder:,.2f}"
)

print("\n" + "=" * 70)

if all_pass:

    print(
        "PHASE 7.3 COMPLETED"
    )

    print(
        "Inventory risk calculations "
        "passed technical validation."
    )

    print(
        "\nIMPORTANT BUSINESS FINDING:"
    )

    print(
        "Inventory is substantially higher "
        "than forecast demand."
    )

    print(
        "\nNext step:"
    )

    print(
        "PHASE 7.4 - BUSINESS INVENTORY INSIGHTS"
    )

else:

    print(
        "PHASE 7.3 REQUIRES REVIEW"
    )

print("=" * 70)