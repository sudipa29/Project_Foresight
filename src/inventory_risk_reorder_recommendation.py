# ============================================================
# PROJECT FORESIGHT
# PHASE 7.2 - CORRECTED INVENTORY RISK & REORDER RECOMMENDATION
#
# Purpose:
#   Convert forecast + historical demand into practical
#   inventory planning recommendations.
#
# Production Forecast:
#   Calibrated Intermittent Forecast
#
# Important:
#   Forecast is NOT used blindly for inventory planning.
#   Planning demand combines:
#
#       40% Recent 30-Day Historical Demand
#       30% 90-Day Historical Demand
#       30% Calibrated 30-Day Forecast
#
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

INPUT_FILE = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "integration"
    / "calibrated_forecast_inventory_integrated.csv"
)

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "inventory_risk"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "corrected_inventory_risk_reorder_recommendations.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "corrected_inventory_risk_summary.csv"
)

REORDER_SUMMARY_FILE = (
    OUTPUT_DIR
    / "corrected_reorder_recommendation_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Planning demand weights
WEIGHT_RECENT = 0.40
WEIGHT_LONG_TERM = 0.30
WEIGHT_FORECAST = 0.30

# Inventory planning periods
SAFETY_STOCK_DAYS = 7
LEAD_TIME_DAYS = 7
TARGET_COVERAGE_DAYS = 30

# Risk thresholds based on inventory coverage
STOCKOUT_DAYS = 7
UNDERSTOCK_DAYS = 14
HEALTHY_MAX_DAYS = 45
OVERSTOCK_MAX_DAYS = 90


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 7.2 - CORRECTED INVENTORY RISK & REORDER RECOMMENDATION")
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
        f"\nInput file not found:\n{INPUT_FILE}"
    )

print("FOUND")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING INTEGRATED FORECAST + INVENTORY DATA")
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
    "units_30d",
    "units_90d",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
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
        "\nMissing required columns:\n"
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
    "units_30d",
    "units_90d",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

    df[col] = df[col].clip(lower=0)


# ============================================================
# KEY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING STORE-SKU DATA")
print("=" * 70)

duplicate_keys = df.duplicated(
    subset=["store_id", "sku_id"]
).sum()

negative_stock = (
    df["stock_on_hand"] < 0
).sum()

print(f"Duplicate Store-SKU: {duplicate_keys}")
print(f"Negative stock: {negative_stock}")

if duplicate_keys > 0:
    raise ValueError(
        "Duplicate Store-SKU rows detected."
    )

if negative_stock > 0:
    raise ValueError(
        "Negative stock detected."
    )


# ============================================================
# HISTORICAL DAILY DEMAND
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING HISTORICAL DAILY DEMAND")
print("=" * 70)

df["historical_daily_demand_30d"] = (
    df["units_30d"] / 30.0
)

df["historical_daily_demand_90d"] = (
    df["units_90d"] / 90.0
)


# ============================================================
# CALIBRATED FORECAST DAILY DEMAND
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CALIBRATED FORECAST DEMAND")
print("=" * 70)

df["calibrated_daily_forecast_30d"] = (
    df["calibrated_forecast_30d"] / 30.0
)

df["calibrated_daily_forecast_60d"] = (
    df["calibrated_forecast_60d"] / 60.0
)

df["calibrated_daily_forecast_90d"] = (
    df["calibrated_forecast_90d"] / 90.0
)


# ============================================================
# PLANNING DEMAND
# ============================================================

print("\n" + "=" * 70)
print("CREATING INVENTORY PLANNING DEMAND")
print("=" * 70)

print(
    "\nPlanning demand weights:"
)

print(
    f"Recent 30-day demand : {WEIGHT_RECENT:.0%}"
)

print(
    f"90-day historical    : {WEIGHT_LONG_TERM:.0%}"
)

print(
    f"Calibrated forecast  : {WEIGHT_FORECAST:.0%}"
)


df["planning_daily_demand"] = (
    WEIGHT_RECENT
    * df["historical_daily_demand_30d"]
    +
    WEIGHT_LONG_TERM
    * df["historical_daily_demand_90d"]
    +
    WEIGHT_FORECAST
    * df["calibrated_daily_forecast_30d"]
)


# ============================================================
# DEMAND STATUS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFYING DEMAND AVAILABILITY")
print("=" * 70)

df["demand_available"] = np.where(
    df["planning_daily_demand"] > 0,
    "YES",
    "NO"
)


# ============================================================
# SAFETY STOCK
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SAFETY STOCK")
print("=" * 70)

df["planning_safety_stock"] = (
    df["planning_daily_demand"]
    * SAFETY_STOCK_DAYS
)


# ============================================================
# REORDER POINT
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING REORDER POINT")
print("=" * 70)

df["planning_reorder_point"] = (
    df["planning_daily_demand"]
    * LEAD_TIME_DAYS
    +
    df["planning_safety_stock"]
)


# ============================================================
# TARGET STOCK
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING TARGET STOCK")
print("=" * 70)

df["planning_target_stock"] = (
    df["planning_daily_demand"]
    * TARGET_COVERAGE_DAYS
    +
    df["planning_safety_stock"]
)


# ============================================================
# DAYS OF INVENTORY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING DAYS OF INVENTORY")
print("=" * 70)

df["planning_days_of_inventory"] = np.where(
    df["planning_daily_demand"] > 0,
    df["stock_on_hand"]
    / df["planning_daily_demand"],
    np.nan
)


# ============================================================
# FORECAST COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING FORECAST COVERAGE")
print("=" * 70)

df["calibrated_forecast_coverage_days"] = np.where(
    df["calibrated_daily_forecast_30d"] > 0,
    df["stock_on_hand"]
    / df["calibrated_daily_forecast_30d"],
    np.nan
)


# ============================================================
# STOCK AFTER FORECAST
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING STOCK POSITION")
print("=" * 70)

df["stock_after_30d_planning_demand"] = (
    df["stock_on_hand"]
    - (
        df["planning_daily_demand"]
        * 30
    )
)

df["stock_after_60d_planning_demand"] = (
    df["stock_on_hand"]
    - (
        df["planning_daily_demand"]
        * 60
    )
)

df["stock_after_90d_planning_demand"] = (
    df["stock_on_hand"]
    - (
        df["planning_daily_demand"]
        * 90
    )
)


# ============================================================
# INVENTORY GAP
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING INVENTORY GAP")
print("=" * 70)

df["inventory_gap_to_target"] = (
    df["planning_target_stock"]
    - df["stock_on_hand"]
)


# ============================================================
# SUGGESTED REORDER QUANTITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SUGGESTED REORDER QUANTITY")
print("=" * 70)

df["suggested_reorder_quantity"] = np.where(
    df["planning_daily_demand"] > 0,
    np.maximum(
        df["planning_target_stock"]
        - df["stock_on_hand"],
        0
    ),
    0
)

df["suggested_reorder_quantity"] = (
    df["suggested_reorder_quantity"]
    .round(0)
    .astype(int)
)


# ============================================================
# STOCKOUT RISK
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFYING STOCKOUT RISK")
print("=" * 70)


def classify_stockout_risk(row):

    demand = row["planning_daily_demand"]
    stock = row["stock_on_hand"]

    if demand <= 0:
        return "NO_FORECAST_DEMAND"

    days = stock / demand

    if days <= STOCKOUT_DAYS:
        return "CRITICAL"

    elif days <= UNDERSTOCK_DAYS:
        return "HIGH"

    elif days <= HEALTHY_MAX_DAYS:
        return "MEDIUM"

    else:
        return "LOW"


df["planning_stockout_risk"] = df.apply(
    classify_stockout_risk,
    axis=1
)


# ============================================================
# INVENTORY RISK
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFYING INVENTORY RISK")
print("=" * 70)


def classify_inventory_risk(row):

    demand = row["planning_daily_demand"]
    stock = row["stock_on_hand"]

    if demand <= 0:
        return "NO_FORECAST_DEMAND"

    days = stock / demand

    if days <= STOCKOUT_DAYS:
        return "STOCKOUT_RISK"

    elif days <= UNDERSTOCK_DAYS:
        return "UNDERSTOCK"

    elif days <= HEALTHY_MAX_DAYS:
        return "HEALTHY"

    elif days <= OVERSTOCK_MAX_DAYS:
        return "OVERSTOCK"

    else:
        return "SEVERE_OVERSTOCK"


df["planning_inventory_risk"] = df.apply(
    classify_inventory_risk,
    axis=1
)


# ============================================================
# REORDER STATUS
# ============================================================

print("\n" + "=" * 70)
print("CREATING REORDER STATUS")
print("=" * 70)


def classify_reorder_status(row):

    demand = row["planning_daily_demand"]
    reorder_qty = row["suggested_reorder_quantity"]

    if demand <= 0:
        return "NO_FORECAST_DEMAND"

    if reorder_qty > 0:
        return "REORDER"

    return "NO_REORDER"


df["planning_reorder_status"] = df.apply(
    classify_reorder_status,
    axis=1
)


# ============================================================
# REORDER PRIORITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING REORDER PRIORITY")
print("=" * 70)


def calculate_priority(row):

    risk = row["planning_stockout_risk"]
    reorder_qty = row["suggested_reorder_quantity"]

    if risk == "CRITICAL" and reorder_qty > 0:
        return "P1 - CRITICAL"

    elif risk == "HIGH" and reorder_qty > 0:
        return "P2 - HIGH"

    elif risk == "MEDIUM" and reorder_qty > 0:
        return "P3 - MEDIUM"

    elif reorder_qty > 0:
        return "P4 - LOW"

    return "P5 - NO ACTION"


df["reorder_priority"] = df.apply(
    calculate_priority,
    axis=1
)


# ============================================================
# BUSINESS ACTION
# ============================================================

print("\n" + "=" * 70)
print("CREATING BUSINESS ACTION")
print("=" * 70)


def business_action(row):

    risk = row["planning_inventory_risk"]
    reorder_qty = row["suggested_reorder_quantity"]

    if risk == "NO_FORECAST_DEMAND":
        return "NO_ACTION_REVIEW_DORMANT"

    elif risk == "STOCKOUT_RISK":
        return "URGENT_REORDER"

    elif risk == "UNDERSTOCK":
        return "REORDER_SOON"

    elif risk == "HEALTHY":
        return "MAINTAIN_STOCK"

    elif risk == "OVERSTOCK":
        return "CONTROL_REPLENISHMENT"

    elif risk == "SEVERE_OVERSTOCK":
        return "STOP_REPLENISHMENT_REVIEW_STOCK"

    return "REVIEW"


df["business_action"] = df.apply(
    business_action,
    axis=1
)


# ============================================================
# FORECAST VS PLANNING DEMAND
# ============================================================

print("\n" + "=" * 70)
print("COMPARING FORECAST WITH PLANNING DEMAND")
print("=" * 70)

df["forecast_vs_planning_ratio"] = np.where(
    df["planning_daily_demand"] > 0,
    df["calibrated_daily_forecast_30d"]
    / df["planning_daily_demand"],
    np.nan
)


# ============================================================
# STOCK TO PLANNING DEMAND RATIO
# ============================================================

df["stock_to_planning_30d_ratio"] = np.where(
    df["planning_daily_demand"] > 0,
    df["stock_on_hand"]
    / (
        df["planning_daily_demand"]
        * 30
    ),
    np.nan
)


# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

negative_planning_demand = (
    df["planning_daily_demand"] < 0
).sum()

negative_reorder = (
    df["suggested_reorder_quantity"] < 0
).sum()

missing_planning_demand = (
    df["planning_daily_demand"].isna()
).sum()

print(f"Rows: {len(df):,}")
print(f"Duplicate Store-SKU: {duplicate_keys}")
print(f"Negative stock: {negative_stock}")
print(f"Negative planning demand: {negative_planning_demand}")
print(f"Negative reorder quantity: {negative_reorder}")
print(f"Missing planning demand: {missing_planning_demand}")


if negative_planning_demand > 0:
    raise ValueError(
        "Negative planning demand detected."
    )

if negative_reorder > 0:
    raise ValueError(
        "Negative reorder quantity detected."
    )

if missing_planning_demand > 0:
    raise ValueError(
        "Missing planning demand detected."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY RISK SUMMARY")
print("=" * 70)

inventory_risk_summary = (
    df["planning_inventory_risk"]
    .value_counts()
)

print(inventory_risk_summary)


print("\n" + "=" * 70)
print("STOCKOUT RISK SUMMARY")
print("=" * 70)

stockout_summary = (
    df["planning_stockout_risk"]
    .value_counts()
)

print(stockout_summary)


print("\n" + "=" * 70)
print("REORDER STATUS SUMMARY")
print("=" * 70)

reorder_status_summary = (
    df["planning_reorder_status"]
    .value_counts()
)

print(reorder_status_summary)


print("\n" + "=" * 70)
print("REORDER PRIORITY SUMMARY")
print("=" * 70)

priority_summary = (
    df["reorder_priority"]
    .value_counts()
)

print(priority_summary)


# ============================================================
# TOP REORDER RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 REORDER RECOMMENDATIONS")
print("=" * 70)

reorder_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "planning_stockout_risk",
    "planning_inventory_risk",
    "suggested_reorder_quantity",
    "planning_reorder_status",
    "reorder_priority",
    "business_action",
]

top_reorders = (
    df[
        df["suggested_reorder_quantity"] > 0
    ]
    .sort_values(
        by=[
            "reorder_priority",
            "suggested_reorder_quantity"
        ],
        ascending=[True, False]
    )
    [reorder_columns]
    .head(20)
)

print(top_reorders.to_string(index=False))


# ============================================================
# TOP STOCKOUT RISK
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 STOCKOUT-RISK STORE-SKU")
print("=" * 70)

stockout_priority = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
    "NO_FORECAST_DEMAND": 5,
}

df["_risk_order"] = (
    df["planning_stockout_risk"]
    .map(stockout_priority)
)

top_stockout = (
    df[
        df["planning_stockout_risk"].isin(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM"
            ]
        )
    ]
    .sort_values(
        by=[
            "_risk_order",
            "planning_days_of_inventory"
        ],
        ascending=[True, True]
    )
    [
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "planning_daily_demand",
            "planning_days_of_inventory",
            "planning_stockout_risk",
            "suggested_reorder_quantity",
            "reorder_priority",
            "business_action",
        ]
    ]
    .head(20)
)

print(top_stockout.to_string(index=False))


# ============================================================
# OVERSTOCK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 OVERSTOCK STORE-SKU")
print("=" * 70)

overstock = (
    df[
        df["planning_inventory_risk"].isin(
            [
                "OVERSTOCK",
                "SEVERE_OVERSTOCK"
            ]
        )
    ]
    .sort_values(
        by="planning_days_of_inventory",
        ascending=False
    )
    [
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "planning_daily_demand",
            "planning_days_of_inventory",
            "planning_inventory_risk",
            "business_action",
        ]
    ]
    .head(20)
)

print(overstock.to_string(index=False))


# ============================================================
# FINAL BUSINESS TOTALS
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS SUMMARY")
print("=" * 70)

total_stock = (
    df["stock_on_hand"].sum()
)

total_forecast_30 = (
    df["calibrated_forecast_30d"].sum()
)

total_forecast_60 = (
    df["calibrated_forecast_60d"].sum()
)

total_forecast_90 = (
    df["calibrated_forecast_90d"].sum()
)

total_planning_30 = (
    df["planning_daily_demand"].sum()
    * 30
)

total_planning_60 = (
    df["planning_daily_demand"].sum()
    * 60
)

total_planning_90 = (
    df["planning_daily_demand"].sum()
    * 90
)

total_reorder = (
    df["suggested_reorder_quantity"].sum()
)

print(
    f"Total Store-SKU: "
    f"{len(df):,}"
)

print(
    f"Total stock on hand: "
    f"{total_stock:,.2f}"
)

print(
    f"Calibrated 30-day forecast: "
    f"{total_forecast_30:,.2f}"
)

print(
    f"Calibrated 60-day forecast: "
    f"{total_forecast_60:,.2f}"
)

print(
    f"Calibrated 90-day forecast: "
    f"{total_forecast_90:,.2f}"
)

print(
    f"Planning 30-day demand: "
    f"{total_planning_30:,.2f}"
)

print(
    f"Planning 60-day demand: "
    f"{total_planning_60:,.2f}"
)

print(
    f"Planning 90-day demand: "
    f"{total_planning_90:,.2f}"
)

print(
    f"Total suggested reorder quantity: "
    f"{total_reorder:,.0f}"
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_rows = []

for risk, count in inventory_risk_summary.items():

    subset = df[
        df["planning_inventory_risk"] == risk
    ]

    summary_rows.append(
        {
            "inventory_risk": risk,
            "store_sku_count": len(subset),
            "total_stock": subset[
                "stock_on_hand"
            ].sum(),
            "total_planning_30d_demand": (
                subset[
                    "planning_daily_demand"
                ].sum() * 30
            ),
            "total_reorder_quantity": subset[
                "suggested_reorder_quantity"
            ].sum(),
            "avg_days_of_inventory": subset[
                "planning_days_of_inventory"
            ].mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by="store_sku_count",
    ascending=False
)


# ============================================================
# REORDER SUMMARY TABLE
# ============================================================

reorder_summary_rows = []

for status, count in reorder_status_summary.items():

    subset = df[
        df["planning_reorder_status"] == status
    ]

    reorder_summary_rows.append(
        {
            "reorder_status": status,
            "store_sku_count": len(subset),
            "total_stock": subset[
                "stock_on_hand"
            ].sum(),
            "total_reorder_quantity": subset[
                "suggested_reorder_quantity"
            ].sum(),
            "avg_planning_days_inventory": subset[
                "planning_days_of_inventory"
            ].mean(),
        }
    )

reorder_summary_df = pd.DataFrame(
    reorder_summary_rows
)


# ============================================================
# REMOVE INTERNAL COLUMN
# ============================================================

df = df.drop(
    columns=["_risk_order"],
    errors="ignore"
)


# ============================================================
# SAVE DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING CORRECTED INVENTORY RISK DATASET")
print("=" * 70)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

reorder_summary_df.to_csv(
    REORDER_SUMMARY_FILE,
    index=False
)

print("\nSaved:")

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    REORDER_SUMMARY_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 7.2 FINAL SUMMARY")
print("=" * 70)

print(
    f"Total Store-SKU: {len(df):,}"
)

print(
    f"Total stock on hand: {total_stock:,.2f}"
)

print(
    f"Calibrated 30-day forecast: "
    f"{total_forecast_30:,.2f}"
)

print(
    f"Planning 30-day demand: "
    f"{total_planning_30:,.2f}"
)

print(
    f"Total suggested reorder quantity: "
    f"{total_reorder:,.0f}"
)

print("\nInventory risk distribution:")
print(inventory_risk_summary)

print("\nStockout risk distribution:")
print(stockout_summary)

print("\nReorder status distribution:")
print(reorder_status_summary)

print("\nPriority distribution:")
print(priority_summary)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 7.2 COMPLETED")
print("=" * 70)

print(
    "Corrected inventory risk and reorder "
    "recommendations successfully generated."
)

print("\nReady for:")
print("PHASE 7.3 - INVENTORY RISK VALIDATION")

print("=" * 70)