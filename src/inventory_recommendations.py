# ============================================================
# PROJECT FORESIGHT
# Phase 6.3 - Inventory Recommendations
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

FORECAST_PATH = (
    PROCESSED_PATH
    / "forecasting"
    / "future"
)

INVENTORY_PATH = PROCESSED_PATH / "inventory_current.csv"

OUTPUT_PATH = (
    FORECAST_PATH
    / "inventory_recommendations"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FILES
# ============================================================

FORECAST_30 = (
    FORECAST_PATH
    / "future_30_day_forecast.csv"
)

FORECAST_60 = (
    FORECAST_PATH
    / "future_60_day_forecast.csv"
)

FORECAST_90 = (
    FORECAST_PATH
    / "future_90_day_forecast.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

section("PROJECT FORESIGHT - PHASE 6.3")
print("INVENTORY RECOMMENDATIONS")

section("CHECKING REQUIRED FILES")

required_files = {
    "30D Forecast": FORECAST_30,
    "60D Forecast": FORECAST_60,
    "90D Forecast": FORECAST_90,
    "Current Inventory": INVENTORY_PATH,
}

for name, path in required_files.items():

    if path.exists():
        print(f"PASS: {name}")
        print(f"      {path}")

    else:
        print(f"FAIL: {name}")
        print(f"      {path}")
        raise FileNotFoundError(path)


# ============================================================
# LOAD INVENTORY
# ============================================================

section("LOADING CURRENT INVENTORY")

inventory = pd.read_csv(
    INVENTORY_PATH
)

print("Inventory rows:", len(inventory))

print("Inventory columns:")
print(inventory.columns.tolist())


# ============================================================
# STANDARDIZE INVENTORY COLUMNS
# ============================================================

required_inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
]

missing_inventory_columns = [
    col
    for col in required_inventory_columns
    if col not in inventory.columns
]

if missing_inventory_columns:

    raise ValueError(
        "Missing inventory columns: "
        + str(missing_inventory_columns)
    )


# ============================================================
# CLEAN INVENTORY
# ============================================================

numeric_inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
]

for col in numeric_inventory_columns:

    inventory[col] = pd.to_numeric(
        inventory[col],
        errors="coerce"
    )


inventory = inventory.dropna(
    subset=[
        "store_id",
        "sku_id"
    ]
)


# ============================================================
# LOAD FORECAST FUNCTION
# ============================================================

def load_forecast(path, horizon):

    section(f"LOADING {horizon}-DAY FORECAST")

    df = pd.read_csv(path)

    print("Rows:", len(df))

    if "forecast_units" not in df.columns:

        raise ValueError(
            f"forecast_units missing from {path}"
        )

    df["forecast_units"] = pd.to_numeric(
        df["forecast_units"],
        errors="coerce"
    ).fillna(0)

    df["store_id"] = pd.to_numeric(
        df["store_id"],
        errors="coerce"
    )

    df["sku_id"] = pd.to_numeric(
        df["sku_id"],
        errors="coerce"
    )

    summary = (
        df
        .groupby(
            ["store_id", "sku_id"],
            as_index=False
        )
        ["forecast_units"]
        .sum()
        .rename(
            columns={
                "forecast_units":
                f"forecast_{horizon}d_units"
            }
        )
    )

    return summary


# ============================================================
# LOAD FORECASTS
# ============================================================

forecast_30 = load_forecast(
    FORECAST_30,
    30
)

forecast_60 = load_forecast(
    FORECAST_60,
    60
)

forecast_90 = load_forecast(
    FORECAST_90,
    90
)


# ============================================================
# MERGE FORECASTS
# ============================================================

section("MERGING FORECASTS WITH INVENTORY")

df = inventory.merge(
    forecast_30,
    on=["store_id", "sku_id"],
    how="left"
)

df = df.merge(
    forecast_60,
    on=["store_id", "sku_id"],
    how="left"
)

df = df.merge(
    forecast_90,
    on=["store_id", "sku_id"],
    how="left"
)


forecast_columns = [
    "forecast_30d_units",
    "forecast_60d_units",
    "forecast_90d_units",
]

for col in forecast_columns:

    df[col] = df[col].fillna(0)


print("Integrated rows:", len(df))


# ============================================================
# DAILY FORECAST
# ============================================================

df["forecast_daily_demand"] = (
    df["forecast_30d_units"] / 30
)


# ============================================================
# DAYS OF INVENTORY
# ============================================================

df["days_of_inventory"] = np.where(
    df["forecast_daily_demand"] > 0,
    df["stock_on_hand"]
    / df["forecast_daily_demand"],
    np.inf
)


# ============================================================
# PROJECTED STOCK AFTER 30 DAYS
# ============================================================

df["projected_stock_after_30d"] = (
    df["stock_on_hand"]
    - df["forecast_30d_units"]
)


# ============================================================
# STOCK GAP
# ============================================================

df["safety_stock_gap"] = (
    df["safety_stock"]
    - df["projected_stock_after_30d"]
)


df["reorder_point_gap"] = (
    df["reorder_point"]
    - df["projected_stock_after_30d"]
)


# ============================================================
# TARGET STOCK
# ============================================================

# Target stock = 30-day forecast + safety stock

df["target_stock"] = (
    df["forecast_30d_units"]
    + df["safety_stock"]
)


# ============================================================
# RECOMMENDED REORDER QUANTITY
# ============================================================

df["suggested_reorder_qty"] = np.maximum(
    df["target_stock"]
    - df["stock_on_hand"],
    0
)


# ============================================================
# INVENTORY RISK
# ============================================================

def determine_inventory_risk(row):

    stock = row["stock_on_hand"]
    reorder = row["reorder_point"]
    safety = row["safety_stock"]
    forecast = row["forecast_30d_units"]
    projected = row["projected_stock_after_30d"]
    days = row["days_of_inventory"]

    # --------------------------------------------------------
    # No demand
    # --------------------------------------------------------

    if forecast <= 0:

        if stock > safety:

            return "NO_DEMAND_OVERSTOCK"

        return "NO_DEMAND"


    # --------------------------------------------------------
    # Immediate stockout
    # --------------------------------------------------------

    if stock <= 0:

        return "STOCKOUT"


    # --------------------------------------------------------
    # Projected stockout
    # --------------------------------------------------------

    if projected <= 0:

        return "HIGH"


    # --------------------------------------------------------
    # Below safety stock
    # --------------------------------------------------------

    if projected < safety:

        return "HIGH"


    # --------------------------------------------------------
    # Below reorder point
    # --------------------------------------------------------

    if projected < reorder:

        return "MEDIUM"


    # --------------------------------------------------------
    # Low days of cover
    # --------------------------------------------------------

    if days < 15:

        return "MEDIUM"


    # --------------------------------------------------------
    # Excess inventory
    # --------------------------------------------------------

    if days > 90:

        return "OVERSTOCK"


    return "LOW"


df["inventory_risk"] = df.apply(
    determine_inventory_risk,
    axis=1
)


# ============================================================
# STOCKOUT RISK
# ============================================================

def stockout_risk(row):

    days = row["days_of_inventory"]

    if row["forecast_30d_units"] <= 0:

        return "NO_DEMAND"

    if row["stock_on_hand"] <= 0:

        return "CRITICAL"

    if days < 7:

        return "CRITICAL"

    if days < 15:

        return "HIGH"

    if days < 30:

        return "MEDIUM"

    return "LOW"


df["stockout_risk"] = df.apply(
    stockout_risk,
    axis=1
)


# ============================================================
# REORDER STATUS
# ============================================================

def reorder_status(row):

    risk = row["inventory_risk"]

    qty = row["suggested_reorder_qty"]

    stock = row["stock_on_hand"]
    reorder = row["reorder_point"]

    if risk == "STOCKOUT":

        return "URGENT_REORDER"

    if risk == "HIGH" and qty > 0:

        return "REORDER_NOW"

    if stock < reorder and qty > 0:

        return "REORDER"

    if risk == "MEDIUM" and qty > 0:

        return "PLAN_REORDER"

    if risk == "OVERSTOCK":

        return "NO_REORDER"

    if risk == "NO_DEMAND_OVERSTOCK":

        return "NO_REORDER"

    return "NO_REORDER"


df["reorder_status"] = df.apply(
    reorder_status,
    axis=1
)


# ============================================================
# BUSINESS PRIORITY
# ============================================================

def business_priority(row):

    risk = row["inventory_risk"]
    stockout = row["stockout_risk"]

    if stockout == "CRITICAL":

        return "CRITICAL"

    if risk == "STOCKOUT":

        return "CRITICAL"

    if risk == "HIGH":

        return "HIGH"

    if stockout == "HIGH":

        return "HIGH"

    if risk == "MEDIUM":

        return "MEDIUM"

    if risk == "OVERSTOCK":

        return "LOW"

    if risk == "NO_DEMAND_OVERSTOCK":

        return "LOW"

    return "NORMAL"


df["business_priority"] = df.apply(
    business_priority,
    axis=1
)


# ============================================================
# BUSINESS ACTION
# ============================================================

def business_action(row):

    priority = row["business_priority"]
    risk = row["inventory_risk"]
    reorder_qty = row["suggested_reorder_qty"]

    if priority == "CRITICAL":

        return (
            "Immediate replenishment required"
        )

    if priority == "HIGH":

        return (
            f"Reorder approximately "
            f"{reorder_qty:.0f} units"
        )

    if risk == "MEDIUM":

        return (
            f"Monitor inventory and plan "
            f"reorder of approximately "
            f"{reorder_qty:.0f} units"
        )

    if risk in [
        "OVERSTOCK",
        "NO_DEMAND_OVERSTOCK"
    ]:

        return (
            "Avoid additional replenishment; "
            "consider inventory reduction"
        )

    if risk == "NO_DEMAND":

        return (
            "Monitor demand before replenishment"
        )

    return "Normal inventory monitoring"


df["business_action"] = df.apply(
    business_action,
    axis=1
)


# ============================================================
# PRIORITY SCORE
# ============================================================

priority_score = {

    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "NORMAL": 1,
    "LOW": 0,
}

df["priority_score"] = (
    df["business_priority"]
    .map(priority_score)
    .fillna(0)
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "priority_score",
        "suggested_reorder_qty",
        "forecast_30d_units"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


# ============================================================
# SAVE MAIN RECOMMENDATIONS
# ============================================================

section("SAVING INVENTORY RECOMMENDATIONS")

main_columns = [
    "store_id",
    "sku_id",

    "stock_on_hand",
    "reorder_point",
    "safety_stock",

    "forecast_30d_units",
    "forecast_60d_units",
    "forecast_90d_units",

    "forecast_daily_demand",

    "days_of_inventory",

    "projected_stock_after_30d",

    "safety_stock_gap",
    "reorder_point_gap",

    "target_stock",

    "suggested_reorder_qty",

    "inventory_risk",
    "stockout_risk",

    "reorder_status",

    "business_priority",
    "priority_score",

    "business_action",
]


recommendations = df[main_columns].copy()


recommendation_file = (
    OUTPUT_PATH
    / "inventory_recommendations.csv"
)

recommendations.to_csv(
    recommendation_file,
    index=False
)

print(
    f"Saved: {recommendation_file}"
)


# ============================================================
# HIGH STOCKOUT RISK
# ============================================================

section("CREATING HIGH STOCKOUT RISK ITEMS")

high_risk = recommendations[
    recommendations["stockout_risk"]
    .isin(["CRITICAL", "HIGH"])
].copy()

high_risk_file = (
    OUTPUT_PATH
    / "high_stockout_risk.csv"
)

high_risk.to_csv(
    high_risk_file,
    index=False
)

print(
    "High-risk items:",
    len(high_risk)
)

print(
    f"Saved: {high_risk_file}"
)


# ============================================================
# REORDER RECOMMENDATIONS
# ============================================================

section("CREATING REORDER RECOMMENDATIONS")

reorder = recommendations[
    recommendations["suggested_reorder_qty"] > 0
].copy()

reorder = reorder.sort_values(
    "suggested_reorder_qty",
    ascending=False
)

reorder_file = (
    OUTPUT_PATH
    / "reorder_recommendations.csv"
)

reorder.to_csv(
    reorder_file,
    index=False
)

print(
    "Reorder items:",
    len(reorder)
)

print(
    f"Saved: {reorder_file}"
)


# ============================================================
# OVERSTOCK
# ============================================================

section("CREATING OVERSTOCK RECOMMENDATIONS")

overstock = recommendations[
    recommendations["inventory_risk"]
    .isin(
        [
            "OVERSTOCK",
            "NO_DEMAND_OVERSTOCK"
        ]
    )
].copy()

overstock = overstock.sort_values(
    "days_of_inventory",
    ascending=False
)

overstock_file = (
    OUTPUT_PATH
    / "overstock_recommendations.csv"
)

overstock.to_csv(
    overstock_file,
    index=False
)

print(
    "Overstock items:",
    len(overstock)
)

print(
    f"Saved: {overstock_file}"
)


# ============================================================
# STORE SUMMARY
# ============================================================

section("CREATING STORE INVENTORY SUMMARY")

store_summary = (
    recommendations
    .groupby("store_id")
    .agg(
        total_stock=(
            "stock_on_hand",
            "sum"
        ),

        forecast_30d=(
            "forecast_30d_units",
            "sum"
        ),

        forecast_60d=(
            "forecast_60d_units",
            "sum"
        ),

        forecast_90d=(
            "forecast_90d_units",
            "sum"
        ),

        suggested_reorder_qty=(
            "suggested_reorder_qty",
            "sum"
        ),

        inventory_items=(
            "sku_id",
            "count"
        ),

        high_risk_items=(
            "inventory_risk",
            lambda x:
            (x == "HIGH").sum()
        ),

        critical_items=(
            "business_priority",
            lambda x:
            (x == "CRITICAL").sum()
        ),

        overstock_items=(
            "inventory_risk",
            lambda x:
            x.isin(
                [
                    "OVERSTOCK",
                    "NO_DEMAND_OVERSTOCK"
                ]
            ).sum()
        ),
    )
    .reset_index()
)


store_summary = store_summary.sort_values(
    "suggested_reorder_qty",
    ascending=False
)


store_summary_file = (
    OUTPUT_PATH
    / "store_inventory_summary.csv"
)

store_summary.to_csv(
    store_summary_file,
    index=False
)

print(
    f"Saved: {store_summary_file}"
)


# ============================================================
# SKU SUMMARY
# ============================================================

section("CREATING SKU INVENTORY SUMMARY")

sku_summary = (
    recommendations
    .groupby("sku_id")
    .agg(
        total_stock=(
            "stock_on_hand",
            "sum"
        ),

        forecast_30d=(
            "forecast_30d_units",
            "sum"
        ),

        forecast_60d=(
            "forecast_60d_units",
            "sum"
        ),

        forecast_90d=(
            "forecast_90d_units",
            "sum"
        ),

        suggested_reorder_qty=(
            "suggested_reorder_qty",
            "sum"
        ),

        stores=(
            "store_id",
            "nunique"
        ),

        high_risk_items=(
            "inventory_risk",
            lambda x:
            (x == "HIGH").sum()
        ),

        critical_items=(
            "business_priority",
            lambda x:
            (x == "CRITICAL").sum()
        ),

        overstock_items=(
            "inventory_risk",
            lambda x:
            x.isin(
                [
                    "OVERSTOCK",
                    "NO_DEMAND_OVERSTOCK"
                ]
            ).sum()
        ),
    )
    .reset_index()
)


sku_summary = sku_summary.sort_values(
    "suggested_reorder_qty",
    ascending=False
)


sku_summary_file = (
    OUTPUT_PATH
    / "sku_inventory_summary.csv"
)

sku_summary.to_csv(
    sku_summary_file,
    index=False
)

print(
    f"Saved: {sku_summary_file}"
)


# ============================================================
# INVENTORY SUMMARY METRICS
# ============================================================

section("CREATING INVENTORY SUMMARY")

total_items = len(recommendations)

total_stock = (
    recommendations["stock_on_hand"]
    .sum()
)

total_forecast_30 = (
    recommendations["forecast_30d_units"]
    .sum()
)

total_forecast_60 = (
    recommendations["forecast_60d_units"]
    .sum()
)

total_forecast_90 = (
    recommendations["forecast_90d_units"]
    .sum()
)

total_reorder = (
    recommendations["suggested_reorder_qty"]
    .sum()
)

critical_count = (
    recommendations["business_priority"]
    .eq("CRITICAL")
    .sum()
)

high_count = (
    recommendations["business_priority"]
    .eq("HIGH")
    .sum()
)

medium_count = (
    recommendations["business_priority"]
    .eq("MEDIUM")
    .sum()
)

overstock_count = (
    recommendations["inventory_risk"]
    .isin(
        [
            "OVERSTOCK",
            "NO_DEMAND_OVERSTOCK"
        ]
    )
    .sum()
)

no_demand_count = (
    recommendations["inventory_risk"]
    .eq("NO_DEMAND")
    .sum()
)


summary = pd.DataFrame({

    "metric": [

        "total_inventory_items",

        "total_stock_on_hand",

        "total_30d_forecast",

        "total_60d_forecast",

        "total_90d_forecast",

        "total_suggested_reorder_qty",

        "critical_items",

        "high_priority_items",

        "medium_priority_items",

        "overstock_items",

        "no_demand_items",
    ],

    "value": [

        total_items,

        total_stock,

        total_forecast_30,

        total_forecast_60,

        total_forecast_90,

        total_reorder,

        critical_count,

        high_count,

        medium_count,

        overstock_count,

        no_demand_count,
    ]
})


summary_file = (
    OUTPUT_PATH
    / "inventory_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"Saved: {summary_file}"
)


# ============================================================
# TOP REORDER ITEMS
# ============================================================

section("TOP 20 REORDER RECOMMENDATIONS")

print(
    reorder[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "forecast_30d_units",
            "days_of_inventory",
            "suggested_reorder_qty",
            "inventory_risk",
            "business_priority"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TOP STOCKOUT RISK
# ============================================================

section("TOP STOCKOUT RISK ITEMS")

print(
    high_risk[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "forecast_30d_units",
            "days_of_inventory",
            "stockout_risk",
            "suggested_reorder_qty"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# BUSINESS REPORT
# ============================================================

section("CREATING INVENTORY RECOMMENDATIONS REPORT")

report_file = (
    OUTPUT_PATH
    / "inventory_recommendations_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT FORESIGHT\n"
    )

    f.write(
        "PHASE 6.3 - INVENTORY RECOMMENDATIONS\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Total inventory items: "
        f"{total_items:,}\n"
    )

    f.write(
        f"Total stock on hand: "
        f"{total_stock:,.2f}\n"
    )

    f.write(
        f"30-day forecast: "
        f"{total_forecast_30:,.2f}\n"
    )

    f.write(
        f"60-day forecast: "
        f"{total_forecast_60:,.2f}\n"
    )

    f.write(
        f"90-day forecast: "
        f"{total_forecast_90:,.2f}\n"
    )

    f.write(
        f"Suggested reorder quantity: "
        f"{total_reorder:,.2f}\n\n"
    )

    f.write(
        "RISK DISTRIBUTION\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        f"Critical items: "
        f"{critical_count:,}\n"
    )

    f.write(
        f"High priority items: "
        f"{high_count:,}\n"
    )

    f.write(
        f"Medium priority items: "
        f"{medium_count:,}\n"
    )

    f.write(
        f"Overstock items: "
        f"{overstock_count:,}\n"
    )

    f.write(
        f"No-demand items: "
        f"{no_demand_count:,}\n\n"
    )

    f.write(
        "BUSINESS INTERPRETATION\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        "Inventory recommendations combine "
        "current stock levels, reorder points, "
        "safety stock and the latest demand forecasts.\n\n"
    )

    f.write(
        "Critical items should receive immediate "
        "inventory attention.\n"
    )

    f.write(
        "High-priority items should be considered "
        "for near-term replenishment.\n"
    )

    f.write(
        "Medium-priority items should be monitored "
        "and replenished according to operational lead time.\n"
    )

    f.write(
        "Overstocked items should not receive "
        "unnecessary replenishment and may require "
        "inventory reduction or redistribution.\n"
    )


print(
    f"Business report saved:\n{report_file}"
)


# ============================================================
# FINAL
# ============================================================

section("PHASE 6.3 COMPLETED")

print(
    "Inventory recommendations completed successfully."
)

print("\nOutputs saved to:")

print(OUTPUT_PATH)

print("\nKey outputs:")

print("1. inventory_recommendations.csv")
print("2. high_stockout_risk.csv")
print("3. reorder_recommendations.csv")
print("4. overstock_recommendations.csv")
print("5. inventory_summary.csv")
print("6. store_inventory_summary.csv")
print("7. sku_inventory_summary.csv")
print("8. inventory_recommendations_report.txt")

print(
    "\nNEXT PHASE: EXECUTIVE BUSINESS DASHBOARD / FINAL PROJECT INTEGRATION"
)