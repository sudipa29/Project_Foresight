# ============================================================
# PROJECT FORESIGHT
# Phase 4 - Inventory Demand Risk Engine
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

INVENTORY_PATH = PROCESSED_PATH / "inventory_clean.csv"
DEMAND_PATH = PROCESSED_PATH / "daily_demand.csv"

OUTPUT_PATH = PROCESSED_PATH / "inventory_analysis"

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - INVENTORY DEMAND RISK ENGINE")
print("=" * 70)

print("\nLoading inventory data...")

inventory = pd.read_csv(
    INVENTORY_PATH,
    low_memory=False
)

print("Inventory shape:", inventory.shape)


print("\nLoading daily demand data...")

demand = pd.read_csv(
    DEMAND_PATH,
    low_memory=False
)

print("Daily demand shape:", demand.shape)


# ============================================================
# DATE CONVERSION
# ============================================================

inventory["snapshot_date"] = pd.to_datetime(
    inventory["snapshot_date"],
    errors="coerce"
)

inventory["last_restock_date"] = pd.to_datetime(
    inventory["last_restock_date"],
    errors="coerce"
)

demand["date"] = pd.to_datetime(
    demand["date"],
    errors="coerce"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)

print("\nInventory latest snapshot:")
print(inventory["snapshot_date"].max())

print("\nDemand latest date:")
print(demand["date"].max())

print("\nInventory stores:", inventory["store_id"].nunique())
print("Demand stores:", demand["store_id"].nunique())

print("\nInventory SKUs:", inventory["sku_id"].nunique())
print("Demand SKUs:", demand["sku_id"].nunique())


# ============================================================
# REFERENCE DATE
# ============================================================

reference_date = inventory["snapshot_date"].max()

print(
    "\nRisk analysis reference date:",
    reference_date.date()
)


# ============================================================
# CREATE DEMAND WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING DEMAND WINDOWS")
print("=" * 70)


demand["days_from_reference"] = (
    reference_date - demand["date"]
).dt.days


# ------------------------------------------------------------
# 7-DAY DEMAND
# ------------------------------------------------------------

demand_7d = demand[
    (demand["days_from_reference"] >= 0)
    &
    (demand["days_from_reference"] < 7)
].copy()


# ------------------------------------------------------------
# 30-DAY DEMAND
# ------------------------------------------------------------

demand_30d = demand[
    (demand["days_from_reference"] >= 0)
    &
    (demand["days_from_reference"] < 30)
].copy()


# ------------------------------------------------------------
# 90-DAY DEMAND
# ------------------------------------------------------------

demand_90d = demand[
    (demand["days_from_reference"] >= 0)
    &
    (demand["days_from_reference"] < 90)
].copy()


print("7-day demand rows:", len(demand_7d))
print("30-day demand rows:", len(demand_30d))
print("90-day demand rows:", len(demand_90d))


# ============================================================
# AGGREGATE DEMAND
# ============================================================

def aggregate_demand(df, column_name):

    result = (
        df.groupby(
            ["store_id", "sku_id"],
            as_index=False
        )["units_sold"]
        .sum()
        .rename(
            columns={
                "units_sold": column_name
            }
        )
    )

    return result


d7 = aggregate_demand(
    demand_7d,
    "units_7d"
)

d30 = aggregate_demand(
    demand_30d,
    "units_30d"
)

d90 = aggregate_demand(
    demand_90d,
    "units_90d"
)


# ============================================================
# MERGE DEMAND WINDOWS WITH INVENTORY
# ============================================================

analysis = inventory.copy()


analysis = analysis.merge(
    d7,
    on=["store_id", "sku_id"],
    how="left"
)


analysis = analysis.merge(
    d30,
    on=["store_id", "sku_id"],
    how="left"
)


analysis = analysis.merge(
    d90,
    on=["store_id", "sku_id"],
    how="left"
)


# ------------------------------------------------------------
# MISSING DEMAND = ZERO OBSERVED DEMAND
# ------------------------------------------------------------

analysis[
    [
        "units_7d",
        "units_30d",
        "units_90d"
    ]
] = analysis[
    [
        "units_7d",
        "units_30d",
        "units_90d"
    ]
].fillna(0)


# ============================================================
# DAILY DEMAND
# ============================================================

analysis["avg_daily_demand_7d"] = (
    analysis["units_7d"] / 7
)

analysis["avg_daily_demand_30d"] = (
    analysis["units_30d"] / 30
)

analysis["avg_daily_demand_90d"] = (
    analysis["units_90d"] / 90
)


# ============================================================
# DEMAND TREND
# ============================================================

analysis["demand_trend_ratio"] = np.where(
    analysis["avg_daily_demand_90d"] > 0,

    analysis["avg_daily_demand_7d"]
    /
    analysis["avg_daily_demand_90d"],

    np.nan
)


def classify_demand_trend(row):

    d7 = row["avg_daily_demand_7d"]
    d90 = row["avg_daily_demand_90d"]

    # --------------------------------------------------------
    # NO DEMAND
    # --------------------------------------------------------

    if d90 == 0 and d7 == 0:
        return "No Demand"

    # --------------------------------------------------------
    # RECENTLY STARTED
    # --------------------------------------------------------

    if d90 == 0 and d7 > 0:
        return "Recently Started"

    # --------------------------------------------------------
    # TREND RATIO
    # --------------------------------------------------------

    ratio = d7 / d90

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


analysis["demand_trend"] = analysis.apply(
    classify_demand_trend,
    axis=1
)


# ============================================================
# DAYS OF INVENTORY
# ============================================================

analysis["days_of_inventory"] = np.where(

    analysis["avg_daily_demand_30d"] > 0,

    analysis["stock_on_hand"]
    /
    analysis["avg_daily_demand_30d"],

    np.inf
)


# ============================================================
# REPORTING-FRIENDLY DOI
# ============================================================

analysis["days_of_inventory_capped"] = np.where(

    np.isfinite(
        analysis["days_of_inventory"]
    ),

    analysis["days_of_inventory"].clip(
        upper=365
    ),

    365
)


# ============================================================
# ZERO DEMAND INDICATOR
# ============================================================

analysis["zero_demand_30d"] = (
    analysis["units_30d"] == 0
)


# ============================================================
# STOCK COVERAGE AGAINST REORDER POINT
# ============================================================

analysis["stock_to_reorder_ratio"] = np.where(

    analysis["reorder_point"] > 0,

    analysis["stock_on_hand"]
    /
    analysis["reorder_point"],

    np.nan
)


# ============================================================
# STOCK COVERAGE AGAINST SAFETY STOCK
# ============================================================

analysis["stock_to_safety_ratio"] = np.where(

    analysis["safety_stock"] > 0,

    analysis["stock_on_hand"]
    /
    analysis["safety_stock"],

    np.nan
)


# ============================================================
# DEMAND / STOCK RATIO
# ============================================================

analysis["demand_30d_to_stock_ratio"] = np.where(

    analysis["stock_on_hand"] > 0,

    analysis["units_30d"]
    /
    analysis["stock_on_hand"],

    np.nan
)


# ============================================================
# RISK CATEGORY
# ============================================================

def classify_risk(row):

    stock = row["stock_on_hand"]

    reorder = row["reorder_point"]

    demand_30 = row["units_30d"]

    doi = row["days_of_inventory"]

    trend = row["demand_trend"]


    # --------------------------------------------------------
    # NO DEMAND
    # --------------------------------------------------------

    if demand_30 == 0:
        return "No Demand"


    # --------------------------------------------------------
    # CURRENT REORDER POINT BREACH
    # --------------------------------------------------------

    if stock <= reorder:
        return "Critical"


    # --------------------------------------------------------
    # VERY LOW INVENTORY COVERAGE
    # --------------------------------------------------------

    if doi < 14:
        return "Critical"


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if doi < 30:
        return "High"


    if (
        trend == "Strongly Increasing"
        and doi < 45
    ):
        return "High"


    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    if doi < 60:
        return "Medium"


    # --------------------------------------------------------
    # POTENTIAL OVERSTOCK
    # --------------------------------------------------------

    if doi > 120:
        return "Potential Overstock"


    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    return "Healthy"


analysis["risk_category"] = analysis.apply(
    classify_risk,
    axis=1
)


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(row):

    stock = row["stock_on_hand"]

    reorder = row["reorder_point"]

    demand_30 = row["units_30d"]

    doi = row["days_of_inventory"]

    trend = row["demand_trend"]


    score = 0


    # --------------------------------------------------------
    # NO DEMAND
    # --------------------------------------------------------

    if demand_30 == 0:

        score += 25


    # --------------------------------------------------------
    # REORDER POINT BREACH
    # --------------------------------------------------------

    if stock <= reorder:

        score += 50


    # --------------------------------------------------------
    # DAYS OF INVENTORY
    # --------------------------------------------------------

    if demand_30 > 0:

        if doi < 14:

            score += 40

        elif doi < 30:

            score += 30

        elif doi < 60:

            score += 15

        elif doi > 120:

            score += 20


    # --------------------------------------------------------
    # DEMAND TREND
    # --------------------------------------------------------

    if trend == "Strongly Increasing":

        score += 25

    elif trend == "Increasing":

        score += 15

    elif trend == "Recently Started":

        score += 20

    elif trend == "Strongly Decreasing":

        score += 5

    elif trend == "Decreasing":

        score += 3


    # --------------------------------------------------------
    # CAP SCORE
    # --------------------------------------------------------

    return min(
        100,
        score
    )


# ------------------------------------------------------------
# APPLY RISK SCORE
# ------------------------------------------------------------

analysis["risk_score"] = analysis.apply(
    calculate_risk_score,
    axis=1
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(row):

    category = row["risk_category"]

    trend = row["demand_trend"]


    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if category == "Critical":

        return "Urgent replenishment review"


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if category == "High":

        return "Prioritize replenishment"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if category == "Medium":

        return "Monitor inventory closely"


    # --------------------------------------------------------
    # POTENTIAL OVERSTOCK
    # --------------------------------------------------------

    if category == "Potential Overstock":

        return "Review excess inventory"


    # --------------------------------------------------------
    # NO DEMAND
    # --------------------------------------------------------

    if category == "No Demand":

        return "Review inactive SKU"


    # --------------------------------------------------------
    # RECENTLY STARTED
    # --------------------------------------------------------

    if trend == "Recently Started":

        return "Monitor newly emerging demand"


    # --------------------------------------------------------
    # STRONGLY INCREASING
    # --------------------------------------------------------

    if trend == "Strongly Increasing":

        return "Monitor demand acceleration"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "Normal monitoring"


# ------------------------------------------------------------
# APPLY RECOMMENDED ACTION
# ------------------------------------------------------------

analysis["recommended_action"] = analysis.apply(
    recommended_action,
    axis=1
)


# ============================================================
# INVENTORY STATUS FLAGS
# ============================================================

analysis["below_reorder_point"] = (

    analysis["stock_on_hand"]
    <=
    analysis["reorder_point"]
)


analysis["below_safety_stock"] = (

    analysis["stock_on_hand"]
    <=
    analysis["safety_stock"]
)


# ============================================================
# INVENTORY STATUS
# ============================================================

analysis["inventory_status"] = np.select(

    [
        analysis["stock_on_hand"]
        <=
        analysis["safety_stock"],

        analysis["stock_on_hand"]
        <=
        analysis["reorder_point"],

        analysis["risk_category"]
        ==
        "Potential Overstock",

        analysis["risk_category"]
        ==
        "No Demand"
    ],

    [
        "Below Safety Stock",

        "Below Reorder Point",

        "Potential Overstock",

        "No Recent Demand"
    ],

    default="Healthy"
)


# ============================================================
# OUTPUT COLUMN ORDER
# ============================================================

output_columns = [

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    "store_id",
    "sku_id",

    # --------------------------------------------------------
    # INVENTORY
    # --------------------------------------------------------

    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "last_restock_date",
    "snapshot_date",

    # --------------------------------------------------------
    # DEMAND WINDOWS
    # --------------------------------------------------------

    "units_7d",
    "units_30d",
    "units_90d",

    # --------------------------------------------------------
    # DAILY DEMAND
    # --------------------------------------------------------

    "avg_daily_demand_7d",
    "avg_daily_demand_30d",
    "avg_daily_demand_90d",

    # --------------------------------------------------------
    # DEMAND TREND
    # --------------------------------------------------------

    "demand_trend_ratio",
    "demand_trend",

    # --------------------------------------------------------
    # INVENTORY COVERAGE
    # --------------------------------------------------------

    "days_of_inventory",
    "days_of_inventory_capped",

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    "stock_to_reorder_ratio",
    "stock_to_safety_ratio",
    "demand_30d_to_stock_ratio",

    # --------------------------------------------------------
    # FLAGS
    # --------------------------------------------------------

    "zero_demand_30d",
    "below_reorder_point",
    "below_safety_stock",

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    "risk_category",
    "risk_score",

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    "recommended_action",

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    "inventory_status"
]


analysis = analysis[
    output_columns
]


# ============================================================
# SAVE COMPLETE ANALYSIS
# ============================================================

output_file = (
    OUTPUT_PATH
    /
    "inventory_demand_risk_analysis.csv"
)


analysis.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK CATEGORY DISTRIBUTION")
print("=" * 70)

print(
    analysis[
        "risk_category"
    ].value_counts()
)


# ============================================================
# DEMAND TREND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DEMAND TREND DISTRIBUTION")
print("=" * 70)

print(
    analysis[
        "demand_trend"
    ].value_counts()
)


# ============================================================
# RISK SCORE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK SCORE SUMMARY")
print("=" * 70)

print(
    analysis[
        "risk_score"
    ].describe()
)


# ============================================================
# INVENTORY STATUS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY STATUS DISTRIBUTION")
print("=" * 70)

print(
    analysis[
        "inventory_status"
    ].value_counts()
)


# ============================================================
# TOP 20 RISK ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 RISK ITEMS")
print("=" * 70)


top_risk_items = (

    analysis

    .sort_values(
        [
            "risk_score",
            "days_of_inventory"
        ],

        ascending=[
            False,
            True
        ]
    )

    .head(20)
)


print(
    top_risk_items[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "units_7d",
            "units_30d",
            "units_90d",
            "days_of_inventory",
            "days_of_inventory_capped",
            "demand_trend",
            "risk_category",
            "risk_score",
            "recommended_action"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# CRITICAL / HIGH ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 CRITICAL / HIGH RISK ITEMS")
print("=" * 70)


risk_items = (

    analysis[
        analysis[
            "risk_category"
        ].isin(
            [
                "Critical",
                "High"
            ]
        )
    ]

    .sort_values(
        [
            "risk_score",
            "days_of_inventory"
        ],

        ascending=[
            False,
            True
        ]
    )
)


if len(risk_items) == 0:

    print(
        "No Critical or High risk items identified."
    )

else:

    print(
        risk_items[
            [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "reorder_point",
                "safety_stock",
                "units_7d",
                "units_30d",
                "units_90d",
                "days_of_inventory",
                "demand_trend",
                "risk_category",
                "risk_score",
                "recommended_action"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# POTENTIAL OVERSTOCK ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 POTENTIAL OVERSTOCK ITEMS")
print("=" * 70)


overstock = (

    analysis[
        analysis[
            "risk_category"
        ]
        ==
        "Potential Overstock"
    ]

    .sort_values(
        "days_of_inventory",
        ascending=False
    )
)


if len(overstock) == 0:

    print(
        "No potential overstock items identified."
    )

else:

    print(
        overstock[
            [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "units_30d",
                "units_90d",
                "days_of_inventory",
                "days_of_inventory_capped",
                "demand_trend",
                "risk_category",
                "risk_score",
                "recommended_action"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# NO DEMAND ITEMS
# ============================================================

print("\n" + "=" * 70)
print("NO DEMAND INVENTORY ITEMS")
print("=" * 70)


no_demand = analysis[
    analysis[
        "risk_category"
    ]
    ==
    "No Demand"
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
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "units_7d",
            "units_30d",
            "units_90d",
            "days_of_inventory_capped",
            "demand_trend",
            "risk_category",
            "risk_score",
            "recommended_action"
        ]
    ]
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT VALIDATION")
print("=" * 70)

print(
    "Final analysis shape:",
    analysis.shape
)

print(
    "Risk score column exists:",
    "risk_score" in analysis.columns
)

print(
    "Recommended action column exists:",
    "recommended_action" in analysis.columns
)

print(
    "Inventory status column exists:",
    "inventory_status" in analysis.columns
)

print(
    "Missing risk scores:",
    analysis["risk_score"].isna().sum()
)

print(
    "Missing risk categories:",
    analysis["risk_category"].isna().sum()
)

print(
    "Missing recommended actions:",
    analysis["recommended_action"].isna().sum()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY DEMAND RISK ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput saved to:")

print(output_file)

print("\n" + "=" * 70)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)