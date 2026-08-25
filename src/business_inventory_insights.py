# ============================================================
# PROJECT FORESIGHT
# PHASE 7.4 - BUSINESS INVENTORY INSIGHTS
#
# Purpose:
# Convert validated forecast + inventory analysis into
# business-facing inventory insights and recommendations.
#
# Input:
# corrected_inventory_risk_reorder_recommendations.csv
# Phase 7.3 validation outputs
#
# Output:
# Business-level inventory insights
# Store-level insights
# SKU-level insights
# Overstock analysis
# Dormant inventory analysis
# Business action recommendations
# Executive summary
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
    / "inventory_risk"
    / "corrected_inventory_risk_reorder_recommendations.csv"
)

VALIDATION_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "inventory_risk"
    / "validation"
)

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(df, columns):

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    return df


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 7.4 - BUSINESS INVENTORY INSIGHTS")
print("=" * 70)


# ============================================================
# CHECK INPUT FILE
# ============================================================

print_section(
    "CHECKING INPUT FILE"
)

print(INPUT_FILE)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

print("FOUND")


# ============================================================
# LOAD DATA
# ============================================================

print_section(
    "LOADING VALIDATED INVENTORY DATA"
)

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


print_section(
    "CHECKING REQUIRED COLUMNS"
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing."
    )

print("All required columns FOUND")


# ============================================================
# PREPARE NUMERIC COLUMNS
# ============================================================

print_section(
    "PREPARING NUMERIC COLUMNS"
)

numeric_columns = [

    "stock_on_hand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "suggested_reorder_quantity",

    "historical_daily_demand_30d",
    "historical_daily_demand_90d",

    "planning_safety_stock",
    "planning_reorder_point",
    "planning_target_stock",

    "stock_after_30d_planning_demand",
    "stock_after_60d_planning_demand",
    "stock_after_90d_planning_demand",

    "inventory_gap_to_target",

    "forecast_vs_planning_ratio",
    "stock_to_planning_30d_ratio"
]

df = safe_numeric(
    df,
    numeric_columns
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print_section(
    "BASIC DATA QUALITY CHECK"
)

duplicate_count = df.duplicated(
    subset=["store_id", "sku_id"]
).sum()

negative_stock = (
    df["stock_on_hand"] < 0
).sum()

negative_forecast = (
    (
        df["calibrated_forecast_30d"] < 0
    )
    |
    (
        df["calibrated_forecast_60d"] < 0
    )
    |
    (
        df["calibrated_forecast_90d"] < 0
    )
).sum()

negative_reorder = (
    df["suggested_reorder_quantity"] < 0
).sum()

missing_planning_demand = (
    df["planning_daily_demand"].isna()
).sum()


print(
    f"Duplicate Store-SKU: {duplicate_count:,}"
)

print(
    f"Negative stock: {negative_stock:,}"
)

print(
    f"Negative forecast rows: {negative_forecast:,}"
)

print(
    f"Negative reorder quantity: {negative_reorder:,}"
)

print(
    f"Missing planning demand: {missing_planning_demand:,}"
)


if duplicate_count > 0:

    raise ValueError(
        "Duplicate Store-SKU combinations detected."
    )

if negative_stock > 0:

    raise ValueError(
        "Negative inventory detected."
    )

if negative_forecast > 0:

    raise ValueError(
        "Negative forecast detected."
    )


# ============================================================
# OVERALL BUSINESS METRICS
# ============================================================

print_section(
    "CALCULATING EXECUTIVE INVENTORY METRICS"
)


total_store_sku = len(df)

total_stock = (
    df["stock_on_hand"].sum()
)

forecast_30 = (
    df["calibrated_forecast_30d"].sum()
)

forecast_60 = (
    df["calibrated_forecast_60d"].sum()
)

forecast_90 = (
    df["calibrated_forecast_90d"].sum()
)

planning_daily = (
    df["planning_daily_demand"].sum()
)

planning_30 = (
    planning_daily * 30
)

planning_60 = (
    planning_daily * 60
)

planning_90 = (
    planning_daily * 90
)


# ============================================================
# INVENTORY COVERAGE
# ============================================================

inventory_to_forecast_30 = (
    total_stock / forecast_30
    if forecast_30 > 0
    else np.inf
)

inventory_to_forecast_60 = (
    total_stock / forecast_60
    if forecast_60 > 0
    else np.inf
)

inventory_to_forecast_90 = (
    total_stock / forecast_90
    if forecast_90 > 0
    else np.inf
)


planning_inventory_days = (
    total_stock / planning_daily
    if planning_daily > 0
    else np.inf
)


# ============================================================
# EXCESS INVENTORY
# ============================================================

excess_vs_30d = max(
    total_stock - forecast_30,
    0
)

excess_vs_60d = max(
    total_stock - forecast_60,
    0
)

excess_vs_90d = max(
    total_stock - forecast_90,
    0
)


# ============================================================
# OVERSTOCK ANALYSIS
# ============================================================

print_section(
    "ANALYZING OVERSTOCK"
)


extreme_overstock_threshold = 365

extreme_overstock = df[
    df["planning_days_of_inventory"]
    > extreme_overstock_threshold
].copy()


severe_overstock_count = (
    (
        df["planning_inventory_risk"]
        == "SEVERE_OVERSTOCK"
    )
    .sum()
)


print(
    f"Severe overstock Store-SKU: "
    f"{severe_overstock_count:,}"
)

print(
    f"Store-SKU >365 days inventory: "
    f"{len(extreme_overstock):,}"
)


overstock_inventory = (
    extreme_overstock[
        "stock_on_hand"
    ].sum()
)


# ============================================================
# NO FORECAST / DORMANT ANALYSIS
# ============================================================

print_section(
    "ANALYZING NO-FORECAST INVENTORY"
)


no_forecast = df[
    df["planning_stockout_risk"]
    == "NO_FORECAST_DEMAND"
].copy()


no_forecast_count = len(
    no_forecast
)


no_forecast_inventory = (
    no_forecast[
        "stock_on_hand"
    ].sum()
)


no_forecast_30d = (
    no_forecast[
        "calibrated_forecast_30d"
    ].sum()
)


print(
    f"No-forecast Store-SKU: "
    f"{no_forecast_count:,}"
)

print(
    f"Inventory held by no-forecast combinations: "
    f"{no_forecast_inventory:,.2f}"
)

print(
    f"30-day forecast from no-forecast combinations: "
    f"{no_forecast_30d:,.2f}"
)


# ============================================================
# REORDER ANALYSIS
# ============================================================

print_section(
    "ANALYZING REPLENISHMENT"
)


total_reorder = (
    df[
        "suggested_reorder_quantity"
    ].sum()
)


reorder_count = (
    (
        df[
            "suggested_reorder_quantity"
        ] > 0
    )
    .sum()
)


print(
    f"Total suggested reorder quantity: "
    f"{total_reorder:,.2f}"
)

print(
    f"Store-SKU requiring reorder: "
    f"{reorder_count:,}"
)


# ============================================================
# FORECAST VS PLANNING DEMAND
# ============================================================

print_section(
    "FORECAST VS PLANNING DEMAND"
)


forecast_planning_ratio = (
    forecast_30 / planning_30
    if planning_30 > 0
    else 0
)


forecast_planning_difference = (
    forecast_30 - planning_30
)


print(
    f"Calibrated 30-day forecast: "
    f"{forecast_30:,.2f}"
)

print(
    f"Planning 30-day demand: "
    f"{planning_30:,.2f}"
)

print(
    f"Forecast / Planning ratio: "
    f"{forecast_planning_ratio:.4f}"
)

print(
    f"Forecast - Planning difference: "
    f"{forecast_planning_difference:,.2f}"
)


# ============================================================
# CREATE STORE-LEVEL BUSINESS ANALYSIS
# ============================================================

print_section(
    "CREATING STORE-LEVEL BUSINESS INSIGHTS"
)


store_summary = (
    df
    .groupby("store_id")
    .agg(

        store_sku_count=(
            "sku_id",
            "count"
        ),

        stock_on_hand=(
            "stock_on_hand",
            "sum"
        ),

        calibrated_forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),

        calibrated_forecast_60d=(
            "calibrated_forecast_60d",
            "sum"
        ),

        calibrated_forecast_90d=(
            "calibrated_forecast_90d",
            "sum"
        ),

        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),

        severe_overstock_count=(
            "planning_inventory_risk",
            lambda x:
            (x == "SEVERE_OVERSTOCK").sum()
        ),

        no_forecast_count=(
            "planning_stockout_risk",
            lambda x:
            (x == "NO_FORECAST_DEMAND").sum()
        ),

        reorder_count=(
            "suggested_reorder_quantity",
            lambda x:
            (x > 0).sum()
        ),

        suggested_reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        )

    )
    .reset_index()
)


# ------------------------------------------------------------
# STORE METRICS
# ------------------------------------------------------------

store_summary[
    "planning_30d_demand"
] = (
    store_summary[
        "planning_daily_demand"
    ] * 30
)


store_summary[
    "stock_to_forecast_30d_ratio"
] = np.where(

    store_summary[
        "calibrated_forecast_30d"
    ] > 0,

    store_summary[
        "stock_on_hand"
    ]
    /
    store_summary[
        "calibrated_forecast_30d"
    ],

    np.inf
)


store_summary[
    "inventory_coverage_days"
] = np.where(

    store_summary[
        "planning_daily_demand"
    ] > 0,

    store_summary[
        "stock_on_hand"
    ]
    /
    store_summary[
        "planning_daily_demand"
    ],

    np.inf
)


store_summary[
    "forecast_coverage_days"
] = np.where(

    store_summary[
        "calibrated_forecast_30d"
    ] > 0,

    (
        store_summary[
            "stock_on_hand"
        ]
        /
        (
            store_summary[
                "calibrated_forecast_30d"
            ] / 30
        )
    ),

    np.inf
)


# ------------------------------------------------------------
# STORE BUSINESS CLASSIFICATION
# ------------------------------------------------------------

def classify_store(row):

    if row[
        "no_forecast_count"
    ] > 0 and row[
        "severe_overstock_count"
    ] > 0:

        return "OVERSTOCK_AND_DORMANT_RISK"

    if row[
        "no_forecast_count"
    ] > 0:

        return "DORMANT_INVENTORY_REVIEW"

    if row[
        "severe_overstock_count"
    ] > 0:

        return "SEVERE_OVERSTOCK"

    if row[
        "reorder_count"
    ] > 0:

        return "REPLENISHMENT_REQUIRED"

    return "NORMAL"


store_summary[
    "business_status"
] = store_summary.apply(
    classify_store,
    axis=1
)


def store_action(row):

    if row[
        "business_status"
    ] == "OVERSTOCK_AND_DORMANT_RISK":

        return (
            "STOP_REPLENISHMENT; "
            "REVIEW_DORMANT_STOCK; "
            "CONSIDER_TRANSFER_OR_LIQUIDATION"
        )

    if row[
        "business_status"
    ] == "DORMANT_INVENTORY_REVIEW":

        return (
            "REVIEW_DORMANT_INVENTORY; "
            "CONSIDER_TRANSFER_OR_LIQUIDATION"
        )

    if row[
        "business_status"
    ] == "SEVERE_OVERSTOCK":

        return (
            "STOP_REPLENISHMENT; "
            "REVIEW_STOCK_LEVELS"
        )

    if row[
        "business_status"
    ] == "REPLENISHMENT_REQUIRED":

        return (
            "REVIEW_REORDER_REQUIREMENT"
        )

    return "MONITOR"


store_summary[
    "business_action"
] = store_summary.apply(
    store_action,
    axis=1
)


store_summary = store_summary.sort_values(
    "stock_to_forecast_30d_ratio",
    ascending=False
)


# ============================================================
# SKU-LEVEL BUSINESS ANALYSIS
# ============================================================

print_section(
    "CREATING SKU-LEVEL BUSINESS INSIGHTS"
)


sku_summary = (
    df
    .groupby("sku_id")
    .agg(

        store_count=(
            "store_id",
            "nunique"
        ),

        stock_on_hand=(
            "stock_on_hand",
            "sum"
        ),

        calibrated_forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),

        calibrated_forecast_60d=(
            "calibrated_forecast_60d",
            "sum"
        ),

        calibrated_forecast_90d=(
            "calibrated_forecast_90d",
            "sum"
        ),

        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),

        severe_overstock_count=(
            "planning_inventory_risk",
            lambda x:
            (x == "SEVERE_OVERSTOCK").sum()
        ),

        no_forecast_count=(
            "planning_stockout_risk",
            lambda x:
            (x == "NO_FORECAST_DEMAND").sum()
        ),

        reorder_count=(
            "suggested_reorder_quantity",
            lambda x:
            (x > 0).sum()
        ),

        suggested_reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        )

    )
    .reset_index()
)


sku_summary[
    "planning_30d_demand"
] = (
    sku_summary[
        "planning_daily_demand"
    ] * 30
)


sku_summary[
    "stock_to_forecast_30d_ratio"
] = np.where(

    sku_summary[
        "calibrated_forecast_30d"
    ] > 0,

    sku_summary[
        "stock_on_hand"
    ]
    /
    sku_summary[
        "calibrated_forecast_30d"
    ],

    np.inf
)


sku_summary[
    "inventory_coverage_days"
] = np.where(

    sku_summary[
        "planning_daily_demand"
    ] > 0,

    sku_summary[
        "stock_on_hand"
    ]
    /
    sku_summary[
        "planning_daily_demand"
    ],

    np.inf
)


# ------------------------------------------------------------
# SKU BUSINESS CLASSIFICATION
# ------------------------------------------------------------

def classify_sku(row):

    if row[
        "no_forecast_count"
    ] > 0 and row[
        "severe_overstock_count"
    ] > 0:

        return "OVERSTOCK_AND_DORMANT"

    if row[
        "no_forecast_count"
    ] > 0:

        return "DORMANT_RISK"

    if row[
        "severe_overstock_count"
    ] > 0:

        return "OVERSTOCK"

    if row[
        "reorder_count"
    ] > 0:

        return "REPLENISHMENT_REQUIRED"

    return "NORMAL"


sku_summary[
    "business_status"
] = sku_summary.apply(
    classify_sku,
    axis=1
)


def sku_action(row):

    if row[
        "business_status"
    ] == "OVERSTOCK_AND_DORMANT":

        return (
            "STOP_REPLENISHMENT; "
            "REVIEW_TRANSFER_LIQUIDATION"
        )

    if row[
        "business_status"
    ] == "DORMANT_RISK":

        return (
            "REVIEW_DORMANT_STOCK"
        )

    if row[
        "business_status"
    ] == "OVERSTOCK":

        return (
            "STOP_REPLENISHMENT; "
            "REVIEW_STOCK"
        )

    if row[
        "business_status"
    ] == "REPLENISHMENT_REQUIRED":

        return (
            "REVIEW_REPLENISHMENT"
        )

    return "MONITOR"


sku_summary[
    "business_action"
] = sku_summary.apply(
    sku_action,
    axis=1
)


sku_summary = sku_summary.sort_values(
    "stock_to_forecast_30d_ratio",
    ascending=False
)


# ============================================================
# EXTREME OVERSTOCK BUSINESS FILE
# ============================================================

print_section(
    "CREATING EXTREME OVERSTOCK ANALYSIS"
)


overstock_columns = [

    "store_id",
    "sku_id",
    "stock_on_hand",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "calibrated_forecast_30d",

    "calibrated_forecast_60d",

    "calibrated_forecast_90d",

    "stock_to_planning_30d_ratio",

    "planning_inventory_risk",

    "business_action"
]


overstock_columns = [
    col
    for col in overstock_columns
    if col in extreme_overstock.columns
]


overstock_business = (
    extreme_overstock[
        overstock_columns
    ]
    .copy()
)


overstock_business[
    "excess_stock_vs_30d_forecast"
] = (
    overstock_business[
        "stock_on_hand"
    ]
    -
    overstock_business[
        "calibrated_forecast_30d"
    ]
)


overstock_business[
    "recommended_action"
] = (
    "STOP_REPLENISHMENT_REVIEW_STOCK"
)


overstock_business = overstock_business.sort_values(
    "planning_days_of_inventory",
    ascending=False
)


# ============================================================
# DORMANT INVENTORY BUSINESS FILE
# ============================================================

print_section(
    "CREATING DORMANT INVENTORY ANALYSIS"
)


dormant_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "planning_daily_demand",

    "planning_stockout_risk",

    "planning_inventory_risk",

    "business_action"
]


dormant_columns = [
    col
    for col in dormant_columns
    if col in no_forecast.columns
]


dormant_business = (
    no_forecast[
        dormant_columns
    ]
    .copy()
)


dormant_business[
    "recommended_action"
] = (
    "REVIEW_DORMANT_STOCK"
)


dormant_business = dormant_business.sort_values(
    "stock_on_hand",
    ascending=False
)


# ============================================================
# STORE TOP 20
# ============================================================

top_stores_by_stock = (
    store_summary
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
    .head(20)
    .copy()
)


top_stores_by_ratio = (
    store_summary
    .sort_values(
        "stock_to_forecast_30d_ratio",
        ascending=False
    )
    .head(20)
    .copy()
)


# ============================================================
# SKU TOP 20
# ============================================================

top_skus_by_stock = (
    sku_summary
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
    .head(20)
    .copy()
)


top_skus_by_ratio = (
    sku_summary
    .sort_values(
        "stock_to_forecast_30d_ratio",
        ascending=False
    )
    .head(20)
    .copy()
)


# ============================================================
# BUSINESS ACTION TABLE
# ============================================================

print_section(
    "CREATING BUSINESS ACTION RECOMMENDATIONS"
)


action_rows = []


# ------------------------------------------------------------
# ACTION 1
# ------------------------------------------------------------

action_rows.append({

    "priority": 1,

    "action_category":
        "REPLENISHMENT",

    "finding":
        "No additional replenishment is required "
        "under the current planning assumptions.",

    "metric":
        "Suggested reorder quantity",

    "metric_value":
        total_reorder,

    "affected_store_sku":
        reorder_count,

    "recommended_action":
        "PAUSE_NEW_REPLENISHMENT",

    "business_reason":
        "Current inventory substantially exceeds "
        "near-term demand."
})


# ------------------------------------------------------------
# ACTION 2
# ------------------------------------------------------------

action_rows.append({

    "priority": 1,

    "action_category":
        "OVERSTOCK",

    "finding":
        "Large number of Store-SKU combinations "
        "have more than one year of inventory coverage.",

    "metric":
        "Store-SKU >365 DOI",

    "metric_value":
        len(extreme_overstock),

    "affected_store_sku":
        len(extreme_overstock),

    "recommended_action":
        "STOP_REPLENISHMENT_AND_REVIEW_STOCK",

    "business_reason":
        "Excess stock creates working-capital and "
        "storage-cost pressure."
})


# ------------------------------------------------------------
# ACTION 3
# ------------------------------------------------------------

action_rows.append({

    "priority": 1,

    "action_category":
        "DORMANT_INVENTORY",

    "finding":
        "Inventory exists for Store-SKU combinations "
        "with no current forecast demand.",

    "metric":
        "No-forecast inventory",

    "metric_value":
        no_forecast_inventory,

    "affected_store_sku":
        no_forecast_count,

    "recommended_action":
        "REVIEW_DORMANT_STOCK",

    "business_reason":
        "Dormant inventory may require transfer, "
        "markdown, liquidation, or discontinuation review."
})


# ------------------------------------------------------------
# ACTION 4
# ------------------------------------------------------------

action_rows.append({

    "priority": 2,

    "action_category":
        "INVENTORY_REBALANCING",

    "finding":
        "Inventory is heavily concentrated relative "
        "to forecast demand.",

    "metric":
        "Inventory / 30-day forecast",

    "metric_value":
        inventory_to_forecast_30,

    "affected_store_sku":
        total_store_sku,

    "recommended_action":
        "EVALUATE_INTER_STORE_TRANSFER",

    "business_reason":
        "Inventory may be better allocated across "
        "locations based on relative demand."
})


# ------------------------------------------------------------
# ACTION 5
# ------------------------------------------------------------

action_rows.append({

    "priority": 2,

    "action_category":
        "DEMAND_MONITORING",

    "finding":
        "Planning demand remains higher than the "
        "calibrated forecast for the integrated inventory set.",

    "metric":
        "Planning / forecast ratio",

    "metric_value":
        (
            planning_30 / forecast_30
            if forecast_30 > 0
            else np.inf
        ),

    "affected_store_sku":
        total_store_sku,

    "recommended_action":
        "MONITOR_DEMAND_AND_REVIEW_FORECAST",

    "business_reason":
        "Planning assumptions incorporate historical "
        "demand and should be monitored against actual sales."
})


# ------------------------------------------------------------
# ACTION 6
# ------------------------------------------------------------

action_rows.append({

    "priority": 3,

    "action_category":
        "INVENTORY_OPTIMIZATION",

    "finding":
        "Inventory coverage is substantially above "
        "near-term demand requirements.",

    "metric":
        "Inventory coverage days",

    "metric_value":
        planning_inventory_days,

    "affected_store_sku":
        total_store_sku,

    "recommended_action":
        "REDUCE_EXCESS_INVENTORY_OVER_TIME",

    "business_reason":
        "Reducing excess stock can improve working-capital efficiency."
})


business_actions = pd.DataFrame(
    action_rows
)


# ============================================================
# EXECUTIVE SUMMARY TABLE
# ============================================================

print_section(
    "CREATING EXECUTIVE SUMMARY"
)


executive_summary = pd.DataFrame({

    "metric": [

        "Total Store-SKU combinations",

        "Total stock on hand",

        "Calibrated 30-day forecast",

        "Calibrated 60-day forecast",

        "Calibrated 90-day forecast",

        "Planning 30-day demand",

        "Planning 60-day demand",

        "Planning 90-day demand",

        "Inventory / 30-day forecast",

        "Inventory / 60-day forecast",

        "Inventory / 90-day forecast",

        "Planning inventory coverage days",

        "Store-SKU >365 days inventory",

        "Severe overstock Store-SKU",

        "No-forecast Store-SKU",

        "Inventory in no-forecast Store-SKU",

        "Suggested reorder quantity",

        "Store-SKU requiring reorder",

        "Forecast / planning 30-day ratio"

    ],

    "value": [

        total_store_sku,

        total_stock,

        forecast_30,

        forecast_60,

        forecast_90,

        planning_30,

        planning_60,

        planning_90,

        inventory_to_forecast_30,

        inventory_to_forecast_60,

        inventory_to_forecast_90,

        planning_inventory_days,

        len(extreme_overstock),

        severe_overstock_count,

        no_forecast_count,

        no_forecast_inventory,

        total_reorder,

        reorder_count,

        forecast_planning_ratio

    ]

})


# ============================================================
# OVERALL BUSINESS INTERPRETATION
# ============================================================

print_section(
    "GENERATING BUSINESS INTERPRETATION"
)


if inventory_to_forecast_30 >= 10:

    overall_inventory_status = (
        "SEVERE_OVERSTOCK"
    )

elif inventory_to_forecast_30 >= 5:

    overall_inventory_status = (
        "HIGH_INVENTORY"
    )

elif inventory_to_forecast_30 >= 2:

    overall_inventory_status = (
        "ELEVATED_INVENTORY"
    )

else:

    overall_inventory_status = (
        "BALANCED"
    )


if total_reorder == 0:

    replenishment_status = (
        "NO_ADDITIONAL_REPLENISHMENT_REQUIRED"
    )

else:

    replenishment_status = (
        "REPLENISHMENT_REQUIRED"
    )


if no_forecast_count > 0:

    dormant_status = (
        "DORMANT_INVENTORY_REVIEW_REQUIRED"
    )

else:

    dormant_status = (
        "NO_DORMANT_INVENTORY_FLAG"
    )


if len(extreme_overstock) > 0:

    overstock_status = (
        "EXCESS_INVENTORY_REVIEW_REQUIRED"
    )

else:

    overstock_status = (
        "NO_EXTREME_OVERSTOCK"
    )


# ============================================================
# BUSINESS INSIGHTS DATASET
# ============================================================

business_insights = pd.DataFrame({

    "insight_id": [

        "BI001",
        "BI002",
        "BI003",
        "BI004",
        "BI005",
        "BI006"

    ],

    "insight_category": [

        "INVENTORY_POSITION",
        "OVERSTOCK",
        "DORMANT_INVENTORY",
        "REPLENISHMENT",
        "INVENTORY_REBALANCING",
        "FORECAST_PLANNING"

    ],

    "status": [

        overall_inventory_status,
        overstock_status,
        dormant_status,
        replenishment_status,
        "REVIEW_REQUIRED",
        "MONITOR"

    ],

    "key_metric": [

        inventory_to_forecast_30,
        len(extreme_overstock),
        no_forecast_inventory,
        total_reorder,
        inventory_to_forecast_30,
        forecast_planning_ratio

    ],

    "key_message": [

        (
            "Inventory is substantially higher "
            "than near-term forecast demand."
        ),

        (
            "A very large number of Store-SKU "
            "combinations have extreme inventory coverage."
        ),

        (
            "Some inventory is held against "
            "Store-SKU combinations with no forecast demand."
        ),

        (
            "Current planning assumptions do not "
            "support additional replenishment."
        ),

        (
            "Inventory should be evaluated for "
            "rebalancing across stores."
        ),

        (
            "Planning demand is based on historical "
            "demand plus calibrated forecast and should "
            "be monitored against actual sales."
        )

    ],

    "recommended_action": [

        "REDUCE_EXCESS_INVENTORY",

        "STOP_REPLENISHMENT_REVIEW_STOCK",

        "REVIEW_TRANSFER_MARKDOWN_LIQUIDATION",

        "PAUSE_NEW_PURCHASES",

        "EVALUATE_INTER_STORE_TRANSFER",

        "MONITOR_ACTUAL_DEMAND"

    ]

})


# ============================================================
# PRINT EXECUTIVE RESULTS
# ============================================================

print_section(
    "PHASE 7.4 EXECUTIVE BUSINESS SUMMARY"
)


print(
    f"Total Store-SKU: "
    f"{total_store_sku:,}"
)

print(
    f"Total stock on hand: "
    f"{total_stock:,.2f}"
)

print(
    f"Calibrated 30-day forecast: "
    f"{forecast_30:,.2f}"
)

print(
    f"Planning 30-day demand: "
    f"{planning_30:,.2f}"
)

print(
    f"Inventory / 30-day forecast: "
    f"{inventory_to_forecast_30:.2f}x"
)

print(
    f"Inventory / 60-day forecast: "
    f"{inventory_to_forecast_60:.2f}x"
)

print(
    f"Inventory / 90-day forecast: "
    f"{inventory_to_forecast_90:.2f}x"
)

print(
    f"Planning inventory coverage: "
    f"{planning_inventory_days:,.2f} days"
)

print(
    f"Store-SKU >365 DOI: "
    f"{len(extreme_overstock):,}"
)

print(
    f"No-forecast Store-SKU: "
    f"{no_forecast_count:,}"
)

print(
    f"No-forecast inventory: "
    f"{no_forecast_inventory:,.2f}"
)

print(
    f"Suggested reorder quantity: "
    f"{total_reorder:,.2f}"
)

print(
    f"Store-SKU requiring reorder: "
    f"{reorder_count:,}"
)


# ============================================================
# TOP 20 STORES
# ============================================================

print_section(
    "TOP 20 STORES BY INVENTORY"
)

print(
    top_stores_by_stock[
        [
            "store_id",
            "store_sku_count",
            "stock_on_hand",
            "calibrated_forecast_30d",
            "planning_daily_demand",
            "severe_overstock_count",
            "no_forecast_count",
            "stock_to_forecast_30d_ratio",
            "business_status"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# TOP 20 SKUS
# ============================================================

print_section(
    "TOP 20 SKUS BY INVENTORY"
)

print(
    top_skus_by_stock[
        [
            "sku_id",
            "store_count",
            "stock_on_hand",
            "calibrated_forecast_30d",
            "planning_daily_demand",
            "severe_overstock_count",
            "no_forecast_count",
            "stock_to_forecast_30d_ratio",
            "business_status"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE FILES
# ============================================================

print_section(
    "SAVING BUSINESS INSIGHT FILES"
)


# ------------------------------------------------------------
# 1. Overall business insights
# ------------------------------------------------------------

business_insights_file = (
    OUTPUT_DIR
    / "business_inventory_insights.csv"
)

business_insights.to_csv(
    business_insights_file,
    index=False
)


# ------------------------------------------------------------
# 2. Executive summary
# ------------------------------------------------------------

executive_file = (
    OUTPUT_DIR
    / "business_inventory_executive_summary.csv"
)

executive_summary.to_csv(
    executive_file,
    index=False
)


# ------------------------------------------------------------
# 3. Store insights
# ------------------------------------------------------------

store_file = (
    OUTPUT_DIR
    / "store_inventory_business_insights.csv"
)

store_summary.to_csv(
    store_file,
    index=False
)


# ------------------------------------------------------------
# 4. SKU insights
# ------------------------------------------------------------

sku_file = (
    OUTPUT_DIR
    / "sku_inventory_business_insights.csv"
)

sku_summary.to_csv(
    sku_file,
    index=False
)


# ------------------------------------------------------------
# 5. Overstock
# ------------------------------------------------------------

overstock_file = (
    OUTPUT_DIR
    / "overstock_business_insights.csv"
)

overstock_business.to_csv(
    overstock_file,
    index=False
)


# ------------------------------------------------------------
# 6. Dormant inventory
# ------------------------------------------------------------

dormant_file = (
    OUTPUT_DIR
    / "dormant_inventory_business_insights.csv"
)

dormant_business.to_csv(
    dormant_file,
    index=False
)


# ------------------------------------------------------------
# 7. Business actions
# ------------------------------------------------------------

actions_file = (
    OUTPUT_DIR
    / "inventory_business_actions.csv"
)

business_actions.to_csv(
    actions_file,
    index=False
)


# ============================================================
# TEXT EXECUTIVE REPORT
# ============================================================

print_section(
    "CREATING EXECUTIVE TEXT REPORT"
)


report_file = (
    OUTPUT_DIR
    / "business_insights_summary.txt"
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
        "PHASE 7.4 - BUSINESS INVENTORY INSIGHTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "EXECUTIVE SUMMARY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Total Store-SKU combinations: "
        f"{total_store_sku:,}\n"
    )

    f.write(
        f"Total stock on hand: "
        f"{total_stock:,.2f}\n"
    )

    f.write(
        f"Calibrated 30-day forecast: "
        f"{forecast_30:,.2f}\n"
    )

    f.write(
        f"Calibrated 60-day forecast: "
        f"{forecast_60:,.2f}\n"
    )

    f.write(
        f"Calibrated 90-day forecast: "
        f"{forecast_90:,.2f}\n"
    )

    f.write(
        f"Planning 30-day demand: "
        f"{planning_30:,.2f}\n"
    )

    f.write(
        f"Inventory / 30-day forecast: "
        f"{inventory_to_forecast_30:.2f}x\n"
    )

    f.write(
        f"Inventory / 60-day forecast: "
        f"{inventory_to_forecast_60:.2f}x\n"
    )

    f.write(
        f"Inventory / 90-day forecast: "
        f"{inventory_to_forecast_90:.2f}x\n"
    )

    f.write(
        f"Planning inventory coverage: "
        f"{planning_inventory_days:,.2f} days\n"
    )

    f.write(
        f"Store-SKU >365 days inventory: "
        f"{len(extreme_overstock):,}\n"
    )

    f.write(
        f"No-forecast Store-SKU: "
        f"{no_forecast_count:,}\n"
    )

    f.write(
        f"Inventory in no-forecast Store-SKU: "
        f"{no_forecast_inventory:,.2f}\n"
    )

    f.write(
        f"Suggested reorder quantity: "
        f"{total_reorder:,.2f}\n"
    )

    f.write(
        f"Store-SKU requiring reorder: "
        f"{reorder_count:,}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "BUSINESS FINDINGS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "1. Inventory position:\n"
    )

    f.write(
        "   Inventory is substantially higher than "
        "near-term forecast demand.\n\n"
    )

    f.write(
        "2. Overstock:\n"
    )

    f.write(
        f"   {len(extreme_overstock):,} Store-SKU combinations "
        "have more than 365 days of inventory coverage.\n\n"
    )

    f.write(
        "3. Dormant inventory:\n"
    )

    f.write(
        f"   {no_forecast_count:,} Store-SKU combinations "
        "have no current forecast demand and hold "
        f"{no_forecast_inventory:,.2f} units of inventory.\n\n"
    )

    f.write(
        "4. Replenishment:\n"
    )

    f.write(
        "   No additional replenishment is recommended "
        "under the current planning assumptions.\n\n"
    )

    f.write(
        "5. Inventory optimization:\n"
    )

    f.write(
        "   Business teams should evaluate stock reduction, "
        "inter-store transfers, markdowns, liquidation, "
        "or SKU rationalization where appropriate.\n\n"
    )

    f.write(
        "6. Demand monitoring:\n"
    )

    f.write(
        "   Actual sales should be monitored against the "
        "calibrated forecast and planning demand before "
        "making major inventory decisions.\n\n"
    )

    f.write(
        "RECOMMENDED BUSINESS ACTIONS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "1. Pause unnecessary replenishment.\n"
    )

    f.write(
        "2. Review extreme overstock Store-SKU combinations.\n"
    )

    f.write(
        "3. Review dormant/no-forecast inventory.\n"
    )

    f.write(
        "4. Evaluate inter-store inventory transfers.\n"
    )

    f.write(
        "5. Consider markdown/liquidation strategies "
        "where commercially appropriate.\n"
    )

    f.write(
        "6. Monitor actual demand versus forecast.\n"
    )

    f.write(
        "7. Recalculate inventory recommendations when "
        "new demand observations become available.\n"
    )


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

print_section(
    "PHASE 7.4 DATA QUALITY CHECK"
)


quality_checks = {

    "input_rows":
        len(df),

    "duplicate_store_sku":
        duplicate_count,

    "negative_stock":
        negative_stock,

    "negative_forecast":
        negative_forecast,

    "negative_reorder":
        negative_reorder,

    "missing_planning_demand":
        missing_planning_demand,

    "business_insight_rows":
        len(business_insights),

    "store_summary_rows":
        len(store_summary),

    "sku_summary_rows":
        len(sku_summary),

    "overstock_rows":
        len(overstock_business),

    "dormant_rows":
        len(dormant_business),

    "business_action_rows":
        len(business_actions)

}


for key, value in quality_checks.items():

    print(
        f"{key:35}: {value:,}"
    )


if (
    duplicate_count == 0
    and negative_stock == 0
    and negative_forecast == 0
    and negative_reorder == 0
    and missing_planning_demand == 0
):

    quality_status = "PASS"

else:

    quality_status = "FAIL"


# ============================================================
# SAVE QUALITY STATUS
# ============================================================

quality_file = (
    OUTPUT_DIR
    / "business_insights_quality_check.csv"
)


quality_df = pd.DataFrame({

    "check": list(
        quality_checks.keys()
    ),

    "value": list(
        quality_checks.values()
    )

})


quality_df.to_csv(
    quality_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print_section(
    "PHASE 7.4 FINAL SUMMARY"
)

print(
    f"Business inventory status       : "
    f"{overall_inventory_status}"
)

print(
    f"Inventory / 30-day forecast     : "
    f"{inventory_to_forecast_30:.2f}x"
)

print(
    f"Extreme overstock Store-SKU     : "
    f"{len(extreme_overstock):,}"
)

print(
    f"No-forecast Store-SKU           : "
    f"{no_forecast_count:,}"
)

print(
    f"No-forecast inventory           : "
    f"{no_forecast_inventory:,.2f}"
)

print(
    f"Suggested reorder quantity      : "
    f"{total_reorder:,.2f}"
)

print(
    f"Store-SKU requiring reorder     : "
    f"{reorder_count:,}"
)

print(
    f"Data quality status             : "
    f"{quality_status}"
)


print_section(
    "OUTPUT FILES"
)

print(
    business_insights_file
)

print(
    executive_file
)

print(
    store_file
)

print(
    sku_file
)

print(
    overstock_file
)

print(
    dormant_file
)

print(
    actions_file
)

print(
    report_file
)

print(
    quality_file
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("PHASE 7.4 COMPLETED")
print("=" * 70)

print()
print(
    "Business inventory insights successfully generated."
)

print()
print(
    "IMPORTANT BUSINESS CONCLUSION:"
)

print(
    "Inventory is substantially higher than "
    "near-term forecast demand."
)

print()
print(
    "Recommended focus:"
)

print(
    "1. Reduce / control excess inventory"
)

print(
    "2. Review dormant inventory"
)

print(
    "3. Pause unnecessary replenishment"
)

print(
    "4. Evaluate inter-store transfers"
)

print(
    "5. Consider markdown / liquidation where appropriate"
)

print(
    "6. Monitor actual demand versus forecast"
)

print()
print(
    "Ready for:"
)

print(
    "PHASE 7.5 - BUSINESS RECOMMENDATION VALIDATION"
)

print()