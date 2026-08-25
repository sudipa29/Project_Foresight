# ============================================================
# PROJECT FORESIGHT
# PHASE 8.1 - FINAL BUSINESS ANALYTICS OUTPUT
#
# Purpose:
#   Create clean, dashboard-ready business analytics datasets
#   from the VALIDATED Phase 7.5 recommendations.
#
# Input:
#   validated_business_recommendations.csv
#
# Output:
#   data/processed/forecasting/business_insights/phase8/
#
# Dashboard Sections:
#   1. Executive Summary
#   2. Forecast Analysis
#   3. Inventory Risk
#   4. Overstock Analysis
#   5. Store Analysis
#   6. SKU Analysis
#   7. Replenishment
#   8. Business Actions
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

INPUT_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "validation"
    / "validated_business_recommendations.csv"
)

OUTPUT_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "phase8"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def save_csv(df, filename):

    path = OUTPUT_PATH / filename

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")

    return path


# ============================================================
# START
# ============================================================

section(
    "PROJECT FORESIGHT\n"
    "PHASE 8.1 - FINAL BUSINESS ANALYTICS OUTPUT"
)


# ============================================================
# CHECK INPUT
# ============================================================

section("CHECKING PHASE 7.5 VALIDATED INPUT")

print(INPUT_PATH)

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nValidated Phase 7.5 file not found:\n{INPUT_PATH}"
    )

print("FOUND")


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING VALIDATED BUSINESS RECOMMENDATIONS")

df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


# ============================================================
# BASIC STRUCTURE CHECK
# ============================================================

required_columns = [

    "store_id",
    "sku_id",
    "stock_on_hand",

    "planning_daily_demand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "calibrated_inventory_coverage_days",

    "planning_days_of_inventory",

    "suggested_reorder_quantity",

    "planning_reorder_status",

    "reorder_priority",

    "business_action",

    "demand_available",

    "planning_inventory_risk",

    "planning_stockout_risk",

]


missing_columns = [
    c for c in required_columns
    if c not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


print("All required columns FOUND")


# ============================================================
# NUMERIC PREPARATION
# ============================================================

section("PREPARING ANALYTICS DATA")


numeric_columns = [

    "stock_on_hand",

    "units_30d",
    "units_90d",

    "avg_daily_demand_30d",
    "avg_daily_demand_90d",

    "forecast_30d_units",
    "forecast_daily_demand",

    "planning_daily_demand",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "calibrated_avg_daily_forecast_30d",
    "calibrated_avg_daily_forecast_60d",
    "calibrated_avg_daily_forecast_90d",

    "calibrated_inventory_coverage_days",

    "planning_days_of_inventory",

    "planning_safety_stock",
    "planning_reorder_point",
    "planning_target_stock",

    "inventory_gap_to_target",

    "suggested_reorder_quantity",

    "planning_stockout_risk",

    "forecast_vs_planning_ratio",

    "stock_to_planning_30d_ratio",

]


for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)


print("Numeric preparation completed.")


# ============================================================
# NORMALIZE DEMAND AVAILABILITY
# ============================================================

section("NORMALIZING DEMAND AVAILABILITY")


def normalize_demand_available(value):

    if pd.isna(value):
        return "FALSE"

    value = str(value).strip().upper()

    if value in [
        "TRUE",
        "1",
        "YES",
        "Y"
    ]:
        return "TRUE"

    return "FALSE"


df["demand_available"] = (
    df["demand_available"]
    .apply(normalize_demand_available)
)


demand_available_count = int(
    (df["demand_available"] == "TRUE").sum()
)

no_forecast_count = int(
    (df["demand_available"] == "FALSE").sum()
)


print(
    f"Demand-available Store-SKU: "
    f"{demand_available_count:,}"
)

print(
    f"No-forecast Store-SKU:       "
    f"{no_forecast_count:,}"
)


# ============================================================
# DERIVED BUSINESS FIELDS
# ============================================================

section("CREATING BUSINESS ANALYTICS FIELDS")


# ------------------------------------------------------------
# Inventory Coverage Category
# ------------------------------------------------------------

df["inventory_coverage_category"] = np.select(

    [

        df["planning_days_of_inventory"] <= 30,

        df["planning_days_of_inventory"] <= 60,

        df["planning_days_of_inventory"] <= 90,

        df["planning_days_of_inventory"] <= 180,

        df["planning_days_of_inventory"] <= 365,

    ],

    [

        "0-30 DAYS",

        "31-60 DAYS",

        "61-90 DAYS",

        "91-180 DAYS",

        "181-365 DAYS",

    ],

    default=">365 DAYS"

)


# ------------------------------------------------------------
# Executive Inventory Flag
# ------------------------------------------------------------

df["executive_inventory_flag"] = np.select(

    [

        df["planning_days_of_inventory"] > 365,

        df["planning_days_of_inventory"] > 180,

        df["planning_days_of_inventory"] > 90,

        df["planning_days_of_inventory"] > 60,

        df["planning_days_of_inventory"] > 30,

    ],

    [

        "EXTREME OVERSTOCK",

        "SEVERE OVERSTOCK",

        "HIGH OVERSTOCK",

        "MODERATE OVERSTOCK",

        "ELEVATED INVENTORY",

    ],

    default="NORMAL"

)


# ------------------------------------------------------------
# Forecast / Planning Difference
# ------------------------------------------------------------

df["forecast_planning_difference_30d"] = (

    df["calibrated_forecast_30d"]
    -
    (
        df["planning_daily_demand"] * 30
    )

)


# ------------------------------------------------------------
# Inventory Units Proxy
# ------------------------------------------------------------

df["inventory_units"] = (
    df["stock_on_hand"]
)


# ------------------------------------------------------------
# Replenishment Decision
# ------------------------------------------------------------

df["replenishment_decision"] = np.select(

    [

        df["suggested_reorder_quantity"] > 0,

        df["planning_days_of_inventory"] > 365,

        df["planning_days_of_inventory"] > 180,

        df["planning_days_of_inventory"] > 90,

    ],

    [

        "REORDER",

        "DO NOT REORDER - EXTREME OVERSTOCK",

        "DO NOT REORDER - SEVERE OVERSTOCK",

        "DO NOT REORDER - HIGH INVENTORY",

    ],

    default="NO IMMEDIATE REORDER"

)


# ============================================================
# PORTFOLIO METRICS
# ============================================================

total_inventory = (
    df["stock_on_hand"].sum()
)


forecast_30d = (
    df["calibrated_forecast_30d"].sum()
)


forecast_60d = (
    df["calibrated_forecast_60d"].sum()
)


forecast_90d = (
    df["calibrated_forecast_90d"].sum()
)


planning_daily = (
    df["planning_daily_demand"].sum()
)


planning_30d = (
    planning_daily * 30
)


planning_60d = (
    planning_daily * 60
)


planning_90d = (
    planning_daily * 90
)


inventory_forecast_ratio = (

    total_inventory / forecast_30d

    if forecast_30d > 0

    else np.nan

)


inventory_planning_ratio = (

    total_inventory / planning_30d

    if planning_30d > 0

    else np.nan

)


over30_count = int(
    (
        df["planning_days_of_inventory"]
        > 30
    ).sum()
)


over60_count = int(
    (
        df["planning_days_of_inventory"]
        > 60
    ).sum()
)


over90_count = int(
    (
        df["planning_days_of_inventory"]
        > 90
    ).sum()
)


over180_count = int(
    (
        df["planning_days_of_inventory"]
        > 180
    ).sum()
)


extreme_overstock_count = int(
    (
        df["planning_days_of_inventory"]
        > 365
    ).sum()
)


severe_overstock_count = over90_count


no_forecast_inventory = (

    df.loc[
        df["demand_available"] == "FALSE",
        "stock_on_hand"
    ]
    .sum()

)


reorder_quantity = (
    df["suggested_reorder_quantity"]
    .sum()
)


reorder_count = int(
    (
        df["suggested_reorder_quantity"]
        > 0
    ).sum()
)


executive_status = (

    "CRITICAL_OVERSTOCK"

    if inventory_forecast_ratio > 10

    else "HIGH_INVENTORY"

    if inventory_forecast_ratio > 5

    else "NORMAL"

)


# ============================================================
# 8.1 EXECUTIVE SUMMARY DATASET
# IMPORTANT:
# Single-row wide format for dashboard validation.
# ============================================================

section("CREATING EXECUTIVE SUMMARY DATASET")


executive_summary = pd.DataFrame({

    "total_store_sku": [
        len(df)
    ],

    "total_stores": [
        df["store_id"].nunique()
    ],

    "total_skus": [
        df["sku_id"].nunique()
    ],

    "total_inventory_units": [
        total_inventory
    ],

    "forecast_30d": [
        forecast_30d
    ],

    "forecast_60d": [
        forecast_60d
    ],

    "forecast_90d": [
        forecast_90d
    ],

    "planning_daily_demand": [
        planning_daily
    ],

    "planning_30d_demand": [
        planning_30d
    ],

    "planning_60d_demand": [
        planning_60d
    ],

    "planning_90d_demand": [
        planning_90d
    ],

    "inventory_to_30d_forecast_ratio": [
        inventory_forecast_ratio
    ],

    "inventory_to_30d_planning_ratio": [
        inventory_planning_ratio
    ],

    "over30_doi_store_sku": [
        over30_count
    ],

    "over60_doi_store_sku": [
        over60_count
    ],

    "over90_doi_store_sku": [
        over90_count
    ],

    "over180_doi_store_sku": [
        over180_count
    ],

    "over365_doi_store_sku": [
        extreme_overstock_count
    ],

    "severe_overstock_count": [
        severe_overstock_count
    ],

    "no_forecast_store_sku": [
        no_forecast_count
    ],

    "no_forecast_inventory": [
        no_forecast_inventory
    ],

    "suggested_reorder_quantity": [
        reorder_quantity
    ],

    "store_sku_requiring_reorder": [
        reorder_count
    ],

    "business_inventory_status": [
        executive_status
    ],

})


save_csv(
    executive_summary,
    "phase8_executive_summary.csv"
)


# ============================================================
# 8.2 FORECAST ANALYSIS
# ============================================================

section("CREATING FORECAST ANALYSIS DATASET")


forecast_columns = [

    "store_id",
    "sku_id",

    "units_30d",
    "units_90d",

    "avg_daily_demand_30d",
    "avg_daily_demand_90d",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "calibrated_avg_daily_forecast_30d",
    "calibrated_avg_daily_forecast_60d",
    "calibrated_avg_daily_forecast_90d",

    "forecast_planning_difference_30d",

    "forecast_vs_planning_ratio",

    "demand_available",

]


forecast_analysis = (
    df[forecast_columns]
    .copy()
)


save_csv(
    forecast_analysis,
    "phase8_forecast_analysis.csv"
)


# ============================================================
# 8.3 INVENTORY RISK
# ============================================================

section("CREATING INVENTORY RISK DATASET")


inventory_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "planning_safety_stock",
    "planning_reorder_point",
    "planning_target_stock",

    "planning_days_of_inventory",

    "calibrated_inventory_coverage_days",

    "planning_inventory_risk",

    "planning_stockout_risk",

    "inventory_coverage_category",

    "executive_inventory_flag",

    "inventory_gap_to_target",

    "stock_after_30d_planning_demand",

    "stock_after_60d_planning_demand",

    "stock_after_90d_planning_demand",

]


inventory_risk = (
    df[inventory_columns]
    .copy()
)


save_csv(
    inventory_risk,
    "phase8_inventory_risk.csv"
)


# ============================================================
# 8.4 OVERSTOCK ANALYSIS
# ============================================================

section("CREATING OVERSTOCK ANALYSIS")


overstock_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "inventory_gap_to_target",

    "planning_inventory_risk",

    "business_action",

    "executive_inventory_flag",

]


overstock = (
    df[
        df["planning_days_of_inventory"] > 30
    ]
    [overstock_columns]
    .copy()
    .sort_values(
        "planning_days_of_inventory",
        ascending=False
    )
)


save_csv(
    overstock,
    "phase8_overstock_analysis.csv"
)


# ============================================================
# 8.5 EXTREME OVERSTOCK
# ============================================================

section("CREATING EXTREME OVERSTOCK DATASET")


extreme_overstock_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "inventory_gap_to_target",

    "business_action",

]


extreme_overstock = (
    df[
        df["planning_days_of_inventory"] > 365
    ]
    [extreme_overstock_columns]
    .copy()
    .sort_values(
        "planning_days_of_inventory",
        ascending=False
    )
)


save_csv(
    extreme_overstock,
    "phase8_extreme_overstock.csv"
)


# ============================================================
# 8.6 DORMANT INVENTORY
# ============================================================

section("CREATING DORMANT INVENTORY DATASET")


dormant_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "demand_available",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "planning_days_of_inventory",

    "business_action",

]


dormant = (
    df[
        df["demand_available"] == "FALSE"
    ]
    [dormant_columns]
    .copy()
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
)


print(
    f"Dormant / no-forecast rows: "
    f"{len(dormant):,}"
)

print(
    f"Expected no-forecast rows:   "
    f"{no_forecast_count:,}"
)


save_csv(
    dormant,
    "phase8_dormant_inventory.csv"
)


# ============================================================
# 8.7 STORE-LEVEL ANALYSIS
# ============================================================

section("CREATING STORE-LEVEL ANALYSIS")


store_analysis = (

    df

    .groupby(
        "store_id",
        as_index=False
    )

    .agg(

        store_inventory_units=(
            "stock_on_hand",
            "sum"
        ),

        store_30d_forecast=(
            "calibrated_forecast_30d",
            "sum"
        ),

        store_60d_forecast=(
            "calibrated_forecast_60d",
            "sum"
        ),

        store_90d_forecast=(
            "calibrated_forecast_90d",
            "sum"
        ),

        store_planning_demand=(
            "planning_daily_demand",
            "sum"
        ),

        store_reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        ),

        store_sku_count=(
            "sku_id",
            "nunique"
        ),

        extreme_overstock_count=(
            "planning_days_of_inventory",
            lambda x: int(
                (x > 365).sum()
            )
        ),

        severe_overstock_count=(
            "planning_days_of_inventory",
            lambda x: int(
                (x > 90).sum()
            )
        ),

        no_forecast_count=(
            "demand_available",
            lambda x: int(
                (x == "FALSE").sum()
            )
        ),

    )

)


store_analysis[
    "store_planning_30d_demand"
] = (

    store_analysis[
        "store_planning_demand"
    ]
    * 30

)


store_analysis[
    "inventory_to_forecast_ratio"
] = np.where(

    store_analysis[
        "store_30d_forecast"
    ] > 0,

    store_analysis[
        "store_inventory_units"
    ]
    /
    store_analysis[
        "store_30d_forecast"
    ],

    np.nan

)


store_analysis[
    "inventory_to_planning_ratio"
] = np.where(

    store_analysis[
        "store_planning_30d_demand"
    ] > 0,

    store_analysis[
        "store_inventory_units"
    ]
    /
    store_analysis[
        "store_planning_30d_demand"
    ],

    np.nan

)


store_analysis = (
    store_analysis
    .sort_values(
        "inventory_to_forecast_ratio",
        ascending=False
    )
)


save_csv(
    store_analysis,
    "phase8_store_analysis.csv"
)


# ============================================================
# 8.8 SKU-LEVEL ANALYSIS
# ============================================================

section("CREATING SKU-LEVEL ANALYSIS")


sku_analysis = (

    df

    .groupby(
        "sku_id",
        as_index=False
    )

    .agg(

        sku_inventory_units=(
            "stock_on_hand",
            "sum"
        ),

        sku_30d_forecast=(
            "calibrated_forecast_30d",
            "sum"
        ),

        sku_60d_forecast=(
            "calibrated_forecast_60d",
            "sum"
        ),

        sku_90d_forecast=(
            "calibrated_forecast_90d",
            "sum"
        ),

        sku_planning_demand=(
            "planning_daily_demand",
            "sum"
        ),

        sku_reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        ),

        store_count=(
            "store_id",
            "nunique"
        ),

        extreme_overstock_count=(
            "planning_days_of_inventory",
            lambda x: int(
                (x > 365).sum()
            )
        ),

        severe_overstock_count=(
            "planning_days_of_inventory",
            lambda x: int(
                (x > 90).sum()
            )
        ),

        no_forecast_count=(
            "demand_available",
            lambda x: int(
                (x == "FALSE").sum()
            )
        ),

    )

)


sku_analysis[
    "sku_planning_30d_demand"
] = (

    sku_analysis[
        "sku_planning_demand"
    ]
    * 30

)


sku_analysis[
    "inventory_to_forecast_ratio"
] = np.where(

    sku_analysis[
        "sku_30d_forecast"
    ] > 0,

    sku_analysis[
        "sku_inventory_units"
    ]
    /
    sku_analysis[
        "sku_30d_forecast"
    ],

    np.nan

)


sku_analysis[
    "inventory_to_planning_ratio"
] = np.where(

    sku_analysis[
        "sku_planning_30d_demand"
    ] > 0,

    sku_analysis[
        "sku_inventory_units"
    ]
    /
    sku_analysis[
        "sku_planning_30d_demand"
    ],

    np.nan

)


sku_analysis = (
    sku_analysis
    .sort_values(
        "inventory_to_forecast_ratio",
        ascending=False
    )
)


save_csv(
    sku_analysis,
    "phase8_sku_analysis.csv"
)


# ============================================================
# 8.9 REPLENISHMENT ANALYSIS
# ============================================================

section("CREATING REPLENISHMENT ANALYSIS")


replenishment_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "planning_daily_demand",

    "planning_reorder_point",
    "planning_target_stock",

    "planning_days_of_inventory",

    "inventory_gap_to_target",

    "suggested_reorder_quantity",

    "planning_reorder_status",

    "reorder_priority",

    "replenishment_decision",

    "business_action",

]


replenishment = (
    df[replenishment_columns]
    .copy()
)


replenishment = (
    replenishment
    .sort_values(

        [

            "suggested_reorder_quantity",

            "planning_days_of_inventory"

        ],

        ascending=[
            False,
            True
        ]

    )
)


save_csv(
    replenishment,
    "phase8_replenishment_analysis.csv"
)


# ============================================================
# 8.10 BUSINESS ACTIONS
# ============================================================

section("CREATING BUSINESS ACTION DATASET")


business_action_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "calibrated_forecast_30d",

    "planning_days_of_inventory",

    "inventory_coverage_category",

    "executive_inventory_flag",

    "planning_inventory_risk",

    "planning_stockout_risk",

    "suggested_reorder_quantity",

    "reorder_priority",

    "business_action",

    "replenishment_decision",

]


business_actions = (
    df[business_action_columns]
    .copy()
)


save_csv(
    business_actions,
    "phase8_business_actions.csv"
)


# ============================================================
# 8.11 INVENTORY COVERAGE DISTRIBUTION
# ============================================================

section("CREATING INVENTORY COVERAGE DISTRIBUTION")


coverage_distribution = (

    df

    .groupby(
        "inventory_coverage_category",
        as_index=False
    )

    .agg(

        store_sku_count=(
            "sku_id",
            "count"
        ),

        inventory_units=(
            "stock_on_hand",
            "sum"
        ),

    )

)


category_order = [

    "0-30 DAYS",
    "31-60 DAYS",
    "61-90 DAYS",
    "91-180 DAYS",
    "181-365 DAYS",
    ">365 DAYS",

]


coverage_distribution[
    "inventory_coverage_category"
] = pd.Categorical(

    coverage_distribution[
        "inventory_coverage_category"
    ],

    categories=category_order,

    ordered=True

)


coverage_distribution = (
    coverage_distribution
    .sort_values(
        "inventory_coverage_category"
    )
)


coverage_distribution[
    "inventory_percentage"
] = (

    coverage_distribution[
        "inventory_units"
    ]

    /
    total_inventory

    * 100

)


save_csv(
    coverage_distribution,
    "phase8_inventory_coverage_distribution.csv"
)


# ============================================================
# 8.12 BUSINESS ACTION DISTRIBUTION
# ============================================================

section("CREATING BUSINESS ACTION DISTRIBUTION")


action_distribution = (

    df

    .groupby(
        "business_action",
        dropna=False,
        as_index=False
    )

    .agg(

        store_sku_count=(
            "sku_id",
            "count"
        ),

        inventory_units=(
            "stock_on_hand",
            "sum"
        ),

        reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        ),

    )

)


action_distribution[
    "inventory_percentage"
] = (

    action_distribution[
        "inventory_units"
    ]

    /
    total_inventory

    * 100

)


action_distribution = (
    action_distribution
    .sort_values(
        "inventory_units",
        ascending=False
    )
)


save_csv(
    action_distribution,
    "phase8_business_action_distribution.csv"
)


# ============================================================
# 8.13 FORECAST HORIZON SUMMARY
# ============================================================

section("CREATING FORECAST HORIZON SUMMARY")


forecast_horizon_summary = pd.DataFrame({

    "horizon": [

        "30 DAYS",
        "60 DAYS",
        "90 DAYS"

    ],

    "forecast_units": [

        forecast_30d,
        forecast_60d,
        forecast_90d

    ],

    "planning_demand_units": [

        planning_30d,
        planning_60d,
        planning_90d

    ],

})


forecast_horizon_summary[
    "forecast_to_planning_ratio"
] = np.where(

    forecast_horizon_summary[
        "planning_demand_units"
    ] > 0,

    forecast_horizon_summary[
        "forecast_units"
    ]

    /

    forecast_horizon_summary[
        "planning_demand_units"
    ],

    np.nan

)


forecast_horizon_summary[
    "inventory_to_forecast_ratio"
] = np.where(

    forecast_horizon_summary[
        "forecast_units"
    ] > 0,

    total_inventory
    /
    forecast_horizon_summary[
        "forecast_units"
    ],

    np.nan

)


save_csv(
    forecast_horizon_summary,
    "phase8_forecast_horizon_summary.csv"
)


# ============================================================
# 8.14 DASHBOARD MASTER DATASET
# ============================================================

section("CREATING DASHBOARD MASTER DATASET")


dashboard_columns = [

    "store_id",
    "sku_id",

    "stock_on_hand",

    "units_30d",
    "units_90d",

    "avg_daily_demand_30d",
    "avg_daily_demand_90d",

    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",

    "planning_daily_demand",

    "planning_days_of_inventory",

    "calibrated_inventory_coverage_days",

    "planning_inventory_risk",

    "planning_stockout_risk",

    "planning_reorder_point",
    "planning_target_stock",

    "inventory_gap_to_target",

    "suggested_reorder_quantity",

    "planning_reorder_status",

    "reorder_priority",

    "demand_available",

    "inventory_coverage_category",

    "executive_inventory_flag",

    "replenishment_decision",

    "business_action",

]


dashboard_master = (
    df[dashboard_columns]
    .copy()
)


save_csv(
    dashboard_master,
    "phase8_dashboard_master.csv"
)


# ============================================================
# PHASE 8.1 INTERNAL CONSISTENCY CHECK
# ============================================================

section("PHASE 8.1 INTERNAL CONSISTENCY CHECK")


internal_checks = {

    "Store-SKU row count":
        len(df) == len(dashboard_master),

    "Demand available count":
        demand_available_count
        ==
        int(
            (
                dashboard_master[
                    "demand_available"
                ]
                == "TRUE"
            ).sum()
        ),

    "No-forecast count":
        no_forecast_count
        ==
        len(dormant),

    "No-forecast inventory":
        np.isclose(
            no_forecast_inventory,
            dormant[
                "stock_on_hand"
            ].sum()
        ),

    "Extreme overstock count":
        extreme_overstock_count
        ==
        len(extreme_overstock),

    "Total inventory":
        np.isclose(
            total_inventory,
            dashboard_master[
                "stock_on_hand"
            ].sum()
        ),

    "30-day forecast":
        np.isclose(
            forecast_30d,
            dashboard_master[
                "calibrated_forecast_30d"
            ].sum()
        ),

    "30-day planning demand":
        np.isclose(
            planning_30d,
            dashboard_master[
                "planning_daily_demand"
            ].sum() * 30
        ),

    "Reorder quantity":
        np.isclose(
            reorder_quantity,
            dashboard_master[
                "suggested_reorder_quantity"
            ].sum()
        ),

    "Reorder count":
        reorder_count
        ==
        int(
            (
                dashboard_master[
                    "suggested_reorder_quantity"
                ]
                > 0
            ).sum()
        ),

    "Executive summary row count":
        len(executive_summary) == 1,

}


for check, result in internal_checks.items():

    status = (
        "PASS"
        if result
        else "FAIL"
    )

    print(
        f"{check:<35}: "
        f"{status}"
    )


internal_status = (

    "PASS"

    if all(
        internal_checks.values()
    )

    else "REVIEW"

)


# ============================================================
# FINAL SUMMARY
# ============================================================

section("PHASE 8.1 SUMMARY")


print(
    f"Total Store-SKU:              {len(df):,}"
)

print(
    f"Total Stores:                 "
    f"{df['store_id'].nunique():,}"
)

print(
    f"Total SKUs:                   "
    f"{df['sku_id'].nunique():,}"
)

print(
    f"Total Inventory:              "
    f"{total_inventory:,.2f}"
)

print(
    f"30-day Forecast:              "
    f"{forecast_30d:,.2f}"
)

print(
    f"60-day Forecast:              "
    f"{forecast_60d:,.2f}"
)

print(
    f"90-day Forecast:              "
    f"{forecast_90d:,.2f}"
)

print(
    f"30-day Planning Demand:       "
    f"{planning_30d:,.2f}"
)

print(
    f"Inventory / 30-day Forecast:  "
    f"{inventory_forecast_ratio:.2f}x"
)

print(
    f"Inventory / Planning Demand:  "
    f"{inventory_planning_ratio:.2f}x"
)

print(
    f">30 DOI:                     "
    f"{over30_count:,}"
)

print(
    f">60 DOI:                     "
    f"{over60_count:,}"
)

print(
    f">90 DOI:                     "
    f"{over90_count:,}"
)

print(
    f">180 DOI:                    "
    f"{over180_count:,}"
)

print(
    f">365 DOI:                    "
    f"{extreme_overstock_count:,}"
)

print(
    f"No-Forecast Store-SKU:        "
    f"{no_forecast_count:,}"
)

print(
    f"No-Forecast Inventory:        "
    f"{no_forecast_inventory:,.2f}"
)

print(
    f"Suggested Reorder Quantity:   "
    f"{reorder_quantity:,.2f}"
)

print(
    f"Store-SKU Requiring Reorder:  "
    f"{reorder_count:,}"
)

print(
    f"Business Status:              "
    f"{executive_status}"
)


# ============================================================
# OUTPUT FILE LIST
# ============================================================

section("PHASE 8.1 OUTPUT FILES")


output_files = sorted(
    OUTPUT_PATH.glob("*.csv")
)


for file in output_files:

    print(file)


# ============================================================
# COMPLETION
# ============================================================

section("PHASE 8.1 COMPLETED")


print(
    "Final business analytics datasets successfully created."
)

print()

print(
    f"Phase 8.1 internal consistency checks: "
    f"{internal_status}"
)

print()

print(
    "Validated Phase 7.5 outputs were used as the source."
)

print()

print(
    "Corrected no-forecast / dormant inventory logic is active."
)

print()

print(
    "Executive summary is saved as a single-row dashboard KPI dataset."
)

print()

print(
    "The dashboard master dataset is ready for Phase 8.2."
)

print()

print(
    "NEXT:"
)

print(
    "PHASE 8.2 - DASHBOARD DATASET VALIDATION"
)

print(
    "Then:"
)

print(
    "PHASE 8.3 - FINAL DASHBOARD / STREAMLIT PREPARATION"
)

print("=" * 70)