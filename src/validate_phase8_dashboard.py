# ============================================================
# PROJECT FORESIGHT
# PHASE 8.2 - DASHBOARD DATASET VALIDATION
#
# Purpose:
#   Validate Phase 8.1 dashboard datasets before Streamlit.
#
# Source:
#   Phase 7.5 validated business recommendations
#
# Validates:
#   1. File existence
#   2. Required columns
#   3. Row counts
#   4. Duplicate Store-SKU
#   5. Numeric integrity
#   6. Inventory totals
#   7. Forecast totals
#   8. Planning demand
#   9. Days of inventory
#  10. Overstock counts
#  11. No-forecast counts
#  12. Reorder recommendations
#  13. Business actions
#  14. Dashboard master consistency
#  15. Phase 7.5 -> Phase 8.1 metric reconciliation
#
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

VALIDATION_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "validation"
)

PHASE8_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "phase8"
)

PHASE75_FILE = (
    VALIDATION_PATH
    / "validated_business_recommendations.csv"
)

DASHBOARD_MASTER_FILE = (
    PHASE8_PATH
    / "phase8_dashboard_master.csv"
)

EXECUTIVE_FILE = (
    PHASE8_PATH
    / "phase8_executive_summary.csv"
)

FORECAST_FILE = (
    PHASE8_PATH
    / "phase8_forecast_analysis.csv"
)

INVENTORY_FILE = (
    PHASE8_PATH
    / "phase8_inventory_risk.csv"
)

OVERSTOCK_FILE = (
    PHASE8_PATH
    / "phase8_overstock_analysis.csv"
)

EXTREME_OVERSTOCK_FILE = (
    PHASE8_PATH
    / "phase8_extreme_overstock.csv"
)

DORMANT_FILE = (
    PHASE8_PATH
    / "phase8_dormant_inventory.csv"
)

STORE_FILE = (
    PHASE8_PATH
    / "phase8_store_analysis.csv"
)

SKU_FILE = (
    PHASE8_PATH
    / "phase8_sku_analysis.csv"
)

REPLENISHMENT_FILE = (
    PHASE8_PATH
    / "phase8_replenishment_analysis.csv"
)

BUSINESS_ACTION_FILE = (
    PHASE8_PATH
    / "phase8_business_actions.csv"
)

COVERAGE_FILE = (
    PHASE8_PATH
    / "phase8_inventory_coverage_distribution.csv"
)

ACTION_DISTRIBUTION_FILE = (
    PHASE8_PATH
    / "phase8_business_action_distribution.csv"
)

HORIZON_FILE = (
    PHASE8_PATH
    / "phase8_forecast_horizon_summary.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = VALIDATION_PATH

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "phase8_dashboard_validation_summary.csv"
)

OUTPUT_CHECKS = (
    OUTPUT_DIR
    / "phase8_dashboard_validation_checks.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "phase8_dashboard_validation_report.txt"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path):

    if path.exists():
        print(f"{path}")
        print("FOUND")
        return True

    print(f"{path}")
    print("MISSING")
    return False


def safe_ratio(a, b):

    if b == 0 or pd.isna(b):
        return np.nan

    return a / b


def compare_value(actual, expected, tolerance=1e-6):

    if pd.isna(actual) or pd.isna(expected):
        return False

    return abs(float(actual) - float(expected)) <= tolerance


# ============================================================
# START
# ============================================================

section(
    "PROJECT FORESIGHT\n"
    "PHASE 8.2 - DASHBOARD DATASET VALIDATION"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK PHASE 7.5 INPUT
# ============================================================

section("CHECKING PHASE 7.5 VALIDATED INPUT")

if not check_file(PHASE75_FILE):

    print()
    print("ERROR: Phase 7.5 validated file is missing.")
    print("Phase 8.2 cannot continue.")

    sys.exit(1)


# ============================================================
# CHECK PHASE 8.1 FILES
# ============================================================

section("CHECKING PHASE 8.1 OUTPUT FILES")

phase8_files = {

    "dashboard_master": DASHBOARD_MASTER_FILE,

    "executive_summary": EXECUTIVE_FILE,

    "forecast_analysis": FORECAST_FILE,

    "inventory_risk": INVENTORY_FILE,

    "overstock_analysis": OVERSTOCK_FILE,

    "extreme_overstock": EXTREME_OVERSTOCK_FILE,

    "dormant_inventory": DORMANT_FILE,

    "store_analysis": STORE_FILE,

    "sku_analysis": SKU_FILE,

    "replenishment_analysis": REPLENISHMENT_FILE,

    "business_actions": BUSINESS_ACTION_FILE,

    "coverage_distribution": COVERAGE_FILE,

    "action_distribution": ACTION_DISTRIBUTION_FILE,

    "forecast_horizon": HORIZON_FILE,
}


missing_files = []

for name, path in phase8_files.items():

    print(f"{name:<30}: ", end="")

    if path.exists():

        print("FOUND")

    else:

        print("MISSING")
        missing_files.append(name)


if missing_files:

    print()
    print("ERROR: Missing Phase 8.1 files:")
    for item in missing_files:
        print(f" - {item}")

    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING PHASE 7.5 AND PHASE 8.1 DATASETS")

source = pd.read_csv(PHASE75_FILE)

dashboard = pd.read_csv(DASHBOARD_MASTER_FILE)

executive = pd.read_csv(EXECUTIVE_FILE)

forecast = pd.read_csv(FORECAST_FILE)

inventory = pd.read_csv(INVENTORY_FILE)

overstock = pd.read_csv(OVERSTOCK_FILE)

extreme_overstock = pd.read_csv(EXTREME_OVERSTOCK_FILE)

dormant = pd.read_csv(DORMANT_FILE)

store = pd.read_csv(STORE_FILE)

sku = pd.read_csv(SKU_FILE)

replenishment = pd.read_csv(REPLENISHMENT_FILE)

business_actions = pd.read_csv(BUSINESS_ACTION_FILE)

coverage = pd.read_csv(COVERAGE_FILE)

action_distribution = pd.read_csv(ACTION_DISTRIBUTION_FILE)

horizon = pd.read_csv(HORIZON_FILE)


print()
print(f"Phase 7.5 rows:          {len(source):,}")
print(f"Dashboard master rows:   {len(dashboard):,}")
print(f"Executive rows:          {len(executive):,}")
print(f"Forecast rows:           {len(forecast):,}")
print(f"Inventory rows:          {len(inventory):,}")
print(f"Store rows:              {len(store):,}")
print(f"SKU rows:                {len(sku):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

section("CHECKING REQUIRED DASHBOARD COLUMNS")

required_source = [

    "store_id",
    "sku_id",
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "demand_available",
    "suggested_reorder_quantity",
    "business_action",
]


required_dashboard = [

    "store_id",
    "sku_id",
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "demand_available",
    "suggested_reorder_quantity",
    "business_action",
]


missing_source = [
    c for c in required_source
    if c not in source.columns
]

missing_dashboard = [
    c for c in required_dashboard
    if c not in dashboard.columns
]


print(
    f"Missing Phase 7.5 columns: "
    f"{len(missing_source)}"
)

print(
    f"Missing dashboard columns: "
    f"{len(missing_dashboard)}"
)


if missing_source:

    print("Missing source columns:")
    for c in missing_source:
        print(f" - {c}")


if missing_dashboard:

    print("Missing dashboard columns:")
    for c in missing_dashboard:
        print(f" - {c}")


# ============================================================
# NUMERIC PREPARATION
# ============================================================

section("PREPARING NUMERIC COLUMNS")

numeric_columns = [

    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "suggested_reorder_quantity",
]


for column in numeric_columns:

    if column in source.columns:

        source[column] = pd.to_numeric(
            source[column],
            errors="coerce"
        )

    if column in dashboard.columns:

        dashboard[column] = pd.to_numeric(
            dashboard[column],
            errors="coerce"
        )


print("Numeric preparation completed.")


# ============================================================
# PHASE 7.5 REFERENCE METRICS
# ============================================================

section("CALCULATING PHASE 7.5 REFERENCE METRICS")

source_total_rows = len(source)

source_total_inventory = (
    source["stock_on_hand"].sum()
)

source_forecast_30 = (
    source["calibrated_forecast_30d"].sum()
)

source_forecast_60 = (
    source["calibrated_forecast_60d"].sum()
)

source_forecast_90 = (
    source["calibrated_forecast_90d"].sum()
)

source_planning_daily = (
    source["planning_daily_demand"].sum()
)

source_planning_30 = (
    source_planning_daily * 30
)

source_planning_60 = (
    source_planning_daily * 60
)

source_planning_90 = (
    source_planning_daily * 90
)

source_no_forecast = (
    source["demand_available"]
    .astype(str)
    .str.upper()
    .isin(
        [
            "FALSE",
            "0",
            "NO",
            "N",
            "NO_FORECAST",
            "NO_FORECAST_DEMAND",
        ]
    )
    .sum()
)

source_no_forecast_inventory = (
    source.loc[
        source["demand_available"]
        .astype(str)
        .str.upper()
        .isin(
            [
                "FALSE",
                "0",
                "NO",
                "N",
                "NO_FORECAST",
                "NO_FORECAST_DEMAND",
            ]
        ),
        "stock_on_hand",
    ].sum()
)

source_over365 = (
    source["planning_days_of_inventory"] > 365
).sum()

source_over30 = (
    source["planning_days_of_inventory"] > 30
).sum()

source_over60 = (
    source["planning_days_of_inventory"] > 60
).sum()

source_over90 = (
    source["planning_days_of_inventory"] > 90
).sum()

source_over180 = (
    source["planning_days_of_inventory"] > 180
).sum()

source_reorder_qty = (
    source["suggested_reorder_quantity"].sum()
)

source_reorder_count = (
    source["suggested_reorder_quantity"] > 0
).sum()


print(
    f"Total Store-SKU:              "
    f"{source_total_rows:,}"
)

print(
    f"Total inventory:              "
    f"{source_total_inventory:,.2f}"
)

print(
    f"30-day calibrated forecast:   "
    f"{source_forecast_30:,.2f}"
)

print(
    f"60-day calibrated forecast:   "
    f"{source_forecast_60:,.2f}"
)

print(
    f"90-day calibrated forecast:   "
    f"{source_forecast_90:,.2f}"
)

print(
    f"Planning daily demand:        "
    f"{source_planning_daily:,.2f}"
)

print(
    f"Planning 30-day demand:       "
    f"{source_planning_30:,.2f}"
)

print(
    f"No-forecast Store-SKU:        "
    f"{source_no_forecast:,}"
)

print(
    f"No-forecast inventory:        "
    f"{source_no_forecast_inventory:,.2f}"
)

print(
    f">30 DOI:                      "
    f"{source_over30:,}"
)

print(
    f">60 DOI:                      "
    f"{source_over60:,}"
)

print(
    f">90 DOI:                      "
    f"{source_over90:,}"
)

print(
    f">180 DOI:                     "
    f"{source_over180:,}"
)

print(
    f">365 DOI:                     "
    f"{source_over365:,}"
)

print(
    f"Suggested reorder quantity:   "
    f"{source_reorder_qty:,.2f}"
)

print(
    f"Store-SKU requiring reorder:  "
    f"{source_reorder_count:,}"
)


# ============================================================
# DASHBOARD MASTER METRICS
# ============================================================

section("CALCULATING DASHBOARD MASTER METRICS")

dashboard_total_rows = len(dashboard)

dashboard_total_inventory = (
    dashboard["stock_on_hand"].sum()
)

dashboard_forecast_30 = (
    dashboard["calibrated_forecast_30d"].sum()
)

dashboard_forecast_60 = (
    dashboard["calibrated_forecast_60d"].sum()
)

dashboard_forecast_90 = (
    dashboard["calibrated_forecast_90d"].sum()
)

dashboard_planning_daily = (
    dashboard["planning_daily_demand"].sum()
)

dashboard_planning_30 = (
    dashboard_planning_daily * 30
)

dashboard_no_forecast_mask = (
    dashboard["demand_available"]
    .astype(str)
    .str.upper()
    .isin(
        [
            "FALSE",
            "0",
            "NO",
            "N",
            "NO_FORECAST",
            "NO_FORECAST_DEMAND",
        ]
    )
)

dashboard_no_forecast = (
    dashboard_no_forecast_mask.sum()
)

dashboard_no_forecast_inventory = (
    dashboard.loc[
        dashboard_no_forecast_mask,
        "stock_on_hand"
    ].sum()
)

dashboard_over30 = (
    dashboard["planning_days_of_inventory"] > 30
).sum()

dashboard_over60 = (
    dashboard["planning_days_of_inventory"] > 60
).sum()

dashboard_over90 = (
    dashboard["planning_days_of_inventory"] > 90
).sum()

dashboard_over180 = (
    dashboard["planning_days_of_inventory"] > 180
).sum()

dashboard_over365 = (
    dashboard["planning_days_of_inventory"] > 365
).sum()

dashboard_reorder_qty = (
    dashboard["suggested_reorder_quantity"].sum()
)

dashboard_reorder_count = (
    dashboard["suggested_reorder_quantity"] > 0
).sum()


print(
    f"Dashboard Store-SKU:          "
    f"{dashboard_total_rows:,}"
)

print(
    f"Dashboard inventory:          "
    f"{dashboard_total_inventory:,.2f}"
)

print(
    f"Dashboard 30-day forecast:    "
    f"{dashboard_forecast_30:,.2f}"
)

print(
    f"Dashboard planning demand:    "
    f"{dashboard_planning_30:,.2f}"
)

print(
    f"Dashboard no-forecast:        "
    f"{dashboard_no_forecast:,}"
)

print(
    f"Dashboard no-forecast stock:  "
    f"{dashboard_no_forecast_inventory:,.2f}"
)


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

section("VALIDATING DASHBOARD STORE-SKU UNIQUENESS")

source_duplicates = (
    source.duplicated(
        subset=["store_id", "sku_id"]
    ).sum()
)

dashboard_duplicates = (
    dashboard.duplicated(
        subset=["store_id", "sku_id"]
    ).sum()
)

print(
    f"Phase 7.5 duplicate Store-SKU: "
    f"{source_duplicates:,}"
)

print(
    f"Dashboard duplicate Store-SKU:  "
    f"{dashboard_duplicates:,}"
)


# ============================================================
# METRIC RECONCILIATION
# ============================================================

section("RECONCILING PHASE 7.5 -> PHASE 8.1")

checks = []


def add_check(
    name,
    actual,
    expected,
    passed,
    tolerance=None,
):

    checks.append(
        {
            "check": name,
            "actual": actual,
            "expected": expected,
            "difference": (
                actual - expected
                if isinstance(actual, (int, float, np.number))
                and isinstance(expected, (int, float, np.number))
                else np.nan
            ),
            "status": "PASS" if passed else "FAIL",
            "tolerance": tolerance,
        }
    )


# ------------------------------------------------------------
# ROW COUNT
# ------------------------------------------------------------

add_check(
    "row_count",
    dashboard_total_rows,
    source_total_rows,
    dashboard_total_rows == source_total_rows,
)


# ------------------------------------------------------------
# INVENTORY
# ------------------------------------------------------------

inventory_match = compare_value(
    dashboard_total_inventory,
    source_total_inventory,
    tolerance=0.01,
)

add_check(
    "total_inventory",
    dashboard_total_inventory,
    source_total_inventory,
    inventory_match,
    0.01,
)


# ------------------------------------------------------------
# FORECASTS
# ------------------------------------------------------------

forecast30_match = compare_value(
    dashboard_forecast_30,
    source_forecast_30,
    tolerance=0.01,
)

forecast60_match = compare_value(
    dashboard_forecast_60,
    source_forecast_60,
    tolerance=0.01,
)

forecast90_match = compare_value(
    dashboard_forecast_90,
    source_forecast_90,
    tolerance=0.01,
)


add_check(
    "forecast_30d",
    dashboard_forecast_30,
    source_forecast_30,
    forecast30_match,
    0.01,
)

add_check(
    "forecast_60d",
    dashboard_forecast_60,
    source_forecast_60,
    forecast60_match,
    0.01,
)

add_check(
    "forecast_90d",
    dashboard_forecast_90,
    source_forecast_90,
    forecast90_match,
    0.01,
)


# ------------------------------------------------------------
# PLANNING DEMAND
# ------------------------------------------------------------

planning_match = compare_value(
    dashboard_planning_30,
    source_planning_30,
    tolerance=0.01,
)

add_check(
    "planning_30d_demand",
    dashboard_planning_30,
    source_planning_30,
    planning_match,
    0.01,
)


# ------------------------------------------------------------
# NO FORECAST
# ------------------------------------------------------------

no_forecast_match = (
    dashboard_no_forecast ==
    source_no_forecast
)

no_forecast_inventory_match = compare_value(
    dashboard_no_forecast_inventory,
    source_no_forecast_inventory,
    tolerance=0.01,
)


add_check(
    "no_forecast_store_sku",
    dashboard_no_forecast,
    source_no_forecast,
    no_forecast_match,
)


add_check(
    "no_forecast_inventory",
    dashboard_no_forecast_inventory,
    source_no_forecast_inventory,
    no_forecast_inventory_match,
    0.01,
)


# ------------------------------------------------------------
# DOI COUNTS
# ------------------------------------------------------------

add_check(
    "over30_doi",
    dashboard_over30,
    source_over30,
    dashboard_over30 == source_over30,
)

add_check(
    "over60_doi",
    dashboard_over60,
    source_over60,
    dashboard_over60 == source_over60,
)

add_check(
    "over90_doi",
    dashboard_over90,
    source_over90,
    dashboard_over90 == source_over90,
)

add_check(
    "over180_doi",
    dashboard_over180,
    source_over180,
    dashboard_over180 == source_over180,
)

add_check(
    "over365_doi",
    dashboard_over365,
    source_over365,
    dashboard_over365 == source_over365,
)


# ------------------------------------------------------------
# REORDER
# ------------------------------------------------------------

reorder_qty_match = compare_value(
    dashboard_reorder_qty,
    source_reorder_qty,
    tolerance=0.01,
)

add_check(
    "reorder_quantity",
    dashboard_reorder_qty,
    source_reorder_qty,
    reorder_qty_match,
    0.01,
)

add_check(
    "reorder_count",
    dashboard_reorder_count,
    source_reorder_count,
    dashboard_reorder_count == source_reorder_count,
)


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

add_check(
    "dashboard_duplicates",
    dashboard_duplicates,
    0,
    dashboard_duplicates == 0,
)


# ============================================================
# PRINT RECONCILIATION
# ============================================================

section("PHASE 8.1 DASHBOARD RECONCILIATION RESULTS")

for check in checks:

    print(
        f"{check['check']:<30}: "
        f"{check['status']}"
    )


# ============================================================
# BUSINESS STATUS VALIDATION
# ============================================================

section("VALIDATING BUSINESS STATUS")

inventory_forecast_ratio = safe_ratio(
    dashboard_total_inventory,
    dashboard_forecast_30,
)

inventory_planning_ratio = safe_ratio(
    dashboard_total_inventory,
    dashboard_planning_30,
)


if inventory_forecast_ratio > 100:

    expected_business_status = "CRITICAL_OVERSTOCK"

elif inventory_forecast_ratio > 50:

    expected_business_status = "SEVERE_OVERSTOCK"

elif inventory_forecast_ratio > 20:

    expected_business_status = "OVERSTOCK"

else:

    expected_business_status = "NORMAL"


print(
    f"Inventory / 30-day forecast: "
    f"{inventory_forecast_ratio:.2f}x"
)

print(
    f"Inventory / planning 30d:    "
    f"{inventory_planning_ratio:.2f}x"
)

print(
    f"Expected business status:    "
    f"{expected_business_status}"
)


# ============================================================
# BUSINESS ACTION VALIDATION
# ============================================================

section("VALIDATING BUSINESS ACTIONS")

business_action_column = "business_action"

if business_action_column in dashboard.columns:

    action_counts = (
        dashboard[business_action_column]
        .fillna("UNKNOWN")
        .value_counts()
    )

    print()
    print("Business Action Distribution:")
    print(action_counts.to_string())

    severe_mask = (
        dashboard["planning_days_of_inventory"]
        > 365
    )

    severe_actions = (
        dashboard.loc[
            severe_mask,
            business_action_column
        ]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    valid_overstock_keywords = [

        "OVERSTOCK",
        "TRANSFER",
        "MARKDOWN",
        "LIQUIDATION",
        "PAUSE",
        "HOLD",
        "REDUCE",
        "REVIEW",
        "EXCESS",
    ]

    severe_action_mismatch = 0

    for action in severe_actions:

        if not any(
            keyword in action
            for keyword in valid_overstock_keywords
        ):

            severe_action_mismatch += 1

    print()
    print(
        f"Extreme overstock action mismatches: "
        f"{severe_action_mismatch:,}"
    )

else:

    severe_action_mismatch = -1

    print(
        "business_action column not found."
    )


# ============================================================
# DATA QUALITY VALIDATION
# ============================================================

section("VALIDATING DATA QUALITY")

negative_stock = (
    (dashboard["stock_on_hand"] < 0)
    .sum()
)

negative_forecast = (
    (
        dashboard[
            [
                "calibrated_forecast_30d",
                "calibrated_forecast_60d",
                "calibrated_forecast_90d",
            ]
        ]
        < 0
    )
    .any(axis=1)
    .sum()
)

negative_reorder = (
    (
        dashboard["suggested_reorder_quantity"]
        < 0
    )
    .sum()
)

missing_planning = (
    dashboard["planning_daily_demand"]
    .isna()
    .sum()
)

missing_inventory = (
    dashboard["stock_on_hand"]
    .isna()
    .sum()
)

missing_forecast = (
    dashboard["calibrated_forecast_30d"]
    .isna()
    .sum()
)


print(
    f"Negative stock:              "
    f"{negative_stock:,}"
)

print(
    f"Negative forecast rows:      "
    f"{negative_forecast:,}"
)

print(
    f"Negative reorder quantity:    "
    f"{negative_reorder:,}"
)

print(
    f"Missing planning demand:      "
    f"{missing_planning:,}"
)

print(
    f"Missing inventory:             "
    f"{missing_inventory:,}"
)

print(
    f"Missing 30d forecast:           "
    f"{missing_forecast:,}"
)


add_check(
    "negative_stock",
    negative_stock,
    0,
    negative_stock == 0,
)

add_check(
    "negative_forecast",
    negative_forecast,
    0,
    negative_forecast == 0,
)

add_check(
    "negative_reorder",
    negative_reorder,
    0,
    negative_reorder == 0,
)

add_check(
    "missing_planning_demand",
    missing_planning,
    0,
    missing_planning == 0,
)

add_check(
    "missing_inventory",
    missing_inventory,
    0,
    missing_inventory == 0,
)

add_check(
    "missing_forecast",
    missing_forecast,
    0,
    missing_forecast == 0,
)


# ============================================================
# STORE / SKU VALIDATION
# ============================================================

section("VALIDATING STORE / SKU DIMENSIONS")

source_store_count = (
    source["store_id"].nunique()
)

source_sku_count = (
    source["sku_id"].nunique()
)

dashboard_store_count = (
    dashboard["store_id"].nunique()
)

dashboard_sku_count = (
    dashboard["sku_id"].nunique()
)


print(
    f"Phase 7.5 stores:       "
    f"{source_store_count:,}"
)

print(
    f"Dashboard stores:       "
    f"{dashboard_store_count:,}"
)

print(
    f"Phase 7.5 SKUs:         "
    f"{source_sku_count:,}"
)

print(
    f"Dashboard SKUs:         "
    f"{dashboard_sku_count:,}"
)


add_check(
    "store_count",
    dashboard_store_count,
    source_store_count,
    dashboard_store_count == source_store_count,
)

add_check(
    "sku_count",
    dashboard_sku_count,
    source_sku_count,
    dashboard_sku_count == source_sku_count,
)


# ============================================================
# STORE-LEVEL FILE
# ============================================================

section("VALIDATING STORE-LEVEL DATASET")

print(
    f"Store analysis rows: "
    f"{len(store):,}"
)

print(
    f"Expected stores:      "
    f"{source_store_count:,}"
)

store_count_pass = (
    len(store) == source_store_count
)

add_check(
    "store_analysis_rows",
    len(store),
    source_store_count,
    store_count_pass,
)


# ============================================================
# SKU-LEVEL FILE
# ============================================================

section("VALIDATING SKU-LEVEL DATASET")

print(
    f"SKU analysis rows: "
    f"{len(sku):,}"
)

print(
    f"Expected SKUs:      "
    f"{source_sku_count:,}"
)

sku_count_pass = (
    len(sku) == source_sku_count
)

add_check(
    "sku_analysis_rows",
    len(sku),
    source_sku_count,
    sku_count_pass,
)


# ============================================================
# EXTREME OVERSTOCK DATASET
# ============================================================

section("VALIDATING EXTREME OVERSTOCK DATASET")

print(
    f"Extreme overstock rows: "
    f"{len(extreme_overstock):,}"
)

print(
    f"Expected >365 DOI:     "
    f"{source_over365:,}"
)

extreme_count_pass = (
    len(extreme_overstock) == source_over365
)

add_check(
    "extreme_overstock_rows",
    len(extreme_overstock),
    source_over365,
    extreme_count_pass,
)


# ============================================================
# DORMANT DATASET
# ============================================================

section("VALIDATING DORMANT INVENTORY DATASET")

print(
    f"Dormant rows: "
    f"{len(dormant):,}"
)

print(
    f"Expected no-forecast: "
    f"{source_no_forecast:,}"
)

dormant_count_pass = (
    len(dormant) == source_no_forecast
)

add_check(
    "dormant_inventory_rows",
    len(dormant),
    source_no_forecast,
    dormant_count_pass,
)


# ============================================================
# REPLENISHMENT DATASET
# ============================================================

section("VALIDATING REPLENISHMENT DATASET")

print(
    f"Replenishment rows: "
    f"{len(replenishment):,}"
)

print(
    f"Expected Store-SKU: "
    f"{source_total_rows:,}"
)

replenishment_count_pass = (
    len(replenishment) == source_total_rows
)

add_check(
    "replenishment_rows",
    len(replenishment),
    source_total_rows,
    replenishment_count_pass,
)


# ============================================================
# EXECUTIVE DATASET
# ============================================================

section("VALIDATING EXECUTIVE SUMMARY DATASET")

print(
    f"Executive summary rows: "
    f"{len(executive):,}"
)

if len(executive) == 1:

    print("Executive summary structure: PASS")

    executive_structure_pass = True

else:

    print("Executive summary structure: REVIEW")

    executive_structure_pass = False


add_check(
    "executive_summary_structure",
    len(executive),
    1,
    executive_structure_pass,
)


# ============================================================
# FORECAST HORIZON DATASET
# ============================================================

section("VALIDATING FORECAST HORIZON DATASET")

print(
    f"Forecast horizon rows: "
    f"{len(horizon):,}"
)

horizon_structure_pass = (
    len(horizon) >= 3
)

print(
    "Forecast horizon structure: "
    + (
        "PASS"
        if horizon_structure_pass
        else "REVIEW"
    )
)

add_check(
    "forecast_horizon_structure",
    len(horizon),
    3,
    horizon_structure_pass,
)


# ============================================================
# CRITICAL NO-FORECAST RECONCILIATION
# ============================================================

section(
    "CRITICAL CHECK - NO-FORECAST METRIC RECONCILIATION"
)

print()
print(
    "Phase 7.5 expected:"
)

print(
    f"  No-forecast Store-SKU = "
    f"{source_no_forecast:,}"
)

print(
    f"  No-forecast inventory = "
    f"{source_no_forecast_inventory:,.2f}"
)

print()
print(
    "Phase 8.1 dashboard master:"
)

print(
    f"  No-forecast Store-SKU = "
    f"{dashboard_no_forecast:,}"
)

print(
    f"  No-forecast inventory = "
    f"{dashboard_no_forecast_inventory:,.2f}"
)


if no_forecast_match and no_forecast_inventory_match:

    print()
    print(
        "NO-FORECAST RECONCILIATION: PASS"
    )

else:

    print()
    print(
        "NO-FORECAST RECONCILIATION: FAIL"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Phase 8.1 appears to have incorrectly "
        "mapped the demand_available field."
    )

    print(
        "Do NOT proceed to Streamlit until "
        "Phase 8.1 is corrected."
    )


# ============================================================
# OVERALL CHECK STATUS
# ============================================================

section("PHASE 8.2 VALIDATION CHECKS")

checks_df = pd.DataFrame(checks)

for _, row in checks_df.iterrows():

    print(
        f"{row['check']:<35}: "
        f"{row['status']}"
    )


# ============================================================
# OVERALL DATA QUALITY
# ============================================================

all_pass = (
    checks_df["status"] == "PASS"
).all()

if severe_action_mismatch == 0:

    action_check_pass = True

else:

    action_check_pass = False


if all_pass and action_check_pass:

    validation_status = "PASS"

else:

    validation_status = "REVIEW"


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

section("PHASE 8.2 EXECUTIVE VALIDATION SUMMARY")

print(
    f"Validation status:            "
    f"{validation_status}"
)

print(
    f"Total Store-SKU:              "
    f"{dashboard_total_rows:,}"
)

print(
    f"Total stores:                 "
    f"{dashboard_store_count:,}"
)

print(
    f"Total SKUs:                   "
    f"{dashboard_sku_count:,}"
)

print(
    f"Total inventory:              "
    f"{dashboard_total_inventory:,.2f}"
)

print(
    f"30-day forecast:              "
    f"{dashboard_forecast_30:,.2f}"
)

print(
    f"60-day forecast:              "
    f"{dashboard_forecast_60:,.2f}"
)

print(
    f"90-day forecast:              "
    f"{dashboard_forecast_90:,.2f}"
)

print(
    f"Planning 30-day demand:       "
    f"{dashboard_planning_30:,.2f}"
)

print(
    f"Inventory / 30d forecast:     "
    f"{inventory_forecast_ratio:.2f}x"
)

print(
    f"Inventory / planning 30d:     "
    f"{inventory_planning_ratio:.2f}x"
)

print(
    f">365 DOI:                     "
    f"{dashboard_over365:,}"
)

print(
    f"No-forecast Store-SKU:        "
    f"{dashboard_no_forecast:,}"
)

print(
    f"No-forecast inventory:        "
    f"{dashboard_no_forecast_inventory:,.2f}"
)

print(
    f"Suggested reorder quantity:   "
    f"{dashboard_reorder_qty:,.2f}"
)

print(
    f"Store-SKU requiring reorder:  "
    f"{dashboard_reorder_count:,}"
)

print(
    f"Business status:              "
    f"{expected_business_status}"
)


# ============================================================
# SAVE CHECKS
# ============================================================

section("SAVING PHASE 8.2 VALIDATION FILES")

checks_df.to_csv(
    OUTPUT_CHECKS,
    index=False
)

summary = pd.DataFrame(
    [
        {
            "validation_status": validation_status,
            "total_store_sku": dashboard_total_rows,
            "total_stores": dashboard_store_count,
            "total_skus": dashboard_sku_count,
            "total_inventory": dashboard_total_inventory,
            "forecast_30d": dashboard_forecast_30,
            "forecast_60d": dashboard_forecast_60,
            "forecast_90d": dashboard_forecast_90,
            "planning_30d": dashboard_planning_30,
            "inventory_to_forecast_30d": inventory_forecast_ratio,
            "inventory_to_planning_30d": inventory_planning_ratio,
            "over30_doi": dashboard_over30,
            "over60_doi": dashboard_over60,
            "over90_doi": dashboard_over90,
            "over180_doi": dashboard_over180,
            "over365_doi": dashboard_over365,
            "no_forecast_store_sku": dashboard_no_forecast,
            "no_forecast_inventory": dashboard_no_forecast_inventory,
            "reorder_quantity": dashboard_reorder_qty,
            "reorder_store_sku": dashboard_reorder_count,
            "expected_business_status": expected_business_status,
            "severe_action_mismatches": severe_action_mismatch,
        }
    ]
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# CREATE EXECUTIVE REPORT
# ============================================================

section("CREATING PHASE 8.2 EXECUTIVE REPORT")

report_lines = [

    "PROJECT FORESIGHT",
    "PHASE 8.2 - DASHBOARD DATASET VALIDATION",
    "",
    "=" * 70,
    "",
    f"Validation Status: {validation_status}",
    "",
    f"Total Store-SKU: {dashboard_total_rows:,}",
    f"Total Stores: {dashboard_store_count:,}",
    f"Total SKUs: {dashboard_sku_count:,}",
    f"Total Inventory: {dashboard_total_inventory:,.2f}",
    "",
    f"30-day Forecast: {dashboard_forecast_30:,.2f}",
    f"60-day Forecast: {dashboard_forecast_60:,.2f}",
    f"90-day Forecast: {dashboard_forecast_90:,.2f}",
    "",
    f"Planning 30-day Demand: {dashboard_planning_30:,.2f}",
    "",
    f"Inventory / 30-day Forecast: "
    f"{inventory_forecast_ratio:.2f}x",
    "",
    f"Inventory / Planning 30-day: "
    f"{inventory_planning_ratio:.2f}x",
    "",
    f">30 DOI: {dashboard_over30:,}",
    f">60 DOI: {dashboard_over60:,}",
    f">90 DOI: {dashboard_over90:,}",
    f">180 DOI: {dashboard_over180:,}",
    f">365 DOI: {dashboard_over365:,}",
    "",
    f"No-forecast Store-SKU: "
    f"{dashboard_no_forecast:,}",
    "",
    f"No-forecast Inventory: "
    f"{dashboard_no_forecast_inventory:,.2f}",
    "",
    f"Suggested Reorder Quantity: "
    f"{dashboard_reorder_qty:,.2f}",
    "",
    f"Store-SKU Requiring Reorder: "
    f"{dashboard_reorder_count:,}",
    "",
    f"Expected Business Status: "
    f"{expected_business_status}",
    "",
    "=" * 70,
    "",
    "PHASE 7.5 -> PHASE 8.1 RECONCILIATION",
    "",
]


for _, row in checks_df.iterrows():

    report_lines.append(
        f"{row['check']}: {row['status']}"
    )


report_lines.extend(
    [
        "",
        "=" * 70,
        "",
        "BUSINESS INTERPRETATION",
        "",
    ]
)


if validation_status == "PASS":

    report_lines.extend(
        [
            "Dashboard datasets successfully "
            "reconcile with Phase 7.5.",
            "",
            "The dashboard may proceed to "
            "Phase 8.3.",
            "",
            "Critical business condition:",
            "Inventory remains substantially "
            "higher than near-term demand.",
            "",
            f"Inventory is "
            f"{inventory_forecast_ratio:.2f}x "
            "the calibrated 30-day forecast.",
            "",
            "Recommended dashboard focus:",
            "1. Executive inventory overview",
            "2. Forecast horizon analysis",
            "3. Inventory coverage",
            "4. Extreme overstock",
            "5. Dormant inventory",
            "6. Store analysis",
            "7. SKU analysis",
            "8. Replenishment control",
            "9. Business actions",
        ]
    )

else:

    report_lines.extend(
        [
            "Dashboard datasets require review "
            "before Phase 8.3.",
            "",
            "Do NOT finalize the Streamlit "
            "dashboard until failed validation "
            "checks are corrected.",
        ]
    )


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


print(
    f"Saved: {OUTPUT_SUMMARY}"
)

print(
    f"Saved: {OUTPUT_CHECKS}"
)

print(
    f"Saved: {OUTPUT_REPORT}"
)


# ============================================================
# FINAL DECISION
# ============================================================

section("PHASE 8.2 FINAL DECISION")

if validation_status == "PASS":

    print(
        "PHASE 8.2 VALIDATION: PASS"
    )

    print()
    print(
        "Phase 8.1 dashboard datasets "
        "reconcile with Phase 7.5."
    )

    print()
    print(
        "READY FOR:"
    )

    print(
        "PHASE 8.3 - FINAL STREAMLIT "
        "DASHBOARD PREPARATION"
    )

else:

    print(
        "PHASE 8.2 VALIDATION: REVIEW"
    )

    print()
    print(
        "Phase 8.1 contains one or more "
        "dataset reconciliation issues."
    )

    print()
    print(
        "DO NOT proceed to final dashboard "
        "until the failed checks are corrected."
    )


print()
print("=" * 70)