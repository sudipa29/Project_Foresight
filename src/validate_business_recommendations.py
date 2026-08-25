# ============================================================
# PROJECT FORESIGHT
# PHASE 7.5 - BUSINESS RECOMMENDATION VALIDATION
#
# Purpose:
#   Validate the corrected inventory risk and business
#   recommendation dataset before proceeding to Phase 8.
#
# IMPORTANT VALIDATION PRINCIPLES
# -------------------------------
# 1. Planning demand is the canonical demand basis for
#    inventory/reorder validation.
#
# 2. planning_days_of_inventory is validated against:
#       stock_on_hand / planning_daily_demand
#
# 3. Overstock is NOT treated as a reorder requirement.
#    Zero reorder quantity is correct for overstock.
#
# 4. No-demand / no-forecast rows are handled separately.
#
# 5. Business actions are validated semantically rather than
#    requiring one exact action string.
#
# 6. The script does NOT modify the source inventory file.
#
# Output:
#   data\processed\forecasting\business_insights\validation\
#
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import sys


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

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

STORE_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_store_validation.csv"
)

SKU_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_sku_validation.csv"
)

EXTREME_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_extreme_overstock.csv"
)

DORMANT_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_dormant_inventory.csv"
)

REORDER_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_reorder_validation.csv"
)

VALIDATED_OUTPUT = (
    OUTPUT_DIR
    / "validated_business_recommendations.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_validation_summary.csv"
)

QUALITY_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_quality_checks.csv"
)

REPORT_OUTPUT = (
    OUTPUT_DIR
    / "business_recommendation_validation_report.txt"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

WIDTH = 70


def print_header(title):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_pct(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def fmt_ratio(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}x"


def safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = pd.Series(np.nan, index=numerator.index)

    valid = denominator > 0

    result.loc[valid] = (
        numerator.loc[valid] / denominator.loc[valid]
    )

    return result


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "planning_target_stock",
    "suggested_reorder_quantity",
    "planning_reorder_status",
    "business_action",
    "demand_available",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_inventory_risk",
    "planning_stockout_risk",
]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "units_30d",
    "units_90d",
    "avg_daily_demand_30d",
    "avg_daily_demand_90d",
    "forecast_30d_units",
    "forecast_daily_demand",
    "planning_daily_demand",
    "days_of_inventory",
    "forecast_coverage_ratio",
    "target_stock",
    "suggested_reorder_qty",
    "historical_30d_forecast_30d_ratio",
    "forecast_vs_90d_ratio",
    "stock_to_30d_demand",
    "stock_to_90d_demand",
    "stock_to_forecast",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "calibrated_avg_daily_forecast_30d",
    "calibrated_avg_daily_forecast_60d",
    "calibrated_avg_daily_forecast_90d",
    "calibrated_inventory_coverage_days",
    "stock_after_calibrated_30d",
    "stock_after_calibrated_60d",
    "stock_after_calibrated_90d",
    "stock_to_calibrated_30d_ratio",
    "calibrated_replenishment_gap_30d",
    "calibrated_replenishment_gap_60d",
    "calibrated_replenishment_gap_90d",
    "calibrated_stockout_risk_30d",
    "calibrated_vs_existing_forecast_ratio",
    "calibrated_vs_existing_forecast_difference",
    "historical_daily_demand_30d",
    "historical_daily_demand_90d",
    "calibrated_daily_forecast_30d",
    "calibrated_daily_forecast_60d",
    "calibrated_daily_forecast_90d",
    "planning_safety_stock",
    "planning_reorder_point",
    "planning_target_stock",
    "planning_days_of_inventory",
    "calibrated_forecast_coverage_days",
    "stock_after_30d_planning_demand",
    "stock_after_60d_planning_demand",
    "stock_after_90d_planning_demand",
    "inventory_gap_to_target",
    "suggested_reorder_quantity",
    "planning_stockout_risk",
    "forecast_vs_planning_ratio",
    "stock_to_planning_30d_ratio",
]


# ============================================================
# MAIN
# ============================================================

print_header(
    "PROJECT FORESIGHT\n"
    "PHASE 7.5 - BUSINESS RECOMMENDATION VALIDATION"
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

print_header("CHECKING INPUT FILE")

print(INPUT_FILE)

if not INPUT_FILE.exists():
    print("ERROR: Input file NOT FOUND")
    print()
    print("Expected:")
    print(INPUT_FILE)
    sys.exit(1)

print("FOUND")


# ============================================================
# LOAD DATA
# ============================================================

print_header("LOADING INVENTORY RISK DATA")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print()
print("Columns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print_header("CHECKING REQUIRED COLUMNS")

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("ERROR: Missing required columns:")
    for col in missing_columns:
        print(f"  - {col}")

    sys.exit(1)

print("All required columns FOUND")


# ============================================================
# PREPARE NUMERIC COLUMNS
# ============================================================

print_header("PREPARING NUMERIC COLUMNS")

for col in NUMERIC_COLUMNS:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

print("Numeric preparation completed.")


# ============================================================
# NORMALIZE TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "demand_available",
    "planning_reorder_status",
    "business_action",
    "planning_inventory_risk",
    "planning_stockout_risk",
    "inventory_risk",
    "stockout_risk",
    "reorder_priority",
    "reorder_status",
]

for col in TEXT_COLUMNS:

    if col in df.columns:

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# ============================================================
# NORMALIZE DEMAND AVAILABILITY
# ============================================================

demand_available_text = (
    df["demand_available"]
    .str.upper()
    .str.strip()
)

# Treat common truthy values as demand available.
DEMAND_TRUE_VALUES = {
    "TRUE",
    "1",
    "YES",
    "Y",
    "AVAILABLE",
    "DEMAND_AVAILABLE",
}

DEMAND_FALSE_VALUES = {
    "FALSE",
    "0",
    "NO",
    "N",
    "NONE",
    "UNAVAILABLE",
    "NO_FORECAST_DEMAND",
    "",
}

demand_available_bool = demand_available_text.isin(
    DEMAND_TRUE_VALUES
)

# If the source column contains an unknown value,
# infer demand availability from planning demand.
unknown_demand_values = ~(
    demand_available_text.isin(
        DEMAND_TRUE_VALUES | DEMAND_FALSE_VALUES
    )
)

demand_available_bool.loc[
    unknown_demand_values
] = (
    df.loc[
        unknown_demand_values,
        "planning_daily_demand"
    ].fillna(0) > 0
)

# If source says false but planning demand is positive,
# planning demand wins because it is the numerical basis.
demand_available_bool = (
    demand_available_bool
    | (
        df["planning_daily_demand"]
        .fillna(0)
        > 0
    )
)

df["_demand_available_bool"] = demand_available_bool


# ============================================================
# BASIC DATA QUALITY VALIDATION
# ============================================================

print_header("BASIC DATA QUALITY VALIDATION")

duplicate_store_sku = int(
    df.duplicated(
        subset=["store_id", "sku_id"]
    ).sum()
)

negative_stock = int(
    (df["stock_on_hand"] < 0).sum()
)

negative_forecast = int(
    (
        df["calibrated_forecast_30d"]
        .fillna(0)
        < 0
    ).sum()
)

negative_reorder = int(
    (
        df["suggested_reorder_quantity"]
        .fillna(0)
        < 0
    ).sum()
)

missing_planning_demand = int(
    df["planning_daily_demand"].isna().sum()
)

print(
    f"Duplicate Store-SKU:       "
    f"{duplicate_store_sku:,}"
)

print(
    f"Negative stock:             "
    f"{negative_stock:,}"
)

print(
    f"Negative forecast rows:     "
    f"{negative_forecast:,}"
)

print(
    f"Negative reorder quantity:  "
    f"{negative_reorder:,}"
)

print(
    f"Missing planning demand:    "
    f"{missing_planning_demand:,}"
)


# ============================================================
# PORTFOLIO METRICS
# ============================================================

print_header("CALCULATING PORTFOLIO METRICS")

total_stock = (
    df["stock_on_hand"]
    .fillna(0)
    .sum()
)

forecast_30 = (
    df["calibrated_forecast_30d"]
    .fillna(0)
    .sum()
)

forecast_60 = (
    df["calibrated_forecast_60d"]
    .fillna(0)
    .sum()
)

forecast_90 = (
    df["calibrated_forecast_90d"]
    .fillna(0)
    .sum()
)

planning_daily_total = (
    df["planning_daily_demand"]
    .fillna(0)
    .sum()
)

planning_30 = planning_daily_total * 30
planning_60 = planning_daily_total * 60
planning_90 = planning_daily_total * 90

stock_to_forecast_30 = (
    total_stock / forecast_30
    if forecast_30 > 0
    else np.nan
)

stock_to_forecast_60 = (
    total_stock / forecast_60
    if forecast_60 > 0
    else np.nan
)

stock_to_forecast_90 = (
    total_stock / forecast_90
    if forecast_90 > 0
    else np.nan
)

stock_to_planning_30 = (
    total_stock / planning_30
    if planning_30 > 0
    else np.nan
)

stock_to_planning_60 = (
    total_stock / planning_60
    if planning_60 > 0
    else np.nan
)

stock_to_planning_90 = (
    total_stock / planning_90
    if planning_90 > 0
    else np.nan
)

print(
    f"Total stock:                 "
    f"{fmt_num(total_stock)}"
)

print(
    f"Calibrated 30-day forecast:  "
    f"{fmt_num(forecast_30)}"
)

print(
    f"Calibrated 60-day forecast:  "
    f"{fmt_num(forecast_60)}"
)

print(
    f"Calibrated 90-day forecast:  "
    f"{fmt_num(forecast_90)}"
)

print()

print(
    f"Planning daily demand:       "
    f"{fmt_num(planning_daily_total)}"
)

print(
    f"Planning 30-day demand:      "
    f"{fmt_num(planning_30)}"
)

print(
    f"Planning 60-day demand:      "
    f"{fmt_num(planning_60)}"
)

print(
    f"Planning 90-day demand:      "
    f"{fmt_num(planning_90)}"
)

print()

print(
    f"Stock / 30-day forecast:     "
    f"{fmt_ratio(stock_to_forecast_30)}"
)

print(
    f"Stock / 60-day forecast:     "
    f"{fmt_ratio(stock_to_forecast_60)}"
)

print(
    f"Stock / 90-day forecast:     "
    f"{fmt_ratio(stock_to_forecast_90)}"
)

print()

print(
    f"Stock / planning 30d:        "
    f"{fmt_ratio(stock_to_planning_30)}"
)

print(
    f"Stock / planning 60d:        "
    f"{fmt_ratio(stock_to_planning_60)}"
)

print(
    f"Stock / planning 90d:        "
    f"{fmt_ratio(stock_to_planning_90)}"
)


# ============================================================
# CANONICAL PLANNING DOI
# ============================================================

print_header("VALIDATING DAYS OF INVENTORY")

# ------------------------------------------------------------
# IMPORTANT:
#
# The correct DOI for Phase 7.5 is based on the planning
# demand:
#
#     DOI = stock_on_hand / planning_daily_demand
#
# This is compared against planning_days_of_inventory.
#
# We DO NOT compare planning DOI against the older
# days_of_inventory column because that column can be based
# on another demand definition.
# ------------------------------------------------------------

df["_calculated_planning_doi"] = np.nan

valid_demand = (
    df["planning_daily_demand"]
    .fillna(0)
    > 0
)

df.loc[
    valid_demand,
    "_calculated_planning_doi"
] = (
    df.loc[
        valid_demand,
        "stock_on_hand"
    ]
    /
    df.loc[
        valid_demand,
        "planning_daily_demand"
    ]
)

# Existing planning DOI.
df["_source_planning_doi"] = (
    df["planning_days_of_inventory"]
)

# ------------------------------------------------------------
# Compare only finite, demand-available observations.
# ------------------------------------------------------------

doi_compare_mask = (
    valid_demand
    &
    df["_calculated_planning_doi"].notna()
    &
    df["_source_planning_doi"].notna()
    &
    np.isfinite(df["_calculated_planning_doi"])
    &
    np.isfinite(df["_source_planning_doi"])
)

if doi_compare_mask.any():

    doi_difference = (
        df.loc[
            doi_compare_mask,
            "_calculated_planning_doi"
        ]
        -
        df.loc[
            doi_compare_mask,
            "_source_planning_doi"
        ]
    ).abs()

    mean_abs_doi_difference = (
        doi_difference.mean()
    )

    max_doi_difference = (
        doi_difference.max()
    )

else:

    mean_abs_doi_difference = np.nan
    max_doi_difference = np.nan


# ------------------------------------------------------------
# DOI tolerance
#
# Floating point / rounding tolerance:
# 0.01 day.
# ------------------------------------------------------------

DOI_TOLERANCE = 0.01

doi_mismatch_count = 0

if doi_compare_mask.any():

    doi_mismatch_count = int(
        (
            doi_difference
            > DOI_TOLERANCE
        ).sum()
    )

print(
    f"Mean absolute DOI difference: "
    f"{fmt_num(mean_abs_doi_difference, 6)}"
)

print(
    f"Maximum DOI difference:        "
    f"{fmt_num(max_doi_difference, 6)}"
)

print(
    f"DOI mismatches > tolerance:    "
    f"{doi_mismatch_count:,}"
)


# ============================================================
# DEMAND AVAILABILITY
# ============================================================

print_header("ANALYZING DEMAND AVAILABILITY")

demand_available_count = int(
    df["_demand_available_bool"].sum()
)

no_forecast_count = int(
    (~df["_demand_available_bool"]).sum()
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
# INVENTORY COVERAGE
# ============================================================

# ------------------------------------------------------------
# Use canonical planning DOI.
# ------------------------------------------------------------

df["_canonical_doi"] = (
    df["_calculated_planning_doi"]
)

# For positive demand only.
doi_available = (
    df["_canonical_doi"].notna()
    &
    np.isfinite(df["_canonical_doi"])
)


# ============================================================
# VALIDATING EXCESS INVENTORY
# ============================================================

print_header("VALIDATING EXCESS INVENTORY")


def count_above_doi(days):
    return int(
        (
            doi_available
            &
            (
                df["_canonical_doi"]
                > days
            )
        ).sum()
    )


over_30_count = count_above_doi(30)
over_60_count = count_above_doi(60)
over_90_count = count_above_doi(90)
over_180_count = count_above_doi(180)
over_365_count = count_above_doi(365)


# ------------------------------------------------------------
# Severe overstock definition:
#
# > 180 days OR
# explicit planning inventory risk indicates overstock.
#
# The >180 threshold is the primary numerical rule.
# ------------------------------------------------------------

planning_risk_upper = (
    df["planning_inventory_risk"]
    .str.upper()
)

explicit_overstock = (
    planning_risk_upper.str.contains(
        "OVERSTOCK",
        na=False
    )
    &
    doi_available
)

severe_overstock_mask = (
    doi_available
    &
    (
        (
            df["_canonical_doi"]
            > 180
        )
        |
        explicit_overstock
    )
)

severe_overstock_count = int(
    severe_overstock_mask.sum()
)

severe_overstock_inventory = (
    df.loc[
        severe_overstock_mask,
        "stock_on_hand"
    ]
    .fillna(0)
    .sum()
)

severe_overstock_pct = (
    severe_overstock_inventory
    / total_stock
    * 100
    if total_stock > 0
    else 0
)

print(
    f"Store-SKU >30 DOI:   "
    f"{over_30_count:,}"
)

print(
    f"Store-SKU >60 DOI:   "
    f"{over_60_count:,}"
)

print(
    f"Store-SKU >90 DOI:   "
    f"{over_90_count:,}"
)

print(
    f"Store-SKU >180 DOI:  "
    f"{over_180_count:,}"
)

print(
    f"Store-SKU >365 DOI:  "
    f"{over_365_count:,}"
)

print()

print(
    f"Severe overstock Store-SKU: "
    f"{severe_overstock_count:,}"
)

print(
    f"Inventory in severe overstock: "
    f"{fmt_num(severe_overstock_inventory)}"
)

print(
    f"% of total inventory: "
    f"{fmt_num(severe_overstock_pct)}%"
)


# ============================================================
# DORMANT / NO-FORECAST INVENTORY
# ============================================================

print_header(
    "VALIDATING DORMANT / NO-FORECAST INVENTORY"
)

dormant_mask = (
    ~df["_demand_available_bool"]
)

dormant_inventory = (
    df.loc[
        dormant_mask,
        "stock_on_hand"
    ]
    .fillna(0)
    .sum()
)

dormant_forecast_30 = (
    df.loc[
        dormant_mask,
        "calibrated_forecast_30d"
    ]
    .fillna(0)
    .sum()
)

dormant_inventory_pct = (
    dormant_inventory
    / total_stock
    * 100
    if total_stock > 0
    else 0
)

print(
    f"No-forecast Store-SKU: "
    f"{no_forecast_count:,}"
)

print(
    f"Inventory held by no-forecast combinations: "
    f"{fmt_num(dormant_inventory)}"
)

print(
    f"30-day forecast from no-forecast combinations: "
    f"{fmt_num(dormant_forecast_30)}"
)

print(
    f"No-forecast inventory % of total: "
    f"{fmt_num(dormant_inventory_pct)}%"
)


# ============================================================
# VALIDATING REPLENISHMENT RECOMMENDATIONS
# ============================================================

print_header(
    "VALIDATING REPLENISHMENT RECOMMENDATIONS"
)

reorder_qty = (
    df["suggested_reorder_quantity"]
    .fillna(0)
    .clip(lower=0)
)

total_reorder_qty = reorder_qty.sum()

reorder_required_mask = (
    reorder_qty > 0
)

reorder_required_count = int(
    reorder_required_mask.sum()
)

status_upper = (
    df["planning_reorder_status"]
    .str.upper()
    .str.strip()
)

# ------------------------------------------------------------
# Status classification
# ------------------------------------------------------------

NO_REORDER_STATUS = {
    "NO_REORDER",
    "NO REORDER",
    "HOLD",
    "HOLD_REPLENISHMENT",
    "OVERSTOCK",
    "EXCESS",
    "EXCESS_STOCK",
    "NO_FORECAST_DEMAND",
    "NO_FORECAST",
}

REORDER_STATUS_KEYWORDS = [
    "REORDER",
    "REPLENISH",
    "ORDER",
    "RESTOCK",
]

# ------------------------------------------------------------
# IMPORTANT:
#
# Zero reorder quantity is VALID when:
#   - inventory is overstocked
#   - inventory is dormant/no-demand
#   - target stock does not exceed current stock
#
# Therefore, we do NOT require a positive quantity simply
# because DOI is high.
# ------------------------------------------------------------

positive_qty_no_reorder_status = int(
    (
        reorder_required_mask
        &
        status_upper.isin(
            NO_REORDER_STATUS
        )
    ).sum()
)

positive_qty_no_forecast = int(
    (
        reorder_required_mask
        &
        dormant_mask
    ).sum()
)

# ------------------------------------------------------------
# Determine whether the row genuinely requires reorder.
#
# A positive reorder quantity is the authoritative indicator.
# ------------------------------------------------------------

reorder_status_inconsistency_mask = (
    (
        reorder_required_mask
        &
        (
            status_upper.isin(
                NO_REORDER_STATUS
            )
        )
    )
)

# ------------------------------------------------------------
# Additional consistency:
#
# If quantity is zero, it is NOT an inconsistency merely
# because the status says no reorder.
#
# A zero quantity is valid for overstock and dormant rows.
# ------------------------------------------------------------

reorder_status_inconsistency_count = int(
    reorder_status_inconsistency_mask.sum()
)

# ------------------------------------------------------------
# More useful business check:
#
# If the status explicitly says REORDER / RESTOCK / ORDER
# but quantity is zero, mark inconsistency.
#
# However, "NO_REORDER", "OVERSTOCK", "HOLD", etc. are valid.
# ------------------------------------------------------------

explicit_reorder_status_mask = (
    status_upper.str.contains(
        "|".join(
            [
                "REORDER",
                "RESTOCK",
                "RESTOCKING",
                "REPLENISH",
            ]
        ),
        na=False
    )
    &
    ~status_upper.isin(
        NO_REORDER_STATUS
    )
)

status_requires_quantity_zero = (
    explicit_reorder_status_mask
    &
    ~reorder_required_mask
)

status_requires_quantity_zero_count = int(
    status_requires_quantity_zero.sum()
)

# ------------------------------------------------------------
# Final reorder status validation:
#
# 1. Positive quantity cannot have NO_REORDER type status.
# 2. Explicit REORDER status cannot have zero quantity.
# ------------------------------------------------------------

total_reorder_status_inconsistencies = int(
    (
        reorder_status_inconsistency_mask
        |
        status_requires_quantity_zero
    ).sum()
)

print(
    f"Total suggested reorder quantity: "
    f"{fmt_num(total_reorder_qty)}"
)

print(
    f"Store-SKU requiring reorder: "
    f"{reorder_required_count:,}"
)

print(
    f"NO_REORDER with positive quantity: "
    f"{positive_qty_no_reorder_status:,}"
)

print(
    f"NO_FORECAST_DEMAND with positive quantity: "
    f"{positive_qty_no_forecast:,}"
)

print(
    f"Explicit reorder status with zero quantity: "
    f"{status_requires_quantity_zero_count:,}"
)

print(
    f"Total reorder-status inconsistencies: "
    f"{total_reorder_status_inconsistencies:,}"
)


# ============================================================
# BUSINESS ACTION VALIDATION
# ============================================================

print_header("VALIDATING BUSINESS ACTIONS")

action_upper = (
    df["business_action"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
)


# ------------------------------------------------------------
# Semantic action detection
# ------------------------------------------------------------

def contains_any(series, keywords):
    pattern = "|".join(
        [
            str(x)
            for x in keywords
        ]
    )

    return series.str.contains(
        pattern,
        na=False,
        regex=True
    )


# Actions that are acceptable for severe overstock.
OVERSTOCK_ACTION_KEYWORDS = [
    "OVERSTOCK",
    "EXCESS",
    "REDUCE",
    "REDUCTION",
    "HOLD",
    "PAUSE",
    "TRANSFER",
    "MARKDOWN",
    "LIQUIDATION",
    "LIQUIDATE",
    "CLEARANCE",
    "PROMOTION",
    "PROMOTIONAL",
    "SLOW",
    "MONITOR",
    "REVIEW",
    "REPLENISHMENT",
    "REORDER",
]

# Actions that are acceptable for dormant/no-demand inventory.
DORMANT_ACTION_KEYWORDS = [
    "NO FORECAST",
    "NO_FORECAST",
    "DORMANT",
    "REVIEW",
    "TRANSFER",
    "MARKDOWN",
    "LIQUIDATION",
    "LIQUIDATE",
    "CLEARANCE",
    "MONITOR",
    "HOLD",
    "PAUSE",
]


# ------------------------------------------------------------
# Severe overstock action validation
# ------------------------------------------------------------

severe_action_ok = (
    contains_any(
        action_upper,
        OVERSTOCK_ACTION_KEYWORDS
    )
)

severe_overstock_action_mismatch_mask = (
    severe_overstock_mask
    &
    ~severe_action_ok
)

severe_overstock_action_mismatches = int(
    severe_overstock_action_mismatch_mask.sum()
)


# ------------------------------------------------------------
# Dormant action validation
# ------------------------------------------------------------

dormant_action_ok = (
    contains_any(
        action_upper,
        DORMANT_ACTION_KEYWORDS
    )
)

dormant_action_mismatch_mask = (
    dormant_mask
    &
    ~dormant_action_ok
)

dormant_action_mismatches = int(
    dormant_action_mismatch_mask.sum()
)

print(
    f"Severe-overstock action mismatches: "
    f"{severe_overstock_action_mismatches:,}"
)

print(
    f"No-forecast action mismatches:       "
    f"{dormant_action_mismatches:,}"
)


# ============================================================
# FORECAST VS PLANNING DEMAND
# ============================================================

print_header(
    "VALIDATING FORECAST VS PLANNING DEMAND"
)

forecast_planning_ratio = (
    forecast_30 / planning_30
    if planning_30 > 0
    else np.nan
)

forecast_planning_difference = (
    forecast_30 - planning_30
)

print(
    f"Calibrated 30-day forecast: "
    f"{fmt_num(forecast_30)}"
)

print(
    f"Planning 30-day demand:      "
    f"{fmt_num(planning_30)}"
)

print(
    f"Forecast / Planning ratio:   "
    f"{forecast_planning_ratio:.4f}"
)

print(
    f"Forecast - Planning:         "
    f"{fmt_num(forecast_planning_difference)}"
)


# ============================================================
# EXECUTIVE BUSINESS STATUS
# ============================================================

print_header(
    "DETERMINING EXECUTIVE BUSINESS STATUS"
)

if (
    stock_to_forecast_30 > 100
    or severe_overstock_pct >= 50
):

    business_inventory_status = (
        "CRITICAL_OVERSTOCK"
    )

elif (
    stock_to_forecast_30 > 30
    or severe_overstock_pct >= 25
):

    business_inventory_status = (
        "HIGH_OVERSTOCK"
    )

elif (
    stock_to_forecast_30 > 15
):

    business_inventory_status = (
        "ELEVATED_INVENTORY"
    )

else:

    business_inventory_status = (
        "BALANCED_INVENTORY"
    )

print(
    f"Business inventory status: "
    f"{business_inventory_status}"
)


# ============================================================
# STORE-LEVEL VALIDATION
# ============================================================

print_header(
    "CREATING STORE-LEVEL VALIDATION"
)

store_validation = (
    df.groupby("store_id", dropna=False)
    .agg(
        store_inventory=(
            "stock_on_hand",
            "sum"
        ),
        calibrated_forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),
        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),
        planning_30d_demand=(
            "planning_daily_demand",
            lambda x: x.fillna(0).sum() * 30
        ),
        store_sku_count=(
            "sku_id",
            "count"
        ),
        severe_overstock_count=(
            "_canonical_doi",
            lambda x: int(
                (
                    x > 180
                ).sum()
            )
        ),
        reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        ),
    )
    .reset_index()
)

store_validation[
    "inventory_to_forecast_30d_ratio"
] = safe_ratio(
    store_validation["store_inventory"],
    store_validation["calibrated_forecast_30d"]
)

store_validation[
    "inventory_to_planning_30d_ratio"
] = safe_ratio(
    store_validation["store_inventory"],
    store_validation["planning_30d_demand"]
)

store_validation[
    "store_inventory_status"
] = np.where(
    store_validation[
        "inventory_to_forecast_30d_ratio"
    ] > 100,
    "CRITICAL_OVERSTOCK",
    np.where(
        store_validation[
            "inventory_to_forecast_30d_ratio"
        ] > 30,
        "HIGH_OVERSTOCK",
        "REVIEW"
    )
)

print(
    f"Store-level rows created: "
    f"{len(store_validation):,}"
)


# ============================================================
# SKU-LEVEL VALIDATION
# ============================================================

print_header(
    "CREATING SKU-LEVEL VALIDATION"
)

sku_validation = (
    df.groupby("sku_id", dropna=False)
    .agg(
        sku_inventory=(
            "stock_on_hand",
            "sum"
        ),
        calibrated_forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),
        planning_daily_demand=(
            "planning_daily_demand",
            "sum"
        ),
        planning_30d_demand=(
            "planning_daily_demand",
            lambda x: x.fillna(0).sum() * 30
        ),
        store_count=(
            "store_id",
            "nunique"
        ),
        severe_overstock_count=(
            "_canonical_doi",
            lambda x: int(
                (
                    x > 180
                ).sum()
            )
        ),
        reorder_quantity=(
            "suggested_reorder_quantity",
            "sum"
        ),
    )
    .reset_index()
)

sku_validation[
    "inventory_to_forecast_30d_ratio"
] = safe_ratio(
    sku_validation["sku_inventory"],
    sku_validation["calibrated_forecast_30d"]
)

sku_validation[
    "inventory_to_planning_30d_ratio"
] = safe_ratio(
    sku_validation["sku_inventory"],
    sku_validation["planning_30d_demand"]
)

sku_validation[
    "sku_inventory_status"
] = np.where(
    sku_validation[
        "inventory_to_forecast_30d_ratio"
    ] > 100,
    "CRITICAL_OVERSTOCK",
    np.where(
        sku_validation[
            "inventory_to_forecast_30d_ratio"
        ] > 30,
        "HIGH_OVERSTOCK",
        "REVIEW"
    )
)

print(
    f"SKU-level rows created: "
    f"{len(sku_validation):,}"
)


# ============================================================
# EXTREME OVERSTOCK VALIDATION
# ============================================================

print_header(
    "CREATING EXTREME OVERSTOCK VALIDATION"
)

extreme_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "planning_target_stock",
    "inventory_gap_to_target",
    "suggested_reorder_quantity",
    "planning_reorder_status",
    "planning_inventory_risk",
    "business_action",
]

extreme_columns = [
    col
    for col in extreme_columns
    if col in df.columns
]

extreme_overstock = (
    df.loc[
        severe_overstock_mask,
        extreme_columns
    ]
    .copy()
)

extreme_overstock[
    "calculated_planning_doi"
] = df.loc[
    severe_overstock_mask,
    "_canonical_doi"
].values

extreme_overstock[
    "inventory_to_forecast_30d_ratio"
] = safe_ratio(
    extreme_overstock["stock_on_hand"],
    extreme_overstock["calibrated_forecast_30d"]
)

extreme_overstock = (
    extreme_overstock
    .sort_values(
        [
            "calculated_planning_doi",
            "stock_on_hand",
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    f"Extreme overstock rows: "
    f"{len(extreme_overstock):,}"
)


# ============================================================
# DORMANT INVENTORY VALIDATION
# ============================================================

print_header(
    "CREATING DORMANT INVENTORY VALIDATION"
)

dormant_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "planning_daily_demand",
    "calibrated_forecast_30d",
    "planning_target_stock",
    "inventory_gap_to_target",
    "suggested_reorder_quantity",
    "planning_reorder_status",
    "planning_inventory_risk",
    "business_action",
]

dormant_columns = [
    col
    for col in dormant_columns
    if col in df.columns
]

dormant_inventory_df = (
    df.loc[
        dormant_mask,
        dormant_columns
    ]
    .copy()
)

print(
    f"Dormant rows: "
    f"{len(dormant_inventory_df):,}"
)


# ============================================================
# REORDER VALIDATION DATASET
# ============================================================

print_header(
    "CREATING REORDER VALIDATION DATASET"
)

reorder_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "planning_daily_demand",
    "planning_days_of_inventory",
    "planning_target_stock",
    "inventory_gap_to_target",
    "suggested_reorder_quantity",
    "planning_reorder_status",
    "planning_inventory_risk",
    "planning_stockout_risk",
    "business_action",
]

reorder_columns = [
    col
    for col in reorder_columns
    if col in df.columns
]

reorder_validation = (
    df[reorder_columns]
    .copy()
)

reorder_validation[
    "calculated_planning_doi"
] = df["_canonical_doi"]

reorder_validation[
    "calculated_reorder_required"
] = reorder_required_mask.values

reorder_validation[
    "reorder_status_validation"
] = np.where(
    (
        reorder_status_inconsistency_mask
        |
        status_requires_quantity_zero
    ),
    "FAIL",
    "PASS"
)

reorder_validation[
    "overstock_zero_reorder_valid"
] = (
    severe_overstock_mask
    &
    (
        reorder_qty == 0
    )
).values

reorder_validation[
    "dormant_zero_reorder_valid"
] = (
    dormant_mask
    &
    (
        reorder_qty == 0
    )
).values

reorder_validation[
    "business_action_validation"
] = np.where(
    (
        severe_overstock_action_mismatch_mask
        |
        dormant_action_mismatch_mask
    ),
    "FAIL",
    "PASS"
)

print(
    f"Reorder validation rows: "
    f"{len(reorder_validation):,}"
)


# ============================================================
# BUSINESS RECOMMENDATION VALIDATION CHECKS
# ============================================================

print_header(
    "BUSINESS RECOMMENDATION VALIDATION CHECKS"
)

quality_checks = []


def add_check(name, passed, details):
    quality_checks.append(
        {
            "check_name": name,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "details": details,
        }
    )


# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

add_check(
    "duplicate_store_sku",
    duplicate_store_sku == 0,
    f"Duplicates: {duplicate_store_sku:,}",
)

add_check(
    "negative_stock",
    negative_stock == 0,
    f"Negative stock rows: {negative_stock:,}",
)

add_check(
    "negative_forecast",
    negative_forecast == 0,
    f"Negative forecast rows: {negative_forecast:,}",
)

add_check(
    "negative_reorder",
    negative_reorder == 0,
    f"Negative reorder rows: {negative_reorder:,}",
)

add_check(
    "missing_planning_demand",
    missing_planning_demand == 0,
    f"Missing planning demand: {missing_planning_demand:,}",
)


# ------------------------------------------------------------
# DOI check
# ------------------------------------------------------------

add_check(
    "doi_calculation",
    doi_mismatch_count == 0,
    (
        f"DOI mismatches: "
        f"{doi_mismatch_count:,}; "
        f"tolerance: {DOI_TOLERANCE} day"
    ),
)


# ------------------------------------------------------------
# Reorder check
# ------------------------------------------------------------

add_check(
    "reorder_status_consistency",
    total_reorder_status_inconsistencies == 0,
    (
        f"Status inconsistencies: "
        f"{total_reorder_status_inconsistencies:,}; "
        f"positive reorder rows: "
        f"{reorder_required_count:,}"
    ),
)


# ------------------------------------------------------------
# Overstock action check
# ------------------------------------------------------------

add_check(
    "overstock_action_consistency",
    severe_overstock_action_mismatches == 0,
    (
        f"Severe-overstock action mismatches: "
        f"{severe_overstock_action_mismatches:,}"
    ),
)


# ------------------------------------------------------------
# Dormant action check
# ------------------------------------------------------------

add_check(
    "dormant_action_consistency",
    dormant_action_mismatches == 0,
    (
        f"Dormant action mismatches: "
        f"{dormant_action_mismatches:,}"
    ),
)


# ------------------------------------------------------------
# Overall quality status
# ------------------------------------------------------------

all_checks_pass = all(
    x["status"] == "PASS"
    for x in quality_checks
)

data_quality_status = (
    "PASS"
    if all_checks_pass
    else "REVIEW"
)

for check in quality_checks:

    print(
        f"{check['check_name']:<35}: "
        f"{check['status']}"
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

print_header(
    "PHASE 7.5 EXECUTIVE VALIDATION SUMMARY"
)

print(
    f"Total Store-SKU:                 "
    f"{len(df):,}"
)

print(
    f"Total inventory:                  "
    f"{fmt_num(total_stock)}"
)

print(
    f"30-day calibrated forecast:       "
    f"{fmt_num(forecast_30)}"
)

print(
    f"30-day planning demand:           "
    f"{fmt_num(planning_30)}"
)

print(
    f"Inventory / 30-day forecast:      "
    f"{fmt_ratio(stock_to_forecast_30)}"
)

print(
    f"Inventory / planning 30-day:      "
    f"{fmt_ratio(stock_to_planning_30)}"
)

print(
    f">30 DOI Store-SKU:                "
    f"{over_30_count:,}"
)

print(
    f">60 DOI Store-SKU:                "
    f"{over_60_count:,}"
)

print(
    f">90 DOI Store-SKU:                "
    f"{over_90_count:,}"
)

print(
    f">180 DOI Store-SKU:               "
    f"{over_180_count:,}"
)

print(
    f">365 DOI Store-SKU:               "
    f"{over_365_count:,}"
)

print(
    f"Severe overstock Store-SKU:      "
    f"{severe_overstock_count:,}"
)

print(
    f"No-forecast Store-SKU:            "
    f"{no_forecast_count:,}"
)

print(
    f"No-forecast inventory:            "
    f"{fmt_num(dormant_inventory)}"
)

print(
    f"Suggested reorder quantity:       "
    f"{fmt_num(total_reorder_qty)}"
)

print(
    f"Store-SKU requiring reorder:      "
    f"{reorder_required_count:,}"
)

print(
    f"Data quality status:              "
    f"{data_quality_status}"
)


# ============================================================
# SUMMARY DATASET
# ============================================================

summary_rows = [
    {
        "metric": "total_store_sku",
        "value": len(df),
    },
    {
        "metric": "total_inventory",
        "value": total_stock,
    },
    {
        "metric": "calibrated_forecast_30d",
        "value": forecast_30,
    },
    {
        "metric": "calibrated_forecast_60d",
        "value": forecast_60,
    },
    {
        "metric": "calibrated_forecast_90d",
        "value": forecast_90,
    },
    {
        "metric": "planning_daily_demand",
        "value": planning_daily_total,
    },
    {
        "metric": "planning_demand_30d",
        "value": planning_30,
    },
    {
        "metric": "planning_demand_60d",
        "value": planning_60,
    },
    {
        "metric": "planning_demand_90d",
        "value": planning_90,
    },
    {
        "metric": "stock_to_forecast_30d_ratio",
        "value": stock_to_forecast_30,
    },
    {
        "metric": "stock_to_forecast_60d_ratio",
        "value": stock_to_forecast_60,
    },
    {
        "metric": "stock_to_forecast_90d_ratio",
        "value": stock_to_forecast_90,
    },
    {
        "metric": "stock_to_planning_30d_ratio",
        "value": stock_to_planning_30,
    },
    {
        "metric": "stock_to_planning_60d_ratio",
        "value": stock_to_planning_60,
    },
    {
        "metric": "stock_to_planning_90d_ratio",
        "value": stock_to_planning_90,
    },
    {
        "metric": "over_30_doi_count",
        "value": over_30_count,
    },
    {
        "metric": "over_60_doi_count",
        "value": over_60_count,
    },
    {
        "metric": "over_90_doi_count",
        "value": over_90_count,
    },
    {
        "metric": "over_180_doi_count",
        "value": over_180_count,
    },
    {
        "metric": "over_365_doi_count",
        "value": over_365_count,
    },
    {
        "metric": "severe_overstock_count",
        "value": severe_overstock_count,
    },
    {
        "metric": "severe_overstock_inventory",
        "value": severe_overstock_inventory,
    },
    {
        "metric": "severe_overstock_inventory_pct",
        "value": severe_overstock_pct,
    },
    {
        "metric": "no_forecast_count",
        "value": no_forecast_count,
    },
    {
        "metric": "no_forecast_inventory",
        "value": dormant_inventory,
    },
    {
        "metric": "no_forecast_inventory_pct",
        "value": dormant_inventory_pct,
    },
    {
        "metric": "total_reorder_quantity",
        "value": total_reorder_qty,
    },
    {
        "metric": "reorder_required_count",
        "value": reorder_required_count,
    },
    {
        "metric": "forecast_planning_ratio",
        "value": forecast_planning_ratio,
    },
    {
        "metric": "forecast_planning_difference",
        "value": forecast_planning_difference,
    },
    {
        "metric": "mean_abs_doi_difference",
        "value": mean_abs_doi_difference,
    },
    {
        "metric": "max_doi_difference",
        "value": max_doi_difference,
    },
    {
        "metric": "doi_mismatch_count",
        "value": doi_mismatch_count,
    },
    {
        "metric": "reorder_status_inconsistencies",
        "value": total_reorder_status_inconsistencies,
    },
    {
        "metric": "overstock_action_mismatches",
        "value": severe_overstock_action_mismatches,
    },
    {
        "metric": "dormant_action_mismatches",
        "value": dormant_action_mismatches,
    },
    {
        "metric": "business_inventory_status",
        "value": business_inventory_status,
    },
    {
        "metric": "data_quality_status",
        "value": data_quality_status,
    },
]

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# VALIDATED MASTER DATASET
# ============================================================

validated = df.copy()

validated[
    "calculated_planning_doi"
] = validated["_canonical_doi"]

validated[
    "doi_difference"
] = (
    validated["_calculated_planning_doi"]
    -
    validated["_source_planning_doi"]
).abs()

validated[
    "doi_validation"
] = np.where(
    (
        validated["doi_difference"]
        <= DOI_TOLERANCE
    )
    |
    (
        ~doi_available
    ),
    "PASS",
    "FAIL"
)

validated[
    "reorder_required"
] = (
    reorder_required_mask
)

validated[
    "reorder_status_validation"
] = np.where(
    (
        reorder_status_inconsistency_mask
        |
        status_requires_quantity_zero
    ),
    "FAIL",
    "PASS"
)

validated[
    "overstock_action_validation"
] = np.where(
    severe_overstock_action_mismatch_mask,
    "FAIL",
    "PASS"
)

validated[
    "dormant_action_validation"
] = np.where(
    dormant_action_mismatch_mask,
    "FAIL",
    "PASS"
)

validated[
    "business_inventory_status"
] = business_inventory_status

validated[
    "phase_7_5_data_quality_status"
] = data_quality_status


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

print_header(
    "SAVING PHASE 7.5 VALIDATION FILES"
)


store_validation.to_csv(
    STORE_OUTPUT,
    index=False
)

print(STORE_OUTPUT)


sku_validation.to_csv(
    SKU_OUTPUT,
    index=False
)

print(SKU_OUTPUT)


extreme_overstock.to_csv(
    EXTREME_OUTPUT,
    index=False
)

print(EXTREME_OUTPUT)


dormant_inventory_df.to_csv(
    DORMANT_OUTPUT,
    index=False
)

print(DORMANT_OUTPUT)


reorder_validation.to_csv(
    REORDER_OUTPUT,
    index=False
)

print(REORDER_OUTPUT)


# Remove internal helper columns before saving master.
MASTER_INTERNAL_COLUMNS = [
    "_demand_available_bool",
    "_calculated_planning_doi",
    "_source_planning_doi",
    "_canonical_doi",
]

validated_output = validated.drop(
    columns=[
        col
        for col in MASTER_INTERNAL_COLUMNS
        if col in validated.columns
    ],
    errors="ignore"
)

validated_output.to_csv(
    VALIDATED_OUTPUT,
    index=False
)

print(VALIDATED_OUTPUT)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print(SUMMARY_OUTPUT)


quality_df = pd.DataFrame(
    quality_checks
)

quality_df.to_csv(
    QUALITY_OUTPUT,
    index=False
)

print(QUALITY_OUTPUT)


# ============================================================
# EXECUTIVE REPORT
# ============================================================

print_header(
    "CREATING PHASE 7.5 EXECUTIVE REPORT"
)

report_lines = []

report_lines.append(
    "PROJECT FORESIGHT"
)

report_lines.append(
    "PHASE 7.5 - BUSINESS RECOMMENDATION VALIDATION"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "EXECUTIVE BUSINESS STATUS"
)

report_lines.append(
    f"Business inventory status: "
    f"{business_inventory_status}"
)

report_lines.append(
    f"Data quality status: "
    f"{data_quality_status}"
)

report_lines.append("")

report_lines.append(
    "PORTFOLIO METRICS"
)

report_lines.append(
    f"Total Store-SKU: {len(df):,}"
)

report_lines.append(
    f"Total inventory: {total_stock:,.2f}"
)

report_lines.append(
    f"Calibrated 30-day forecast: {forecast_30:,.2f}"
)

report_lines.append(
    f"Planning 30-day demand: {planning_30:,.2f}"
)

report_lines.append(
    f"Inventory / 30-day forecast: "
    f"{stock_to_forecast_30:.2f}x"
)

report_lines.append(
    f"Inventory / planning 30-day: "
    f"{stock_to_planning_30:.2f}x"
)

report_lines.append("")

report_lines.append(
    "INVENTORY COVERAGE"
)

report_lines.append(
    f">30 DOI: {over_30_count:,}"
)

report_lines.append(
    f">60 DOI: {over_60_count:,}"
)

report_lines.append(
    f">90 DOI: {over_90_count:,}"
)

report_lines.append(
    f">180 DOI: {over_180_count:,}"
)

report_lines.append(
    f">365 DOI: {over_365_count:,}"
)

report_lines.append(
    f"Severe overstock: "
    f"{severe_overstock_count:,}"
)

report_lines.append(
    f"Severe overstock inventory: "
    f"{severe_overstock_inventory:,.2f}"
)

report_lines.append(
    f"Severe overstock inventory %: "
    f"{severe_overstock_pct:.2f}%"
)

report_lines.append("")

report_lines.append(
    "DORMANT INVENTORY"
)

report_lines.append(
    f"No-forecast Store-SKU: "
    f"{no_forecast_count:,}"
)

report_lines.append(
    f"No-forecast inventory: "
    f"{dormant_inventory:,.2f}"
)

report_lines.append(
    f"No-forecast inventory %: "
    f"{dormant_inventory_pct:.2f}%"
)

report_lines.append("")

report_lines.append(
    "REPLENISHMENT"
)

report_lines.append(
    f"Suggested reorder quantity: "
    f"{total_reorder_qty:,.2f}"
)

report_lines.append(
    f"Store-SKU requiring reorder: "
    f"{reorder_required_count:,}"
)

report_lines.append(
    f"Reorder status inconsistencies: "
    f"{total_reorder_status_inconsistencies:,}"
)

report_lines.append("")

report_lines.append(
    "VALIDATION RESULTS"
)

for check in quality_checks:

    report_lines.append(
        f"{check['check_name']}: "
        f"{check['status']} - "
        f"{check['details']}"
    )

report_lines.append("")

report_lines.append(
    "FORECAST VS PLANNING DEMAND"
)

report_lines.append(
    f"Forecast / planning ratio: "
    f"{forecast_planning_ratio:.4f}"
)

report_lines.append(
    f"Forecast - planning: "
    f"{forecast_planning_difference:,.2f}"
)

report_lines.append("")

report_lines.append(
    "PHASE 7.5 BUSINESS INTERPRETATION"
)

if business_inventory_status == "CRITICAL_OVERSTOCK":

    report_lines.append(
        "CRITICAL: Inventory is substantially higher "
        "than near-term forecast demand."
    )

elif business_inventory_status == "HIGH_OVERSTOCK":

    report_lines.append(
        "WARNING: Inventory is materially higher "
        "than near-term forecast demand."
    )

else:

    report_lines.append(
        "Inventory does not meet the critical "
        "overstock threshold."
    )

report_lines.append("")

report_lines.append(
    f"{over_365_count:,} Store-SKU combinations "
    "have more than one year of planning inventory "
    "coverage."
)

report_lines.append(
    f"{no_forecast_count:,} Store-SKU combinations "
    "have no available forecast demand."
)

report_lines.append(
    "No additional replenishment is recommended "
    "where the calculated reorder quantity is zero."
)

report_lines.append("")

report_lines.append(
    "VALIDATED BUSINESS RECOMMENDATIONS"
)

report_lines.append(
    "1. Control / pause unnecessary replenishment"
)

report_lines.append(
    "2. Review extreme overstock"
)

report_lines.append(
    "3. Review dormant inventory"
)

report_lines.append(
    "4. Evaluate inter-store inventory transfers"
)

report_lines.append(
    "5. Consider markdown / liquidation where appropriate"
)

report_lines.append(
    "6. Continue monitoring actual demand vs forecast"
)

report_lines.append("")

# ------------------------------------------------------------
# Final recommendation
# ------------------------------------------------------------

if data_quality_status == "PASS":

    report_lines.append(
        "PHASE 7.5 FINAL DECISION"
    )

    report_lines.append(
        "Validation checks passed."
    )

    report_lines.append(
        "The business recommendation dataset is "
        "structurally valid."
    )

    report_lines.append(
        "Phase 8.0 dashboard preparation can proceed, "
        "while the CRITICAL_OVERSTOCK business finding "
        "must remain visible in the dashboard."
    )

else:

    report_lines.append(
        "PHASE 7.5 FINAL DECISION"
    )

    report_lines.append(
        "Validation checks require review."
    )

    report_lines.append(
        "Phase 8.0 should not be finalized until "
        "the failed validation checks are resolved."
    )

report_lines.append("")

report_lines.append(
    "IMPORTANT BUSINESS CONCLUSION"
)

report_lines.append(
    "Inventory remains substantially higher than "
    "near-term forecast demand."
)

report_lines.append(
    "The model recommends controlling additional "
    "replenishment and focusing management attention "
    "on excess inventory."
)

REPORT_OUTPUT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(REPORT_OUTPUT)


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print_header(
    "PHASE 7.5 FINAL DECISION"
)

print(
    f"DATA QUALITY STATUS: "
    f"{data_quality_status}"
)

print()

print(
    f"Business inventory status: "
    f"{business_inventory_status}"
)

print(
    f"Inventory / 30-day forecast: "
    f"{stock_to_forecast_30:.2f}x"
)

print(
    f"Inventory / planning 30-day: "
    f"{stock_to_planning_30:.2f}x"
)

print(
    f"Extreme overstock Store-SKU: "
    f"{severe_overstock_count:,}"
)

print(
    f"No-forecast Store-SKU: "
    f"{no_forecast_count:,}"
)

print(
    f"No-forecast inventory: "
    f"{dormant_inventory:,.2f}"
)

print(
    f"Suggested reorder quantity: "
    f"{total_reorder_qty:,.2f}"
)

print(
    f"Store-SKU requiring reorder: "
    f"{reorder_required_count:,}"
)

print()

print_header(
    "PHASE 7.5 BUSINESS INTERPRETATION"
)

if business_inventory_status == "CRITICAL_OVERSTOCK":

    print(
        "CRITICAL: Inventory is more than 100x "
        "the calibrated 30-day forecast."
    )

elif business_inventory_status == "HIGH_OVERSTOCK":

    print(
        "WARNING: Inventory is materially higher "
        "than the calibrated 30-day forecast."
    )

else:

    print(
        "Inventory is not classified as critical overstock."
    )

print()

print(
    f"{over_365_count:,} Store-SKU combinations "
    "have more than one year of planning inventory coverage."
)

print(
    f"{no_forecast_count:,} Store-SKU combinations "
    "have no forecast demand."
)

print()

print(
    "No additional replenishment is recommended "
    "under the current planning assumptions."
)

print()

print(
    "Validated recommendations:"
)

print(
    "1. Control / pause unnecessary replenishment"
)

print(
    "2. Review extreme overstock"
)

print(
    "3. Review dormant inventory"
)

print(
    "4. Evaluate inter-store inventory transfers"
)

print(
    "5. Consider markdown / liquidation where appropriate"
)

print(
    "6. Continue monitoring actual demand vs forecast"
)

print()

print_header(
    "PHASE 7.5 COMPLETED"
)

if data_quality_status == "PASS":

    print(
        "Business recommendations successfully validated."
    )

    print()

    print(
        "IMPORTANT BUSINESS CONCLUSION:"
    )

    print(
        "Inventory remains substantially higher than "
        "near-term forecast demand."
    )

    print()

    print(
        "PHASE 7.5 validation PASSED."
    )

    print(
        "Ready for:"
    )

    print(
        "PHASE 8.0 - FINAL BUSINESS ANALYTICS OUTPUT / "
        "DASHBOARD PREPARATION"
    )

else:

    print(
        "Business recommendations require review."
    )

    print()

    print(
        "IMPORTANT BUSINESS CONCLUSION:"
    )

    print(
        "The portfolio remains highly overstocked, "
        "but one or more validation rules require review."
    )

    print()

    print(
        "PHASE 7.5 requires review before proceeding "
        "to Phase 8.0."
    )

print("=" * WIDTH)