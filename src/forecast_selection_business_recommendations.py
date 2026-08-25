# ============================================================
# PROJECT FORESIGHT
# Phase 5.5 - Forecast Selection & Business Recommendations
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

FORECAST_PATH = PROCESSED_PATH / "forecasting"
EVALUATION_PATH = FORECAST_PATH / "evaluation"

INVENTORY_RISK_PATH = (
    PROCESSED_PATH
    / "inventory_analysis"
    / "inventory_demand_risk_analysis.csv"
)

BASELINE_PATH = (
    FORECAST_PATH
    / "demand_forecast_baseline.csv"
)

INTERMITTENT_PATH = (
    FORECAST_PATH
    / "intermittent_demand_forecast.csv"
)

STORE_SKU_EVALUATION_PATH = (
    EVALUATION_PATH
    / "store_sku_forecast_evaluation.csv"
)

OUTPUT_PATH = (
    FORECAST_PATH
    / "business_recommendations"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FORECAST SELECTION")
print("AND BUSINESS RECOMMENDATIONS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading store-SKU evaluation...")

evaluation = pd.read_csv(
    STORE_SKU_EVALUATION_PATH,
    low_memory=False
)

print(
    "Store-SKU evaluation shape:",
    evaluation.shape
)


print("\nLoading inventory risk analysis...")

inventory = pd.read_csv(
    INVENTORY_RISK_PATH,
    low_memory=False
)

print(
    "Inventory risk shape:",
    inventory.shape
)


print("\nLoading baseline forecast...")

baseline = pd.read_csv(
    BASELINE_PATH,
    low_memory=False
)

print(
    "Baseline forecast shape:",
    baseline.shape
)


print("\nLoading intermittent forecast...")

intermittent = pd.read_csv(
    INTERMITTENT_PATH,
    low_memory=False
)

print(
    "Intermittent forecast shape:",
    intermittent.shape
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)


required_evaluation_columns = [
    "store_id",
    "sku_id",
    "actual_units_30d",
    "baseline_forecast",
    "intermittent_forecast_30d",
    "baseline_abs_error",
    "intermittent_abs_error",
    "best_model"
]


missing_evaluation = [
    col
    for col in required_evaluation_columns
    if col not in evaluation.columns
]


if missing_evaluation:

    raise ValueError(
        "Missing evaluation columns: "
        + str(missing_evaluation)
    )


required_inventory_columns = [
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
    "inventory_status"
]


missing_inventory = [
    col
    for col in required_inventory_columns
    if col not in inventory.columns
]


if missing_inventory:

    raise ValueError(
        "Missing inventory columns: "
        + str(missing_inventory)
    )


print("\nEvaluation columns validated.")

print("Inventory columns validated.")


# ============================================================
# PREPARE FORECAST DATA
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FORECAST DATA")
print("=" * 70)


baseline_required = [
    "store_id",
    "sku_id",
    "forecast_weighted",
    "forecastability",
    "baseline_trend"
]


missing_baseline = [
    col
    for col in baseline_required
    if col not in baseline.columns
]


if missing_baseline:

    raise ValueError(
        "Missing baseline columns: "
        + str(missing_baseline)
    )


intermittent_required = [
    "store_id",
    "sku_id",
    "intermittent_forecast",
    "forecastability",
    "forecast_confidence"
]


missing_intermittent = [
    col
    for col in intermittent_required
    if col not in intermittent.columns
]


if missing_intermittent:

    raise ValueError(
        "Missing intermittent columns: "
        + str(missing_intermittent)
    )


baseline_selected = baseline[
    [
        "store_id",
        "sku_id",
        "forecast_weighted",
        "forecastability",
        "baseline_trend"
    ]
].copy()


baseline_selected = baseline_selected.rename(
    columns={
        "forecast_weighted":
            "baseline_daily_forecast",
        "forecastability":
            "baseline_forecastability",
        "baseline_trend":
            "baseline_demand_trend"
    }
)


intermittent_selected = intermittent[
    [
        "store_id",
        "sku_id",
        "intermittent_forecast",
        "forecastability",
        "forecast_confidence"
    ]
].copy()


intermittent_selected = intermittent_selected.rename(
    columns={
        "intermittent_forecast":
            "intermittent_daily_forecast",
        "forecastability":
            "intermittent_forecastability"
    }
)


# ============================================================
# MERGE FORECAST INFORMATION
# ============================================================

print("\nMerging forecast information...")


result = evaluation[
    [
        "store_id",
        "sku_id",
        "actual_units_30d",
        "baseline_forecast",
        "intermittent_forecast_30d",
        "baseline_abs_error",
        "intermittent_abs_error",
        "best_model"
    ]
].copy()


result = result.merge(
    baseline_selected,
    on=["store_id", "sku_id"],
    how="left"
)


result = result.merge(
    intermittent_selected,
    on=["store_id", "sku_id"],
    how="left"
)


# ============================================================
# MERGE INVENTORY RISK
# ============================================================

print("Merging inventory risk information...")


inventory_selected = inventory[
    required_inventory_columns
].copy()


result = result.merge(
    inventory_selected,
    on=["store_id", "sku_id"],
    how="left"
)


# ============================================================
# SELECT RECOMMENDED MODEL
# ============================================================

print("\n" + "=" * 70)
print("SELECTING RECOMMENDED FORECAST MODEL")
print("=" * 70)


result["recommended_model"] = np.where(
    result["best_model"].eq("Intermittent"),
    "Intermittent",
    "Baseline"
)


# ============================================================
# RECOMMENDED DAILY FORECAST
# ============================================================

result["recommended_daily_forecast"] = np.where(
    result["recommended_model"].eq("Intermittent"),
    result["intermittent_daily_forecast"],
    result["baseline_daily_forecast"]
)


result["recommended_daily_forecast"] = (
    result["recommended_daily_forecast"]
    .clip(lower=0)
)


# ============================================================
# RECOMMENDED 30-DAY FORECAST
# ============================================================

result["recommended_30d_forecast"] = (
    result["recommended_daily_forecast"] * 30
)


# ============================================================
# FORECAST VS CURRENT STOCK
# ============================================================

result["forecast_30d_to_stock_ratio"] = np.where(
    result["stock_on_hand"] > 0,
    result["recommended_30d_forecast"]
    / result["stock_on_hand"],
    np.nan
)


result["forecast_30d_stock_gap"] = (
    result["recommended_30d_forecast"]
    - result["stock_on_hand"]
)


# ============================================================
# PROJECTED STOCK AFTER 30 DAYS
# ============================================================

result["projected_stock_after_30d"] = (
    result["stock_on_hand"]
    - result["recommended_30d_forecast"]
)


# ============================================================
# REORDER QUANTITY
# ============================================================

result["recommended_reorder_qty"] = np.maximum(
    result["recommended_30d_forecast"]
    - result["stock_on_hand"],
    0
)


# ============================================================
# SAFETY STOCK ADJUSTMENT
# ============================================================

result["target_stock_level"] = (
    result["recommended_30d_forecast"]
    + result["safety_stock"]
)


result["recommended_reorder_qty_with_safety"] = np.maximum(
    result["target_stock_level"]
    - result["stock_on_hand"],
    0
)


# ============================================================
# FORECAST CONFIDENCE
# ============================================================

def determine_forecast_confidence(row):

    model = row["recommended_model"]

    baseline_conf = row[
        "baseline_forecastability"
    ]

    intermittent_conf = row[
        "intermittent_forecastability"
    ]

    intermittent_confidence = row[
        "forecast_confidence"
    ]

    if model == "Intermittent":

        if intermittent_confidence == "High":
            return "High"

        elif intermittent_confidence == "Medium":
            return "Medium"

        elif intermittent_confidence == "Low-Medium":
            return "Low-Medium"

        else:
            return "Low"

    else:

        if baseline_conf == "Difficult":
            return "Low"

        elif baseline_conf == "Very Difficult":
            return "Very Low"

        else:
            return "Medium"


result["recommended_forecast_confidence"] = (
    result.apply(
        determine_forecast_confidence,
        axis=1
    )
)


# ============================================================
# BUSINESS ACTION
# ============================================================

def determine_business_action(row):

    risk = row["risk_category"]
    model = row["recommended_model"]
    reorder_qty = row[
        "recommended_reorder_qty_with_safety"
    ]

    stock = row["stock_on_hand"]
    safety = row["safety_stock"]

    demand_30 = row["actual_units_30d"]

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if risk == "Critical":

        if reorder_qty > 0:
            return "Urgent replenishment"

        return "Urgent inventory review"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if risk == "High":

        if reorder_qty > 0:
            return "Prioritize replenishment"

        return "Monitor high-risk inventory"

    # --------------------------------------------------------
    # NO DEMAND
    # --------------------------------------------------------

    if risk == "No Demand":

        return "Review inactive inventory"

    # --------------------------------------------------------
    # OVERSTOCK
    # --------------------------------------------------------

    if risk == "Potential Overstock":

        if demand_30 == 0:
            return "Stop replenishment and review inactive stock"

        return "Review excess inventory"

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if risk == "Medium":

        if reorder_qty > 0:
            return "Plan replenishment"

        return "Monitor inventory"

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    if reorder_qty > 0:
        return "Plan replenishment"

    return "Maintain normal inventory"


result["business_action"] = (
    result.apply(
        determine_business_action,
        axis=1
    )
)


# ============================================================
# REPLENISHMENT PRIORITY
# ============================================================

def determine_replenishment_priority(row):

    risk = row["risk_category"]

    reorder_qty = row[
        "recommended_reorder_qty_with_safety"
    ]

    confidence = row[
        "recommended_forecast_confidence"
    ]

    if risk == "Critical" and reorder_qty > 0:
        return "P1 - Urgent"

    if risk == "High" and reorder_qty > 0:
        return "P2 - High"

    if risk == "Medium" and reorder_qty > 0:
        return "P3 - Medium"

    if reorder_qty > 0:
        return "P4 - Planned"

    return "No Replenishment"


result["replenishment_priority"] = (
    result.apply(
        determine_replenishment_priority,
        axis=1
    )
)


# ============================================================
# INVENTORY DECISION
# ============================================================

def determine_inventory_decision(row):

    risk = row["risk_category"]

    projected_stock = row[
        "projected_stock_after_30d"
    ]

    safety = row["safety_stock"]

    reorder_qty = row[
        "recommended_reorder_qty_with_safety"
    ]

    if risk == "No Demand":

        return "Investigate inactive stock"

    if risk == "Potential Overstock":

        return "Reduce excess inventory"

    if projected_stock <= 0:

        return "Replenish immediately"

    if projected_stock <= safety:

        return "Replenish before safety breach"

    if reorder_qty > 0:

        return "Plan replenishment"

    return "Maintain stock"


result["inventory_decision"] = (
    result.apply(
        determine_inventory_decision,
        axis=1
    )
)


# ============================================================
# FINAL BUSINESS PRIORITY SCORE
# ============================================================

def calculate_business_priority(row):

    score = 0

    risk = row["risk_category"]

    reorder_qty = row[
        "recommended_reorder_qty_with_safety"
    ]

    projected_stock = row[
        "projected_stock_after_30d"
    ]

    safety = row["safety_stock"]

    # Risk contribution

    if risk == "Critical":
        score += 60

    elif risk == "High":
        score += 45

    elif risk == "Medium":
        score += 25

    elif risk == "Potential Overstock":
        score += 20

    elif risk == "No Demand":
        score += 15

    # Replenishment requirement

    if reorder_qty > 0:
        score += 20

    # Projected stock risk

    if projected_stock <= 0:
        score += 20

    elif projected_stock <= safety:
        score += 10

    return min(score, 100)


result["business_priority_score"] = (
    result.apply(
        calculate_business_priority,
        axis=1
    )
)


# ============================================================
# FINAL PRIORITY CATEGORY
# ============================================================

def classify_business_priority(score):

    if score >= 75:
        return "Critical Priority"

    elif score >= 50:
        return "High Priority"

    elif score >= 25:
        return "Medium Priority"

    elif score > 0:
        return "Low Priority"

    return "Normal"


result["business_priority"] = (
    result["business_priority_score"]
    .apply(classify_business_priority)
)


# ============================================================
# MODEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDED MODEL DISTRIBUTION")
print("=" * 70)

print(
    result["recommended_model"]
    .value_counts()
)


# ============================================================
# BUSINESS ACTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS ACTION DISTRIBUTION")
print("=" * 70)

print(
    result["business_action"]
    .value_counts()
)


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS PRIORITY DISTRIBUTION")
print("=" * 70)

print(
    result["business_priority"]
    .value_counts()
)


# ============================================================
# REPLENISHMENT PRIORITY
# ============================================================

print("\n" + "=" * 70)
print("REPLENISHMENT PRIORITY DISTRIBUTION")
print("=" * 70)

print(
    result["replenishment_priority"]
    .value_counts()
)


# ============================================================
# TOP REPLENISHMENT ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 REPLENISHMENT RECOMMENDATIONS")
print("=" * 70)


replenishment = (
    result[
        result["recommended_reorder_qty_with_safety"] > 0
    ]
    .sort_values(
        [
            "business_priority_score",
            "recommended_reorder_qty_with_safety"
        ],
        ascending=[False, False]
    )
)


replenishment_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "safety_stock",
    "recommended_30d_forecast",
    "recommended_reorder_qty_with_safety",
    "risk_category",
    "risk_score",
    "recommended_model",
    "recommended_forecast_confidence",
    "business_action",
    "replenishment_priority",
    "business_priority"
]


print(
    replenishment[
        replenishment_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TOP OVERSTOCK ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 OVERSTOCK RECOMMENDATIONS")
print("=" * 70)


overstock = (
    result[
        result["risk_category"]
        == "Potential Overstock"
    ]
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
)


overstock_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "units_30d",
    "recommended_30d_forecast",
    "days_of_inventory",
    "demand_trend",
    "risk_category",
    "risk_score",
    "business_action",
    "inventory_decision"
]


print(
    overstock[
        overstock_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TOP NO-DEMAND ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 NO-DEMAND INVENTORY ITEMS")
print("=" * 70)


no_demand = (
    result[
        result["risk_category"]
        == "No Demand"
    ]
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
)


no_demand_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "units_30d",
    "units_90d",
    "days_of_inventory_capped",
    "risk_score",
    "business_action",
    "inventory_decision"
]


print(
    no_demand[
        no_demand_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TOP CRITICAL PRIORITY ITEMS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 CRITICAL BUSINESS PRIORITY ITEMS")
print("=" * 70)


critical_priority = (
    result[
        result["business_priority"]
        == "Critical Priority"
    ]
    .sort_values(
        "business_priority_score",
        ascending=False
    )
)


priority_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "recommended_30d_forecast",
    "recommended_reorder_qty_with_safety",
    "projected_stock_after_30d",
    "risk_category",
    "risk_score",
    "recommended_model",
    "business_action",
    "inventory_decision",
    "business_priority_score",
    "business_priority"
]


print(
    critical_priority[
        priority_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# STORE LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STORE-LEVEL BUSINESS SUMMARY")
print("=" * 70)


store_summary = (
    result
    .groupby("store_id")
    .agg(
        total_sku=("sku_id", "nunique"),
        critical_items=(
            "risk_category",
            lambda x: (x == "Critical").sum()
        ),
        high_risk_items=(
            "risk_category",
            lambda x: (x == "High").sum()
        ),
        overstock_items=(
            "risk_category",
            lambda x: (
                x == "Potential Overstock"
            ).sum()
        ),
        no_demand_items=(
            "risk_category",
            lambda x: (
                x == "No Demand"
            ).sum()
        ),
        recommended_replenishment=(
            "recommended_reorder_qty_with_safety",
            lambda x: (x > 0).sum()
        ),
        total_reorder_quantity=(
            "recommended_reorder_qty_with_safety",
            "sum"
        ),
        avg_business_priority=(
            "business_priority_score",
            "mean"
        )
    )
    .reset_index()
)


store_summary = store_summary.sort_values(
    "avg_business_priority",
    ascending=False
)


print(
    store_summary
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SKU LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SKU-LEVEL BUSINESS SUMMARY")
print("=" * 70)


sku_summary = (
    result
    .groupby("sku_id")
    .agg(
        stores=("store_id", "nunique"),
        critical_items=(
            "risk_category",
            lambda x: (x == "Critical").sum()
        ),
        high_risk_items=(
            "risk_category",
            lambda x: (x == "High").sum()
        ),
        overstock_items=(
            "risk_category",
            lambda x: (
                x == "Potential Overstock"
            ).sum()
        ),
        no_demand_items=(
            "risk_category",
            lambda x: (
                x == "No Demand"
            ).sum()
        ),
        total_reorder_quantity=(
            "recommended_reorder_qty_with_safety",
            "sum"
        ),
        avg_business_priority=(
            "business_priority_score",
            "mean"
        )
    )
    .reset_index()
)


sku_summary = sku_summary.sort_values(
    "avg_business_priority",
    ascending=False
)


print(
    sku_summary
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE COMPLETE DECISION DATASET
# ============================================================

output_file = (
    OUTPUT_PATH
    / "forecast_selection_business_recommendations.csv"
)


result.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE REPLENISHMENT FILE
# ============================================================

replenishment_file = (
    OUTPUT_PATH
    / "replenishment_recommendations.csv"
)


replenishment.to_csv(
    replenishment_file,
    index=False
)


# ============================================================
# SAVE OVERSTOCK FILE
# ============================================================

overstock_file = (
    OUTPUT_PATH
    / "overstock_recommendations.csv"
)


overstock.to_csv(
    overstock_file,
    index=False
)


# ============================================================
# SAVE NO-DEMAND FILE
# ============================================================

no_demand_file = (
    OUTPUT_PATH
    / "no_demand_inventory.csv"
)


no_demand.to_csv(
    no_demand_file,
    index=False
)


# ============================================================
# SAVE STORE SUMMARY
# ============================================================

store_summary_file = (
    OUTPUT_PATH
    / "store_business_summary.csv"
)


store_summary.to_csv(
    store_summary_file,
    index=False
)


# ============================================================
# SAVE SKU SUMMARY
# ============================================================

sku_summary_file = (
    OUTPUT_PATH
    / "sku_business_summary.csv"
)


sku_summary.to_csv(
    sku_summary_file,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    "Final result shape:",
    result.shape
)


print(
    "Missing recommended models:",
    result["recommended_model"].isna().sum()
)


print(
    "Missing recommended forecasts:",
    result[
        "recommended_30d_forecast"
    ].isna().sum()
)


print(
    "Negative recommended forecasts:",
    (
        result[
            "recommended_30d_forecast"
        ] < 0
    ).sum()
)


print(
    "Missing business actions:",
    result["business_action"].isna().sum()
)


print(
    "Missing inventory decisions:",
    result["inventory_decision"].isna().sum()
)


print(
    "Missing business priority:",
    result["business_priority"].isna().sum()
)


print(
    "Total recommended reorder quantity:",
    round(
        result[
            "recommended_reorder_qty_with_safety"
        ].sum(),
        2
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5.5 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nMain output:")
print(output_file)


print("\nReplenishment output:")
print(replenishment_file)


print("\nOverstock output:")
print(overstock_file)


print("\nNo-demand output:")
print(no_demand_file)


print("\nStore summary:")
print(store_summary_file)


print("\nSKU summary:")
print(sku_summary_file)


print("\n" + "=" * 70)
print("NEXT PHASE: FORECAST + INVENTORY INTEGRATION")
print("=" * 70)