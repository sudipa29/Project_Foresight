# ================================================================
# PROJECT FORESIGHT
# FORECAST + INVENTORY INTEGRATION
# ================================================================

import os
import glob
import pandas as pd
import numpy as np


# ================================================================
# CONFIGURATION
# ================================================================

BASE_PATH = r"E:\Zidio_Development_Internship\Project_Foresight"

PROCESSED_PATH = os.path.join(
    BASE_PATH,
    "data",
    "processed"
)

FORECASTING_PATH = os.path.join(
    PROCESSED_PATH,
    "forecasting"
)

BUSINESS_PATH = os.path.join(
    FORECASTING_PATH,
    "business_recommendations"
)

OUTPUT_PATH = os.path.join(
    FORECASTING_PATH,
    "integration"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ================================================================
# INPUT FILES
# ================================================================

BUSINESS_FILE = os.path.join(
    BUSINESS_PATH,
    "forecast_selection_business_recommendations.csv"
)

BASELINE_FILE = os.path.join(
    FORECASTING_PATH,
    "demand_forecast_baseline.csv"
)

INTERMITTENT_FILE = os.path.join(
    FORECASTING_PATH,
    "intermittent_demand_forecast.csv"
)


# ================================================================
# REQUIRED INVENTORY COLUMNS
# ================================================================

REQUIRED_INVENTORY_COLUMNS = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]


# ================================================================
# FIND DETAILED INVENTORY FILE
# ================================================================

def find_inventory_file():

    print("=" * 70)
    print("SEARCHING FOR DETAILED INVENTORY FILE")
    print("=" * 70)

    search_paths = [
        PROCESSED_PATH,
        os.path.join(PROCESSED_PATH, "inventory_analysis")
    ]

    candidates = []

    for search_path in search_paths:

        if not os.path.exists(search_path):
            continue

        pattern = os.path.join(search_path, "*.csv")

        for file in glob.glob(pattern):

            filename = os.path.basename(file).lower()

            # Ignore forecast summary / metric files
            if "forecast_inventory_risk_summary" in filename:
                continue

            try:

                df_sample = pd.read_csv(
                    file,
                    nrows=5
                )

                columns = set(df_sample.columns)

                if all(
                    col in columns
                    for col in REQUIRED_INVENTORY_COLUMNS
                ):

                    candidates.append(file)

            except Exception as e:

                print(
                    f"Could not inspect: {file}"
                )

    if not candidates:

        raise FileNotFoundError(
            "\nNo detailed inventory file was found.\n"
            "The file must contain these columns:\n"
            f"{REQUIRED_INVENTORY_COLUMNS}\n"
            "\nPlease check the files inside:\n"
            f"{PROCESSED_PATH}\\inventory_analysis"
        )

    print("\nDetailed inventory candidates found:")

    for i, file in enumerate(candidates, 1):
        print(f"{i}. {file}")

    # Prefer inventory risk/detail files
    preferred = [
        f for f in candidates
        if "risk" in os.path.basename(f).lower()
    ]

    if preferred:
        selected = preferred[0]
    else:
        selected = candidates[0]

    print("\nSelected inventory file:")
    print(selected)

    return selected


# ================================================================
# START
# ================================================================

print("=" * 70)
print("PROJECT FORESIGHT - FORECAST + INVENTORY INTEGRATION")
print("=" * 70)


# ================================================================
# CHECK FILES
# ================================================================

print("\nChecking input files...")

if not os.path.exists(BUSINESS_FILE):
    raise FileNotFoundError(
        f"Missing business recommendations file:\n{BUSINESS_FILE}"
    )

print("Found: Business recommendations")


if not os.path.exists(BASELINE_FILE):
    raise FileNotFoundError(
        f"Missing baseline forecast:\n{BASELINE_FILE}"
    )

print("Found: Baseline forecast")


if not os.path.exists(INTERMITTENT_FILE):
    raise FileNotFoundError(
        f"Missing intermittent forecast:\n{INTERMITTENT_FILE}"
    )

print("Found: Intermittent forecast")


# Find correct inventory file
INVENTORY_FILE = find_inventory_file()


# ================================================================
# LOAD DATA
# ================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)


print("\nLoading business recommendations...")

business = pd.read_csv(
    BUSINESS_FILE
)

print(
    f"Business recommendations shape: {business.shape}"
)


print("\nLoading inventory risk/detail data...")

inventory = pd.read_csv(
    INVENTORY_FILE
)

print(
    f"Inventory shape: {inventory.shape}"
)


print("\nLoading baseline forecast...")

baseline = pd.read_csv(
    BASELINE_FILE
)

print(
    f"Baseline forecast shape: {baseline.shape}"
)


print("\nLoading intermittent forecast...")

intermittent = pd.read_csv(
    INTERMITTENT_FILE
)

print(
    f"Intermittent forecast shape: {intermittent.shape}"
)


# ================================================================
# BASIC VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)


print("\nBusiness columns:")
print(list(business.columns))


print("\nInventory columns:")
print(list(inventory.columns))


# ================================================================
# VALIDATE INVENTORY
# ================================================================

missing_inventory_columns = [
    col
    for col in REQUIRED_INVENTORY_COLUMNS
    if col not in inventory.columns
]

if missing_inventory_columns:

    raise ValueError(
        "\nInventory is missing required columns:\n"
        f"{missing_inventory_columns}\n"
        "\nActual columns:\n"
        f"{list(inventory.columns)}"
    )


print("\nInventory required columns validated successfully.")


# ================================================================
# VALIDATE BUSINESS
# ================================================================

required_business_columns = [
    "store_id",
    "sku_id",
    "recommended_model",
    "recommended_daily_forecast",
    "recommended_30d_forecast",
    "recommended_reorder_qty",
    "recommended_reorder_qty_with_safety",
    "business_action",
    "replenishment_priority",
    "inventory_decision",
    "business_priority"
]


missing_business_columns = [
    col
    for col in required_business_columns
    if col not in business.columns
]


if missing_business_columns:

    raise ValueError(
        "\nBusiness recommendations are missing columns:\n"
        f"{missing_business_columns}"
    )


print(
    "\nBusiness recommendation columns validated successfully."
)


# ================================================================
# VALIDATE FORECAST FILES
# ================================================================

baseline_required = [
    "store_id",
    "sku_id",
    "forecast_weighted"
]

missing_baseline = [
    col
    for col in baseline_required
    if col not in baseline.columns
]

if missing_baseline:

    raise ValueError(
        "\nBaseline forecast missing columns:\n"
        f"{missing_baseline}"
    )


intermittent_required = [
    "store_id",
    "sku_id",
    "intermittent_forecast"
]

missing_intermittent = [
    col
    for col in intermittent_required
    if col not in intermittent.columns
]

if missing_intermittent:

    raise ValueError(
        "\nIntermittent forecast missing columns:\n"
        f"{missing_intermittent}"
    )


print(
    "Forecast columns validated successfully."
)


# ================================================================
# STANDARDIZE IDS
# ================================================================

print("\n" + "=" * 70)
print("STANDARDIZING STORE-SKU IDENTIFIERS")
print("=" * 70)


for df in [
    business,
    inventory,
    baseline,
    intermittent
]:

    df["store_id"] = pd.to_numeric(
        df["store_id"],
        errors="coerce"
    )

    df["sku_id"] = pd.to_numeric(
        df["sku_id"],
        errors="coerce"
    )


# Remove invalid IDs
business = business.dropna(
    subset=["store_id", "sku_id"]
)

inventory = inventory.dropna(
    subset=["store_id", "sku_id"]
)

baseline = baseline.dropna(
    subset=["store_id", "sku_id"]
)

intermittent = intermittent.dropna(
    subset=["store_id", "sku_id"]
)


business["store_id"] = business["store_id"].astype(int)
business["sku_id"] = business["sku_id"].astype(int)

inventory["store_id"] = inventory["store_id"].astype(int)
inventory["sku_id"] = inventory["sku_id"].astype(int)

baseline["store_id"] = baseline["store_id"].astype(int)
baseline["sku_id"] = baseline["sku_id"].astype(int)

intermittent["store_id"] = intermittent["store_id"].astype(int)
intermittent["sku_id"] = intermittent["sku_id"].astype(int)


# ================================================================
# NUMERIC CONVERSION
# ================================================================

numeric_columns_inventory = [
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

for col in numeric_columns_inventory:

    inventory[col] = pd.to_numeric(
        inventory[col],
        errors="coerce"
    )


forecast_numeric_columns = [
    "recommended_daily_forecast",
    "recommended_30d_forecast",
    "recommended_reorder_qty",
    "recommended_reorder_qty_with_safety"
]

for col in forecast_numeric_columns:

    if col in business.columns:

        business[col] = pd.to_numeric(
            business[col],
            errors="coerce"
        )


baseline["forecast_weighted"] = pd.to_numeric(
    baseline["forecast_weighted"],
    errors="coerce"
)

intermittent["intermittent_forecast"] = pd.to_numeric(
    intermittent["intermittent_forecast"],
    errors="coerce"
)


# ================================================================
# REMOVE DUPLICATES
# ================================================================

business = business.drop_duplicates(
    subset=["store_id", "sku_id"],
    keep="first"
)

baseline = baseline.drop_duplicates(
    subset=["store_id", "sku_id"],
    keep="first"
)

intermittent = intermittent.drop_duplicates(
    subset=["store_id", "sku_id"],
    keep="first"
)

inventory = inventory.drop_duplicates(
    subset=["store_id", "sku_id"],
    keep="last"
)


# ================================================================
# PREPARE FORECAST DATA
# ================================================================

print("\n" + "=" * 70)
print("PREPARING FORECAST DATA")
print("=" * 70)


baseline_small = baseline[
    [
        "store_id",
        "sku_id",
        "forecast_weighted"
    ]
].copy()


baseline_small = baseline_small.rename(
    columns={
        "forecast_weighted":
        "baseline_daily_forecast_integration"
    }
)


intermittent_small = intermittent[
    [
        "store_id",
        "sku_id",
        "intermittent_forecast"
    ]
].copy()


intermittent_small = intermittent_small.rename(
    columns={
        "intermittent_forecast":
        "intermittent_daily_forecast_integration"
    }
)


# ================================================================
# MERGE BUSINESS + INVENTORY
# ================================================================

print("\nMerging business recommendations with inventory...")

integrated = business.merge(
    inventory,
    on=["store_id", "sku_id"],
    how="left",
    suffixes=("", "_inventory")
)


print(
    f"After business + inventory merge: {integrated.shape}"
)


# ================================================================
# MERGE BASELINE
# ================================================================

print("\nMerging baseline forecasts...")

integrated = integrated.merge(
    baseline_small,
    on=["store_id", "sku_id"],
    how="left"
)


# ================================================================
# MERGE INTERMITTENT
# ================================================================

print("Merging intermittent forecasts...")

integrated = integrated.merge(
    intermittent_small,
    on=["store_id", "sku_id"],
    how="left"
)


# ================================================================
# FORECAST INTEGRATION LOGIC
# ================================================================

print("\n" + "=" * 70)
print("CREATING INTEGRATED FORECAST")
print("=" * 70)


# Recommended daily forecast already selected
# by Phase 5.5.

integrated[
    "integrated_daily_forecast"
] = integrated[
    "recommended_daily_forecast"
]


# Recommended 30-day forecast

integrated[
    "integrated_30d_forecast"
] = integrated[
    "recommended_30d_forecast"
]


# ================================================================
# INVENTORY PROJECTION
# ================================================================

print("\nCalculating projected inventory...")


integrated[
    "stock_before_forecast"
] = integrated[
    "stock_on_hand"
]


integrated[
    "projected_stock_after_30d_integration"
] = (
    integrated["stock_on_hand"]
    -
    integrated["integrated_30d_forecast"]
)


# ================================================================
# SAFETY STOCK GAP
# ================================================================

integrated[
    "safety_stock_gap"
] = (
    integrated["safety_stock"]
    -
    integrated[
        "projected_stock_after_30d_integration"
    ]
)


# ================================================================
# REPLENISHMENT QUANTITY
# ================================================================

integrated[
    "integration_reorder_qty"
] = (
    integrated["safety_stock_gap"]
    .clip(lower=0)
)


# ================================================================
# TARGET STOCK
# ================================================================

integrated[
    "integration_target_stock"
] = (
    integrated["integrated_30d_forecast"]
    +
    integrated["safety_stock"]
)


# ================================================================
# STOCK COVER
# ================================================================

integrated[
    "forecast_daily_positive"
] = integrated[
    "integrated_daily_forecast"
].replace(
    0,
    np.nan
)


integrated[
    "forecast_days_of_cover"
] = (
    integrated["stock_on_hand"]
    /
    integrated["forecast_daily_positive"]
)


integrated[
    "forecast_days_of_cover"
] = integrated[
    "forecast_days_of_cover"
].replace(
    [np.inf, -np.inf],
    np.nan
)


integrated[
    "forecast_days_of_cover"
] = integrated[
    "forecast_days_of_cover"
].fillna(365)


# ================================================================
# INVENTORY STATUS
# ================================================================

def classify_inventory(row):

    stock = row["stock_on_hand"]

    safety = row["safety_stock"]

    forecast = row[
        "integrated_30d_forecast"
    ]

    if pd.isna(stock):

        return "Inventory Data Missing"

    if stock <= 0:

        return "Stockout"

    if stock < safety:

        return "Below Safety Stock"

    if forecast > 0:

        if stock < forecast:

            return "Potential Stockout"

        if stock > forecast * 6:

            return "Potential Overstock"

    return "Healthy"


integrated[
    "integrated_inventory_status"
] = integrated.apply(
    classify_inventory,
    axis=1
)


# ================================================================
# BUSINESS PRIORITY
# ================================================================

def calculate_priority(row):

    status = row[
        "integrated_inventory_status"
    ]

    action = str(
        row.get(
            "business_action",
            ""
        )
    )

    if status in [
        "Stockout",
        "Below Safety Stock",
        "Potential Stockout"
    ]:

        return "Critical"

    if (
        status == "Potential Overstock"
        or
        "inactive" in action.lower()
        or
        "excess" in action.lower()
    ):

        return "High"

    return "Normal"


integrated[
    "integration_business_priority"
] = integrated.apply(
    calculate_priority,
    axis=1
)


# ================================================================
# REPLENISHMENT DECISION
# ================================================================

def replenishment_decision(row):

    if row[
        "integrated_inventory_status"
    ] in [
        "Stockout",
        "Below Safety Stock",
        "Potential Stockout"
    ]:

        if row[
            "integration_reorder_qty"
        ] > 0:

            return "Replenish"

    return "No Replenishment"


integrated[
    "integration_replenishment_decision"
] = integrated.apply(
    replenishment_decision,
    axis=1
)


# ================================================================
# OVERSTOCK DECISION
# ================================================================

def overstock_decision(row):

    status = row[
        "integrated_inventory_status"
    ]

    if status == "Potential Overstock":

        return "Review Excess Inventory"

    if (
        "inactive"
        in str(
            row.get(
                "business_action",
                ""
            )
        ).lower()
    ):

        return "Review Inactive Inventory"

    return "No Overstock Action"


integrated[
    "integration_overstock_decision"
] = integrated.apply(
    overstock_decision,
    axis=1
)


# ================================================================
# FINAL CLEANING
# ================================================================

print("\n" + "=" * 70)
print("FINAL CLEANING")
print("=" * 70)


# Negative forecasts cannot occur
integrated[
    "integrated_daily_forecast"
] = integrated[
    "integrated_daily_forecast"
].clip(
    lower=0
)


integrated[
    "integrated_30d_forecast"
] = integrated[
    "integrated_30d_forecast"
].clip(
    lower=0
)


integrated[
    "integration_reorder_qty"
] = integrated[
    "integration_reorder_qty"
].clip(
    lower=0
)


# ================================================================
# VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    f"Final integration shape: {integrated.shape}"
)


print(
    "Missing store IDs:",
    integrated["store_id"].isna().sum()
)


print(
    "Missing SKU IDs:",
    integrated["sku_id"].isna().sum()
)


print(
    "Missing integrated forecasts:",
    integrated[
        "integrated_30d_forecast"
    ].isna().sum()
)


print(
    "Missing inventory:",
    integrated[
        "stock_on_hand"
    ].isna().sum()
)


print(
    "Negative forecasts:",
    (
        integrated[
            "integrated_30d_forecast"
        ] < 0
    ).sum()
)


print(
    "Negative reorder quantities:",
    (
        integrated[
            "integration_reorder_qty"
        ] < 0
    ).sum()
)


print(
    "\nInventory status distribution:"
)

print(
    integrated[
        "integrated_inventory_status"
    ].value_counts()
)


print(
    "\nBusiness priority distribution:"
)

print(
    integrated[
        "integration_business_priority"
    ].value_counts()
)


print(
    "\nReplenishment decision distribution:"
)

print(
    integrated[
        "integration_replenishment_decision"
    ].value_counts()
)


# ================================================================
# KEY BUSINESS METRICS
# ================================================================

print("\n" + "=" * 70)
print("KEY BUSINESS METRICS")
print("=" * 70)


replenishment_items = (
    integrated[
        "integration_replenishment_decision"
    ]
    == "Replenish"
).sum()


overstock_items = (
    integrated[
        "integrated_inventory_status"
    ]
    == "Potential Overstock"
).sum()


stockout_items = (
    integrated[
        "integrated_inventory_status"
    ]
    == "Stockout"
).sum()


critical_items = (
    integrated[
        "integration_business_priority"
    ]
    == "Critical"
).sum()


total_reorder_qty = integrated[
    "integration_reorder_qty"
].sum()


print(
    f"Items requiring replenishment: "
    f"{replenishment_items}"
)


print(
    f"Potential overstock items: "
    f"{overstock_items}"
)


print(
    f"Stockout items: "
    f"{stockout_items}"
)


print(
    f"Critical items: "
    f"{critical_items}"
)


print(
    f"Total integration reorder quantity: "
    f"{total_reorder_qty:.2f}"
)


# ================================================================
# TOP REPLENISHMENT ITEMS
# ================================================================

print("\n" + "=" * 70)
print("TOP 20 REPLENISHMENT ITEMS")
print("=" * 70)


replenishment_top = integrated[
    integrated[
        "integration_replenishment_decision"
    ]
    == "Replenish"
].copy()


replenishment_top = replenishment_top.sort_values(
    "integration_reorder_qty",
    ascending=False
)


print(
    replenishment_top[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "safety_stock",
            "integrated_30d_forecast",
            "integration_reorder_qty",
            "integrated_inventory_status",
            "integration_business_priority"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ================================================================
# TOP OVERSTOCK ITEMS
# ================================================================

print("\n" + "=" * 70)
print("TOP 20 OVERSTOCK ITEMS")
print("=" * 70)


overstock_top = integrated[
    integrated[
        "integrated_inventory_status"
    ]
    == "Potential Overstock"
].copy()


overstock_top = overstock_top.sort_values(
    "forecast_days_of_cover",
    ascending=False
)


print(
    overstock_top[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "integrated_30d_forecast",
            "forecast_days_of_cover",
            "demand_trend",
            "risk_category",
            "integration_business_priority"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ================================================================
# TOP CRITICAL ITEMS
# ================================================================

print("\n" + "=" * 70)
print("TOP 20 CRITICAL ITEMS")
print("=" * 70)


critical_top = integrated[
    integrated[
        "integration_business_priority"
    ]
    == "Critical"
].copy()


critical_top = critical_top.sort_values(
    "integration_reorder_qty",
    ascending=False
)


print(
    critical_top[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "safety_stock",
            "integrated_30d_forecast",
            "integration_reorder_qty",
            "integrated_inventory_status",
            "integration_business_priority"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ================================================================
# STORE SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("STORE-LEVEL INTEGRATION SUMMARY")
print("=" * 70)


store_summary = (
    integrated
    .groupby("store_id")
    .agg(
        total_sku=(
            "sku_id",
            "nunique"
        ),

        critical_items=(
            "integration_business_priority",
            lambda x:
            (x == "Critical").sum()
        ),

        replenishment_items=(
            "integration_replenishment_decision",
            lambda x:
            (x == "Replenish").sum()
        ),

        overstock_items=(
            "integrated_inventory_status",
            lambda x:
            (x == "Potential Overstock").sum()
        ),

        stockout_items=(
            "integrated_inventory_status",
            lambda x:
            (x == "Stockout").sum()
        ),

        total_reorder_quantity=(
            "integration_reorder_qty",
            "sum"
        ),

        total_stock=(
            "stock_on_hand",
            "sum"
        ),

        total_30d_forecast=(
            "integrated_30d_forecast",
            "sum"
        )
    )
    .reset_index()
)


store_summary[
    "store_stock_gap"
] = (
    store_summary["total_stock"]
    -
    store_summary["total_30d_forecast"]
)


print(
    store_summary.sort_values(
        "critical_items",
        ascending=False
    ).head(20).to_string(
        index=False
    )
)


# ================================================================
# SKU SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("SKU-LEVEL INTEGRATION SUMMARY")
print("=" * 70)


sku_summary = (
    integrated
    .groupby("sku_id")
    .agg(
        stores=(
            "store_id",
            "nunique"
        ),

        critical_items=(
            "integration_business_priority",
            lambda x:
            (x == "Critical").sum()
        ),

        replenishment_items=(
            "integration_replenishment_decision",
            lambda x:
            (x == "Replenish").sum()
        ),

        overstock_items=(
            "integrated_inventory_status",
            lambda x:
            (x == "Potential Overstock").sum()
        ),

        stockout_items=(
            "integrated_inventory_status",
            lambda x:
            (x == "Stockout").sum()
        ),

        total_reorder_quantity=(
            "integration_reorder_qty",
            "sum"
        ),

        total_stock=(
            "stock_on_hand",
            "sum"
        ),

        total_30d_forecast=(
            "integrated_30d_forecast",
            "sum"
        )
    )
    .reset_index()
)


sku_summary[
    "stock_forecast_gap"
] = (
    sku_summary["total_stock"]
    -
    sku_summary["total_30d_forecast"]
)


print(
    sku_summary.sort_values(
        "critical_items",
        ascending=False
    ).head(20).to_string(
        index=False
    )
)


# ================================================================
# SAVE MAIN OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("SAVING OUTPUT FILES")
print("=" * 70)


MAIN_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "forecast_inventory_integrated.csv"
)


REPLENISHMENT_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "integrated_replenishment_recommendations.csv"
)


OVERSTOCK_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "integrated_overstock_recommendations.csv"
)


CRITICAL_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "integrated_critical_items.csv"
)


STORE_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "integrated_store_summary.csv"
)


SKU_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "integrated_sku_summary.csv"
)


integrated.to_csv(
    MAIN_OUTPUT,
    index=False
)


replenishment_top.to_csv(
    REPLENISHMENT_OUTPUT,
    index=False
)


overstock_top.to_csv(
    OVERSTOCK_OUTPUT,
    index=False
)


critical_top.to_csv(
    CRITICAL_OUTPUT,
    index=False
)


store_summary.to_csv(
    STORE_OUTPUT,
    index=False
)


sku_summary.to_csv(
    SKU_OUTPUT,
    index=False
)


# ================================================================
# FINAL VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    "Main output rows:",
    len(integrated)
)


print(
    "Missing recommended model:",
    integrated[
        "recommended_model"
    ].isna().sum()
)


print(
    "Missing integrated forecast:",
    integrated[
        "integrated_30d_forecast"
    ].isna().sum()
)


print(
    "Missing inventory status:",
    integrated[
        "integrated_inventory_status"
    ].isna().sum()
)


print(
    "Missing business priority:",
    integrated[
        "integration_business_priority"
    ].isna().sum()
)


print(
    "Total reorder quantity:",
    integrated[
        "integration_reorder_qty"
    ].sum()
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)
print("PHASE 5.6 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nMain integrated output:")
print(MAIN_OUTPUT)


print("\nReplenishment output:")
print(REPLENISHMENT_OUTPUT)


print("\nOverstock output:")
print(OVERSTOCK_OUTPUT)


print("\nCritical items:")
print(CRITICAL_OUTPUT)


print("\nStore summary:")
print(STORE_OUTPUT)


print("\nSKU summary:")
print(SKU_OUTPUT)


print("\n" + "=" * 70)
print("NEXT PHASE: BUSINESS DASHBOARD / EXECUTIVE INSIGHTS")
print("=" * 70)