import pandas as pd
import numpy as np

print("=" * 70)
print("PROJECT FORESIGHT - INVENTORY BUSINESS LOGIC VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

inventory_path = "data/processed/inventory_clean.csv"
risk_path = (
    "data/processed/inventory_analysis/"
    "inventory_forecast_risk_analysis.csv"
)

inventory = pd.read_csv(inventory_path)
risk = pd.read_csv(risk_path)

print("\nInventory shape:", inventory.shape)
print("Risk analysis shape:", risk.shape)


# ============================================================
# 2. BASIC COLUMN CHECK
# ============================================================

print("\n" + "=" * 70)
print("REQUIRED COLUMN CHECK")
print("=" * 70)

inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

risk_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "forecast_30d_units",
    "forecast_daily_demand",
    "planning_daily_demand",
    "days_of_inventory",
    "forecast_coverage_ratio",
    "inventory_risk",
    "stockout_risk",
    "target_stock",
    "suggested_reorder_qty",
    "reorder_status"
]

for column in inventory_columns:

    if column in inventory.columns:
        print(f"PASS: Inventory column exists -> {column}")
    else:
        print(f"FAIL: Missing inventory column -> {column}")


for column in risk_columns:

    if column in risk.columns:
        print(f"PASS: Risk column exists -> {column}")
    else:
        print(f"FAIL: Missing risk column -> {column}")


# ============================================================
# 3. INVENTORY QUANTITY LOGIC
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY QUANTITY LOGIC")
print("=" * 70)

print(
    "\nNegative stock:",
    (inventory["stock_on_hand"] < 0).sum()
)

print(
    "Negative reorder point:",
    (inventory["reorder_point"] < 0).sum()
)

print(
    "Negative safety stock:",
    (inventory["safety_stock"] < 0).sum()
)

print(
    "Safety stock > reorder point:",
    (
        inventory["safety_stock"]
        > inventory["reorder_point"]
    ).sum()
)

print(
    "Stock below safety stock:",
    (
        inventory["stock_on_hand"]
        <= inventory["safety_stock"]
    ).sum()
)

print(
    "Stock below reorder point:",
    (
        inventory["stock_on_hand"]
        <= inventory["reorder_point"]
    ).sum()
)


# ============================================================
# 4. RISK DATA CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("RISK DATA CONSISTENCY")
print("=" * 70)

# Compare inventory and risk stock values

merged = inventory.merge(
    risk[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock"
        ]
    ],
    on=["store_id", "sku_id"],
    how="left",
    suffixes=("_inventory", "_risk")
)

print(
    "\nInventory rows:",
    len(inventory)
)

print(
    "Risk rows:",
    len(risk)
)

print(
    "Inventory rows missing from risk:",
    merged["stock_on_hand_risk"].isna().sum()
)


stock_mismatch = (
    merged["stock_on_hand_inventory"]
    != merged["stock_on_hand_risk"]
).sum()

reorder_mismatch = (
    merged["reorder_point_inventory"]
    != merged["reorder_point_risk"]
).sum()

safety_mismatch = (
    merged["safety_stock_inventory"]
    != merged["safety_stock_risk"]
).sum()

print(
    "\nStock-on-hand mismatches:",
    stock_mismatch
)

print(
    "Reorder-point mismatches:",
    reorder_mismatch
)

print(
    "Safety-stock mismatches:",
    safety_mismatch
)


# ============================================================
# 5. FORECAST DEMAND VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FORECAST DEMAND VALIDATION")
print("=" * 70)

print(
    "\nNegative forecast 30-day demand:",
    (
        risk["forecast_30d_units"] < 0
    ).sum()
)

print(
    "Negative forecast daily demand:",
    (
        risk["forecast_daily_demand"] < 0
    ).sum()
)

print(
    "Zero forecast daily demand:",
    (
        risk["forecast_daily_demand"] == 0
    ).sum()
)

print(
    "\nForecast 30-day demand statistics:"
)

print(
    risk["forecast_30d_units"].describe()
)

print(
    "\nForecast daily demand statistics:"
)

print(
    risk["forecast_daily_demand"].describe()
)


# ============================================================
# 6. CHECK 30-DAY FORECAST MATHEMATICS
# ============================================================

print("\n" + "=" * 70)
print("FORECAST MATHEMATICS")
print("=" * 70)

expected_daily_forecast = (
    risk["forecast_30d_units"] / 30
)

forecast_difference = (
    risk["forecast_daily_demand"]
    - expected_daily_forecast
).abs()

print(
    "\nMaximum difference between:"
)

print(
    "forecast_daily_demand"
)

print(
    "and forecast_30d_units / 30:"
)

print(
    forecast_difference.max()
)

forecast_math_errors = (
    forecast_difference > 0.01
).sum()

print(
    "\nRows with forecast calculation difference > 0.01:",
    forecast_math_errors
)


# ============================================================
# 7. PLANNING DEMAND VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PLANNING DEMAND VALIDATION")
print("=" * 70)

planning_difference = (
    risk["planning_daily_demand"]
    - risk["forecast_daily_demand"]
).abs()

print(
    "\nMaximum planning-demand difference:",
    planning_difference.max()
)

print(
    "Rows where planning demand differs from forecast demand:",
    (planning_difference > 0.01).sum()
)

print(
    "\nPlanning daily demand statistics:"
)

print(
    risk["planning_daily_demand"].describe()
)


# ============================================================
# 8. DAYS OF INVENTORY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DAYS OF INVENTORY VALIDATION")
print("=" * 70)

expected_days = (
    risk["stock_on_hand"]
    / risk["planning_daily_demand"]
)

# Avoid comparing infinite values
valid_days = (
    np.isfinite(expected_days)
    &
    np.isfinite(risk["days_of_inventory"])
)

days_difference = (
    risk.loc[valid_days, "days_of_inventory"]
    - expected_days[valid_days]
).abs()

print(
    "\nRows with valid demand:",
    valid_days.sum()
)

print(
    "Rows with zero planning demand:",
    (
        risk["planning_daily_demand"] == 0
    ).sum()
)

print(
    "Maximum days-of-inventory calculation difference:",
    days_difference.max()
)

print(
    "Rows with difference > 0.01:",
    (days_difference > 0.01).sum()
)


# ============================================================
# 9. FORECAST COVERAGE RATIO
# ============================================================

print("\n" + "=" * 70)
print("FORECAST COVERAGE VALIDATION")
print("=" * 70)

expected_coverage = (
    risk["stock_on_hand"]
    / risk["forecast_30d_units"]
)

valid_coverage = (
    np.isfinite(expected_coverage)
    &
    np.isfinite(risk["forecast_coverage_ratio"])
)

coverage_difference = (
    risk.loc[valid_coverage, "forecast_coverage_ratio"]
    - expected_coverage[valid_coverage]
).abs()

print(
    "\nMaximum coverage-ratio difference:",
    coverage_difference.max()
)

print(
    "Rows with difference > 0.01:",
    (coverage_difference > 0.01).sum()
)


# ============================================================
# 10. TARGET STOCK VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET STOCK VALIDATION")
print("=" * 70)

print(
    "\nTarget stock statistics:"
)

print(
    risk["target_stock"].describe()
)

target_below_safety = (
    risk["target_stock"]
    < risk["safety_stock"]
).sum()

print(
    "\nTarget stock below safety stock:",
    target_below_safety
)

target_below_forecast = (
    risk["target_stock"]
    < risk["forecast_30d_units"]
).sum()

print(
    "Target stock below forecast 30-day demand:",
    target_below_forecast
)


# ============================================================
# 11. REORDER QUANTITY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("REORDER QUANTITY VALIDATION")
print("=" * 70)

print(
    "\nNegative suggested reorder quantities:",
    (
        risk["suggested_reorder_qty"] < 0
    ).sum()
)

print(
    "Items with suggested reorder:",
    (
        risk["suggested_reorder_qty"] > 0
    ).sum()
)

print(
    "Total suggested reorder quantity:",
    risk["suggested_reorder_qty"].sum()
)


# Check logical relationship

reorder_logic_error = (
    (
        risk["suggested_reorder_qty"] > 0
    )
    &
    (
        risk["reorder_status"]
        == "Sufficient Stock"
    )
).sum()

print(
    "\nRows with reorder quantity > 0 but status = Sufficient Stock:",
    reorder_logic_error
)


# ============================================================
# 12. RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

print(
    "\nInventory risk:"
)

print(
    risk["inventory_risk"].value_counts()
)

print(
    "\nStockout risk:"
)

print(
    risk["stockout_risk"].value_counts()
)

print(
    "\nReorder status:"
)

print(
    risk["reorder_status"].value_counts()
)

print(
    "\nPriority:"
)

print(
    risk["priority"].value_counts()
)


# ============================================================
# 13. CHECK IF ALL ITEMS HAVE IDENTICAL RISK
# ============================================================

print("\n" + "=" * 70)
print("RISK VARIATION CHECK")
print("=" * 70)

risk_unique = risk["inventory_risk"].nunique()
stockout_unique = risk["stockout_risk"].nunique()
priority_unique = risk["priority"].nunique()

print(
    "\nUnique inventory risk levels:",
    risk_unique
)

print(
    "Unique stockout risk levels:",
    stockout_unique
)

print(
    "Unique priority levels:",
    priority_unique
)

if risk_unique == 1:
    print(
        "\nWARNING: All inventory items have the same risk category."
    )
else:
    print(
        "\nPASS: Inventory risk categories have variation."
    )


# ============================================================
# 14. STOCK / DEMAND RATIO
# ============================================================

print("\n" + "=" * 70)
print("STOCK VS FORECAST DEMAND")
print("=" * 70)

risk["stock_to_forecast_ratio"] = np.where(
    risk["forecast_30d_units"] > 0,
    risk["stock_on_hand"]
    / risk["forecast_30d_units"],
    np.inf
)

print(
    "\nStock-to-forecast ratio statistics:"
)

print(
    risk["stock_to_forecast_ratio"].replace(
        np.inf,
        np.nan
    ).describe()
)


# ============================================================
# 15. EXTREME INVENTORY LEVELS
# ============================================================

print("\n" + "=" * 70)
print("EXTREME INVENTORY LEVELS")
print("=" * 70)

print(
    "\nTop 10 highest stock-to-forecast ratios:"
)

print(
    risk[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "forecast_30d_units",
            "stock_to_forecast_ratio",
            "inventory_risk"
        ]
    ]
    .sort_values(
        "stock_to_forecast_ratio",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 16. FINAL BUSINESS LOGIC SCORECARD
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS LOGIC SCORECARD")
print("=" * 70)

checks = {

    "No negative stock":
        (
            inventory["stock_on_hand"] >= 0
        ).all(),

    "No negative reorder point":
        (
            inventory["reorder_point"] >= 0
        ).all(),

    "No negative safety stock":
        (
            inventory["safety_stock"] >= 0
        ).all(),

    "Safety stock <= reorder point":
        (
            inventory["safety_stock"]
            <= inventory["reorder_point"]
        ).all(),

    "Inventory stock values preserved":
        stock_mismatch == 0,

    "Reorder points preserved":
        reorder_mismatch == 0,

    "Safety stock values preserved":
        safety_mismatch == 0,

    "No negative forecast":
        (
            risk["forecast_30d_units"] >= 0
        ).all(),

    "Forecast daily demand mathematically consistent":
        forecast_math_errors == 0,

    "Days of inventory mathematically consistent":
        (
            (days_difference <= 0.01).all()
            if len(days_difference) > 0
            else True
        ),

    "No negative reorder quantity":
        (
            risk["suggested_reorder_qty"] >= 0
        ).all(),

    "No reorder/status contradiction":
        reorder_logic_error == 0
}


passed = 0

for check, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{status}: {check}"
    )

    if result:
        passed += 1


print(
    "\nBusiness logic validation score:",
    passed,
    "/",
    len(checks)
)


# ============================================================
# 17. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if passed == len(checks):

    print(
        "\nPASS: Core inventory business logic is internally consistent."
    )

else:

    print(
        "\nWARNING: One or more business logic checks failed."
    )

    print(
        "Review the failed checks before finalizing the inventory model."
    )


print("\n" + "=" * 70)
print("BUSINESS LOGIC VALIDATION COMPLETED")
print("=" * 70)