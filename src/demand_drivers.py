# Demand Driver Analysis
import pandas as pd
import numpy as np
import os

# ============================================================
# PROJECT FORESIGHT - DEMAND DRIVER ANALYSIS
# ============================================================

PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

print("=" * 70)
print("PROJECT FORESIGHT - DEMAND DRIVER ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading analytics dataset...")

df = pd.read_csv(
    os.path.join(PROCESSED_PATH, "analytics_dataset.csv")
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()

df["quarter"] = df["date"].dt.quarter

df["day_of_week"] = df["date"].dt.dayofweek

df["day_name"] = df["date"].dt.day_name()

# Weekend flag
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

print("\nTime features created.")

# ============================================================
# 3. OVERALL PROMOTION IMPACT
# ============================================================

print("\n" + "=" * 70)
print("PROMOTION IMPACT")
print("=" * 70)

promotion_analysis = (
    df.groupby("promotion_flag")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean"),
        records=("sku_id", "count")
    )
    .reset_index()
)

promotion_analysis["gross_margin_pct"] = (
    promotion_analysis["gross_profit"]
    / promotion_analysis["revenue"]
    * 100
)

print(promotion_analysis)

# ============================================================
# 4. DISCOUNT BAND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DISCOUNT BAND ANALYSIS")
print("=" * 70)

df["discount_band"] = pd.cut(
    df["avg_discount"],
    bins=[-0.01, 0, 5, 10, 15, 20, 25, 30, 35],
    labels=[
        "0%",
        "1-5%",
        "6-10%",
        "11-15%",
        "16-20%",
        "21-25%",
        "26-30%",
        "31-35%"
    ]
)

discount_analysis = (
    df.groupby("discount_band", observed=False)
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_units=("units_sold", "mean"),
        avg_revenue=("revenue", "mean"),
        records=("sku_id", "count")
    )
    .reset_index()
)

discount_analysis["gross_margin_pct"] = (
    discount_analysis["gross_profit"]
    / discount_analysis["revenue"]
    * 100
)

print(discount_analysis)

# ============================================================
# 5. CATEGORY × PROMOTION
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY × PROMOTION")
print("=" * 70)

category_promotion = (
    df.groupby(
        ["category", "promotion_flag"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .reset_index()
)

category_promotion["gross_margin_pct"] = (
    category_promotion["gross_profit"]
    / category_promotion["revenue"]
    * 100
)

print(category_promotion)

# ============================================================
# 6. CATEGORY DEMAND LIFT FROM PROMOTIONS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY PROMOTION DEMAND LIFT")
print("=" * 70)

pivot = category_promotion.pivot(
    index="category",
    columns="promotion_flag",
    values="units_sold"
)

pivot = pivot.rename(
    columns={
        0: "non_promo_units",
        1: "promo_units"
    }
)

pivot["demand_lift_pct"] = (
    (
        pivot["promo_units"]
        - pivot["non_promo_units"]
    )
    / pivot["non_promo_units"]
    * 100
)

print(
    pivot.sort_values(
        "demand_lift_pct",
        ascending=False
    )
)

# ============================================================
# 7. STORE TYPE × PROMOTION
# ============================================================

print("\n" + "=" * 70)
print("STORE TYPE × PROMOTION")
print("=" * 70)

store_promotion = (
    df.groupby(
        ["store_type", "promotion_flag"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .reset_index()
)

store_promotion["gross_margin_pct"] = (
    store_promotion["gross_profit"]
    / store_promotion["revenue"]
    * 100
)

print(store_promotion)

# ============================================================
# 8. MONTH × PROMOTION
# ============================================================

print("\n" + "=" * 70)
print("MONTH × PROMOTION")
print("=" * 70)

month_promotion = (
    df.groupby(
        ["month", "month_name", "promotion_flag"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

month_promotion["gross_margin_pct"] = (
    month_promotion["gross_profit"]
    / month_promotion["revenue"]
    * 100
)

print(month_promotion)

# ============================================================
# 9. WEEKDAY × PROMOTION
# ============================================================

print("\n" + "=" * 70)
print("DAY-OF-WEEK × PROMOTION")
print("=" * 70)

weekday_promotion = (
    df.groupby(
        [
            "day_of_week",
            "day_name",
            "promotion_flag"
        ]
    )
    .agg(
        avg_units=("units_sold", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_profit=("gross_profit", "mean")
    )
    .reset_index()
)

print(weekday_promotion)

# ============================================================
# 10. CATEGORY × DISCOUNT
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY × DISCOUNT")
print("=" * 70)

category_discount = (
    df.groupby("category")
    .agg(
        avg_discount=("avg_discount", "mean"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

category_discount["gross_margin_pct"] = (
    category_discount["gross_profit"]
    / category_discount["revenue"]
    * 100
)

print(category_discount)

# ============================================================
# 11. PRICE / REVENUE RELATIONSHIP
# ============================================================

print("\n" + "=" * 70)
print("PRICE / DEMAND RELATIONSHIP")
print("=" * 70)

df["estimated_unit_price"] = (
    df["revenue"]
    / df["units_sold"]
)

price_demand = (
    df.groupby("category")
    .agg(
        avg_unit_price=("estimated_unit_price", "mean"),
        avg_units_sold=("units_sold", "mean"),
        total_units=("units_sold", "sum"),
        total_revenue=("revenue", "sum")
    )
    .reset_index()
)

print(price_demand)

# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL CORRELATION")
print("=" * 70)

correlation_columns = [
    "units_sold",
    "revenue",
    "avg_discount",
    "cost_price",
    "gross_profit",
    "promotion_flag",
    "month",
    "day_of_week",
    "is_weekend"
]

correlation_matrix = df[
    correlation_columns
].corr()

print(correlation_matrix)

# ============================================================
# 13. HIGH-DISCOUNT LOW-MARGIN CASES
# ============================================================

print("\n" + "=" * 70)
print("HIGH DISCOUNT / LOW MARGIN CASES")
print("=" * 70)

high_discount_risk = (
    df[
        df["avg_discount"] >= 25
    ]
    .groupby("category")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .reset_index()
)

high_discount_risk["gross_margin_pct"] = (
    high_discount_risk["gross_profit"]
    / high_discount_risk["revenue"]
    * 100
)

print(
    high_discount_risk.sort_values(
        "gross_margin_pct"
    )
)

# ============================================================
# 14. SAVE ANALYSIS TABLES
# ============================================================

output_path = os.path.join(
    PROCESSED_PATH,
    "demand_driver_analysis"
)

os.makedirs(
    output_path,
    exist_ok=True
)

promotion_analysis.to_csv(
    os.path.join(
        output_path,
        "promotion_analysis.csv"
    ),
    index=False
)

discount_analysis.to_csv(
    os.path.join(
        output_path,
        "discount_analysis.csv"
    ),
    index=False
)

category_promotion.to_csv(
    os.path.join(
        output_path,
        "category_promotion.csv"
    ),
    index=False
)

store_promotion.to_csv(
    os.path.join(
        output_path,
        "store_promotion.csv"
    ),
    index=False
)

month_promotion.to_csv(
    os.path.join(
        output_path,
        "month_promotion.csv"
    ),
    index=False
)

weekday_promotion.to_csv(
    os.path.join(
        output_path,
        "weekday_promotion.csv"
    ),
    index=False
)

category_discount.to_csv(
    os.path.join(
        output_path,
        "category_discount.csv"
    ),
    index=False
)

high_discount_risk.to_csv(
    os.path.join(
        output_path,
        "high_discount_risk.csv"
    ),
    index=False
)

correlation_matrix.to_csv(
    os.path.join(
        output_path,
        "correlation_matrix.csv"
    )
)

print("\n" + "=" * 70)
print("DEMAND DRIVER ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nAnalysis files saved to:"
)

print(output_path)

# Demand Forecasting Preparation
