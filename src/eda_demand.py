import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# PROJECT FORESIGHT - PHASE 6
# ADVANCED DEMAND EDA
# ============================================================

PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

# ============================================================
# 1. LOAD ANALYTICS DATASET
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - ADVANCED DEMAND EDA")
print("=" * 70)

print("\nLoading analytics dataset...")

df = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "analytics_dataset.csv"
    )
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("Dataset loaded successfully!")

print("\nShape:")
print(df.shape)

# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.day_name()

print("\nTime features created.")

# ============================================================
# 3. OVERALL DATE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("DATE COVERAGE")
print("=" * 70)

print("Start Date :", df["date"].min())
print("End Date   :", df["date"].max())
print("Days       :", df["date"].nunique())

print("\nRecords by year:")

print(
    df.groupby("year")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        records=("date", "count")
    )
)

# ============================================================
# 4. FAIR YEAR-OVER-YEAR COMPARISON
# ============================================================
# 2025 only contains Jan-Oct.
# Therefore compare Jan-Oct for every year.

print("\n" + "=" * 70)
print("JAN-OCT YEAR-OVER-YEAR COMPARISON")
print("=" * 70)

ytd = df[df["month"] <= 10]

yoy = (
    ytd.groupby("year")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

yoy["revenue_growth_pct"] = (
    yoy["revenue"]
    .pct_change()
    * 100
)

yoy["units_growth_pct"] = (
    yoy["units_sold"]
    .pct_change()
    * 100
)

print(yoy)

# ============================================================
# 5. MONTHLY DEMAND TREND
# ============================================================

print("\n" + "=" * 70)
print("MONTHLY DEMAND TREND")
print("=" * 70)

monthly = (
    df.groupby(
        pd.Grouper(key="date", freq="ME")
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print("\nFirst 12 months:")
print(monthly.head(12))

print("\nLast 12 months:")
print(monthly.tail(12))

# ============================================================
# 6. MONTH-OF-YEAR SEASONALITY
# ============================================================

print("\n" + "=" * 70)
print("MONTH-OF-YEAR SEASONALITY")
print("=" * 70)

seasonality = (
    df.groupby("month")
    .agg(
        avg_units=("units_sold", "mean"),
        total_units=("units_sold", "sum"),
        avg_revenue=("revenue", "mean"),
        total_revenue=("revenue", "sum"),
        avg_profit=("gross_profit", "mean"),
        total_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print(seasonality)

# Add month names
seasonality["month_name"] = (
    pd.to_datetime(
        seasonality["month"],
        format="%m"
    ).dt.month_name()
)

seasonality = seasonality[
    [
        "month",
        "month_name",
        "avg_units",
        "total_units",
        "avg_revenue",
        "total_revenue",
        "avg_profit",
        "total_profit"
    ]
]

print("\nSeasonality table:")
print(seasonality)

# ============================================================
# 7. QUARTERLY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("QUARTERLY PERFORMANCE")
print("=" * 70)

quarterly = (
    df.groupby(
        ["year", "quarter"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print(quarterly)

# ============================================================
# 8. DAY-OF-WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DAY-OF-WEEK ANALYSIS")
print("=" * 70)

dow = (
    df.groupby(
        ["day_of_week", "day_name"]
    )
    .agg(
        avg_units=("units_sold", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_profit=("gross_profit", "mean")
    )
    .reset_index()
    .sort_values("day_of_week")
)

print(dow)

# ============================================================
# 9. CATEGORY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

category = (
    df.groupby("category")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .reset_index()
)

category["gross_margin_pct"] = (
    category["gross_profit"]
    / category["revenue"]
    * 100
)

category = category.sort_values(
    "revenue",
    ascending=False
)

print(category)

# ============================================================
# 10. CATEGORY YEARLY TREND
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY YEARLY TREND")
print("=" * 70)

category_year = (
    df.groupby(
        ["year", "category"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print(category_year)

# ============================================================
# 11. STORE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("STORE PERFORMANCE")
print("=" * 70)

store = (
    df.groupby(
        ["store_id", "store_name", "store_city", "store_type"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

store["gross_margin_pct"] = (
    store["gross_profit"]
    / store["revenue"]
    * 100
)

store = store.sort_values(
    "revenue",
    ascending=False
)

print("\nTop 10 Stores:")
print(store.head(10))

print("\nBottom 10 Stores:")
print(
    store.tail(10)
)

# ============================================================
# 12. STORE TYPE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("STORE TYPE PERFORMANCE")
print("=" * 70)

store_type = (
    df.groupby("store_type")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

store_type["gross_margin_pct"] = (
    store_type["gross_profit"]
    / store_type["revenue"]
    * 100
)

print(store_type)

# ============================================================
# 13. PROMOTION VS NON-PROMOTION
# ============================================================

print("\n" + "=" * 70)
print("PROMOTION VS NON-PROMOTION")
print("=" * 70)

promotion_analysis = (
    df.groupby("promotion_flag")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean"),
        records=("promotion_flag", "count")
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
# 14. PROMOTION NAME PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("PROMOTION PERFORMANCE")
print("=" * 70)

promotion_name = (
    df[df["promotion_flag"] == 1]
    .groupby("promo_name")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .reset_index()
)

promotion_name["gross_margin_pct"] = (
    promotion_name["gross_profit"]
    / promotion_name["revenue"]
    * 100
)

promotion_name = promotion_name.sort_values(
    "revenue",
    ascending=False
)

print("\nTop promotions by revenue:")
print(
    promotion_name.head(15)
)

# ============================================================
# 15. DISCOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DISCOUNT ANALYSIS")
print("=" * 70)

discount_analysis = (
    df.groupby("avg_discount")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
    .sort_values("avg_discount")
)

discount_analysis["gross_margin_pct"] = (
    discount_analysis["gross_profit"]
    / discount_analysis["revenue"]
    * 100
)

print(discount_analysis)

# ============================================================
# 16. PROFITABILITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PROFITABILITY ANALYSIS")
print("=" * 70)

profitability = (
    df.groupby(
        ["sku_id", "sku_name", "category"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

profitability["gross_margin_pct"] = (
    profitability["gross_profit"]
    / profitability["revenue"]
    * 100
)

print("\nTop 10 SKUs by Gross Profit:")
print(
    profitability
    .sort_values("gross_profit", ascending=False)
    .head(10)
)

print("\nBottom 10 SKUs by Gross Profit:")
print(
    profitability
    .sort_values("gross_profit", ascending=True)
    .head(10)
)

# ============================================================
# 17. LOSS-MAKING SKU ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LOSS-MAKING SKU ANALYSIS")
print("=" * 70)

loss_making = profitability[
    profitability["gross_profit"] < 0
].sort_values(
    "gross_profit"
)

print("Number of loss-making SKUs:")
print(loss_making.shape[0])

print("\nLoss-making SKUs:")
print(loss_making.head(20))

# ============================================================
# 18. HIGH REVENUE / LOW PROFIT SKUS
# ============================================================

print("\n" + "=" * 70)
print("HIGH REVENUE / LOW PROFIT ANALYSIS")
print("=" * 70)

revenue_threshold = profitability["revenue"].quantile(0.75)
profit_threshold = profitability["gross_profit"].quantile(0.25)

high_revenue_low_profit = profitability[
    (profitability["revenue"] >= revenue_threshold) &
    (profitability["gross_profit"] <= profit_threshold)
].sort_values(
    "revenue",
    ascending=False
)

print(high_revenue_low_profit)

# ============================================================
# 19. DEMAND VOLATILITY BY SKU
# ============================================================

print("\n" + "=" * 70)
print("DEMAND VOLATILITY BY SKU")
print("=" * 70)

sku_demand = (
    df.groupby("sku_id")
    .agg(
        avg_daily_demand=("units_sold", "mean"),
        demand_std=("units_sold", "std"),
        total_units=("units_sold", "sum")
    )
    .reset_index()
)

sku_demand["coefficient_of_variation"] = (
    sku_demand["demand_std"]
    / sku_demand["avg_daily_demand"]
)

sku_demand = sku_demand.replace(
    [np.inf, -np.inf],
    np.nan
)

print("\nMost volatile SKUs:")

print(
    sku_demand
    .sort_values(
        "coefficient_of_variation",
        ascending=False
    )
    .head(15)
)

# ============================================================
# 20. DEMAND VOLATILITY BY CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("DEMAND VOLATILITY BY CATEGORY")
print("=" * 70)

category_volatility = (
    df.groupby("category")
    .agg(
        avg_daily_demand=("units_sold", "mean"),
        demand_std=("units_sold", "std")
    )
    .reset_index()
)

category_volatility["coefficient_of_variation"] = (
    category_volatility["demand_std"]
    / category_volatility["avg_daily_demand"]
)

print(category_volatility)

# ============================================================
# 21. DAILY DEMAND TREND PLOT
# ============================================================

daily = (
    df.groupby("date")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    daily["date"],
    daily["units_sold"]
)

plt.title("Daily Demand Trend")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# ============================================================
# 22. MONTHLY REVENUE TREND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly["date"],
    monthly["revenue"]
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# ============================================================
# 23. MONTH-OF-YEAR SEASONALITY PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    seasonality["month_name"],
    seasonality["avg_units"],
    marker="o"
)

plt.title("Average Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# ============================================================
# 24. CATEGORY REVENUE PLOT
# ============================================================

category_plot = category.sort_values(
    "revenue",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    category_plot["category"],
    category_plot["revenue"]
)

plt.title("Revenue by Category")
plt.xlabel("Revenue")
plt.ylabel("Category")
plt.tight_layout()

plt.show()

# ============================================================
# 25. SAVE EDA OUTPUT TABLES
# ============================================================

EDA_PATH = os.path.join(
    PROCESSED_PATH,
    "eda_outputs"
)

os.makedirs(
    EDA_PATH,
    exist_ok=True
)

monthly.to_csv(
    os.path.join(EDA_PATH, "monthly_performance.csv"),
    index=False
)

seasonality.to_csv(
    os.path.join(EDA_PATH, "seasonality.csv"),
    index=False
)

yoy.to_csv(
    os.path.join(EDA_PATH, "yoy_comparison.csv"),
    index=False
)

category.to_csv(
    os.path.join(EDA_PATH, "category_performance.csv"),
    index=False
)

category_year.to_csv(
    os.path.join(EDA_PATH, "category_yearly.csv"),
    index=False
)

store.to_csv(
    os.path.join(EDA_PATH, "store_performance.csv"),
    index=False
)

promotion_analysis.to_csv(
    os.path.join(EDA_PATH, "promotion_analysis.csv"),
    index=False
)

promotion_name.to_csv(
    os.path.join(EDA_PATH, "promotion_performance.csv"),
    index=False
)

discount_analysis.to_csv(
    os.path.join(EDA_PATH, "discount_analysis.csv"),
    index=False
)

profitability.to_csv(
    os.path.join(EDA_PATH, "sku_profitability.csv"),
    index=False
)

sku_demand.to_csv(
    os.path.join(EDA_PATH, "sku_demand_volatility.csv"),
    index=False
)

category_volatility.to_csv(
    os.path.join(EDA_PATH, "category_demand_volatility.csv"),
    index=False
)

print("\n" + "=" * 70)
print("PHASE 6 COMPLETE")
print("=" * 70)

print("\nEDA outputs saved to:")
print(EDA_PATH)

print("\nReady for Phase 7: Feature Engineering.")