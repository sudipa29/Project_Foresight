import pandas as pd
import numpy as np
import os

PROCESSED_PATH = r"E:\Zidio_Development_Internship\Project_Foresight\data\processed"

# Load analytics dataset
df = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "analytics_dataset.csv"
    )
)

# Convert date
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("Analytics dataset loaded successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# Data Types
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# Missing Values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isna().sum())

# Duplicate Rows
print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())

# Descriptive stats
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(
    df[
        [
            "units_sold",
            "revenue",
            "avg_discount",
            "cost_price",
            "gross_profit"
        ]
    ].describe()
)

# Create Business KPI
total_units = df["units_sold"].sum()

total_revenue = df["revenue"].sum()

total_gross_profit = df["gross_profit"].sum()

average_daily_units = (
    df.groupby("date")["units_sold"]
    .sum()
    .mean()
)

number_of_skus = df["sku_id"].nunique()

number_of_stores = df["store_id"].nunique()

number_of_categories = df["category"].nunique()

print("\n" + "=" * 60)
print("KEY BUSINESS KPIs")
print("=" * 60)

print(f"Total Units Sold       : {total_units:,.0f}")
print(f"Total Revenue          : {total_revenue:,.2f}")
print(f"Total Gross Profit     : {total_gross_profit:,.2f}")
print(f"Average Daily Units    : {average_daily_units:,.2f}")
print(f"Number of SKUs         : {number_of_skus}")
print(f"Number of Stores       : {number_of_stores}")
print(f"Number of Categories   : {number_of_categories}")

# Daily Sales Trend
daily_sales = (
    df.groupby("date", as_index=False)
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
)

print("\nDaily sales shape:")
print(daily_sales.shape)

print("\nDaily sales:")
print(daily_sales.head())

# Monthly Sale
monthly_sales = (
    df.set_index("date")
    .resample("ME")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print("\nMonthly Sales:")
print(monthly_sales.head(12))

# Calendar Feature
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.day_name()
df["day_of_week_num"] = df["date"].dt.dayofweek
df["week"] = df["date"].dt.isocalendar().week.astype(int)

# Yearly Performance
yearly_sales = (
    df.groupby("year")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .reset_index()
)

print("\nYearly Performance:")
print(yearly_sales)

# Category Performance
category_performance = (
    df.groupby("category")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .sort_values("revenue", ascending=False)
)

print("\nCategory Performance:")
print(category_performance)

# Top 10 SKUS
top_skus = (
    df.groupby(
        ["sku_id", "sku_name"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .sort_values(
        "units_sold",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 SKUs:")
print(top_skus)

# Low Demand / Dead Stock Candidates
bottom_skus = (
    df.groupby(
        ["sku_id", "sku_name"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "units_sold",
        ascending=True
    )
    .head(10)
)

print("\nBottom 10 SKUs:")
print(bottom_skus)

# Promotion vs No Promotion
promotion_performance = (
    df.groupby("promotion_flag")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
)

print("\nPromotion vs Non-Promotion:")
print(promotion_performance)

# Promotion-Level Performance
promotion_performance_detail = (
    df[df["promotion_flag"] == 1]
    .groupby("promo_name")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        avg_discount=("avg_discount", "mean")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\nPromotion Performance:")
print(
    promotion_performance_detail.head(20)
)

# Store Performance
store_performance = (
    df.groupby(
        ["store_id", "store_name", "store_city"]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
)

print("\nTop Stores:")
print(store_performance.head(10))

# 