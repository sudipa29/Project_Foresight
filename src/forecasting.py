import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("PROJECT FORESIGHT - FORECASTING DATA PREPARATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading analytics dataset...")

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "analytics_dataset.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. SORT DATA
# ============================================================

df = df.sort_values(["date", "store_id", "sku_id"])

print("\nData sorted by date, store and SKU.")


# ============================================================
# 3. DAILY TOTAL DEMAND
# ============================================================

daily = (
    df.groupby("date")
      .agg(
          units_sold=("units_sold", "sum"),
          revenue=("revenue", "sum"),
          gross_profit=("gross_profit", "sum"),
          avg_discount=("avg_discount", "mean"),
          promotion_flag=("promotion_flag", "max")
      )
      .reset_index()
)

print("\n" + "=" * 70)
print("DAILY DEMAND DATA")
print("=" * 70)

print(daily.head())
print("\nShape:", daily.shape)


# ============================================================
# 4. CHECK DATE CONTINUITY
# ============================================================

print("\n" + "=" * 70)
print("DATE CONTINUITY CHECK")
print("=" * 70)

full_dates = pd.date_range(
    start=daily["date"].min(),
    end=daily["date"].max(),
    freq="D"
)

missing_dates = full_dates.difference(daily["date"])

print("Expected dates :", len(full_dates))
print("Actual dates   :", len(daily))
print("Missing dates  :", len(missing_dates))

if len(missing_dates) > 0:
    print("\nMissing dates:")
    print(missing_dates[:20])
else:
    print("No missing dates found.")


# ============================================================
# 5. REINDEX DAILY DATA
# ============================================================

daily = (
    daily.set_index("date")
         .reindex(full_dates)
         .rename_axis("date")
         .reset_index()
)

# Demand on missing dates should be treated carefully.
# We keep missing demand as NaN initially.
print("\nDaily dataset after date reindexing:")
print(daily.head())


# ============================================================
# 6. CALENDAR FEATURES
# ============================================================

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["quarter"] = daily["date"].dt.quarter
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["day_of_month"] = daily["date"].dt.day
daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)


# ============================================================
# 7. LAG FEATURES
# ============================================================

daily["lag_1"] = daily["units_sold"].shift(1)
daily["lag_7"] = daily["units_sold"].shift(7)
daily["lag_14"] = daily["units_sold"].shift(14)
daily["lag_28"] = daily["units_sold"].shift(28)


# ============================================================
# 8. ROLLING DEMAND FEATURES
# ============================================================

daily["rolling_mean_7"] = (
    daily["units_sold"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["rolling_mean_14"] = (
    daily["units_sold"]
    .shift(1)
    .rolling(14)
    .mean()
)

daily["rolling_mean_28"] = (
    daily["units_sold"]
    .shift(1)
    .rolling(28)
    .mean()
)

daily["rolling_std_7"] = (
    daily["units_sold"]
    .shift(1)
    .rolling(7)
    .std()
)

daily["rolling_std_28"] = (
    daily["units_sold"]
    .shift(1)
    .rolling(28)
    .std()
)


# ============================================================
# 9. DEMAND TREND
# ============================================================

daily["demand_change_7"] = (
    daily["units_sold"] -
    daily["lag_7"]
)

daily["demand_growth_7_pct"] = (
    daily["demand_change_7"] /
    daily["lag_7"].replace(0, np.nan)
) * 100


# ============================================================
# 10. SAVE FORECASTING DATASET
# ============================================================

OUTPUT_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "forecasting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "daily_forecasting_dataset.csv"
)

daily.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FORECASTING DATASET SUMMARY")
print("=" * 70)

print("Rows:", len(daily))
print("Columns:", len(daily.columns))

print("\nDate range:")
print("Start:", daily["date"].min())
print("End  :", daily["date"].max())

print("\nForecasting columns:")
print(daily.columns.tolist())

print("\nMissing values:")
print(
    daily[
        [
            "units_sold",
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28"
        ]
    ].isna().sum()
)

print("\nSample forecasting dataset:")
print(daily.tail(10))

print("\n" + "=" * 70)
print("FORECASTING DATA PREPARATION COMPLETED")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_PATH)