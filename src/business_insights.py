# ============================================================
# PROJECT FORESIGHT
# Phase 6.2 - Business Insights
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

FORECAST_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
)

EVALUATION_PATH = FORECAST_PATH / "evaluation"

OUTPUT_PATH = FORECAST_PATH / "business_insights"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def save_csv(df, filename):
    path = OUTPUT_PATH / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


# ============================================================
# START
# ============================================================

section("PROJECT FORESIGHT - PHASE 6.2")
print("BUSINESS INSIGHTS & DEMAND PRIORITIZATION")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

section("CHECKING REQUIRED FILES")

required_files = {
    "30D Forecast":
        FORECAST_PATH / "future_30_day_forecast.csv",

    "60D Forecast":
        FORECAST_PATH / "future_60_day_forecast.csv",

    "90D Forecast":
        FORECAST_PATH / "future_90_day_forecast.csv",

    "Store Evaluation":
        EVALUATION_PATH / "store_forecast_evaluation.csv",

    "SKU Evaluation":
        EVALUATION_PATH / "sku_forecast_evaluation.csv",

    "Store-SKU Evaluation":
        EVALUATION_PATH / "store_sku_forecast_evaluation.csv",

    "Daily Forecast":
        EVALUATION_PATH / "future_daily_forecast_summary.csv",
}


for name, path in required_files.items():

    if path.exists():
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}")
        print(f"Missing: {path}")
        raise FileNotFoundError(path)


# ============================================================
# LOAD 30-DAY FORECAST
# ============================================================

section("LOADING 30-DAY FORECAST")

forecast_30 = pd.read_csv(
    required_files["30D Forecast"]
)

forecast_30["date"] = pd.to_datetime(
    forecast_30["date"]
)

forecast_30["forecast_units"] = pd.to_numeric(
    forecast_30["forecast_units"],
    errors="coerce"
).fillna(0)

print("Rows:", len(forecast_30))
print("Columns:", forecast_30.columns.tolist())


# ============================================================
# LOAD 60-DAY FORECAST
# ============================================================

section("LOADING 60-DAY FORECAST")

forecast_60 = pd.read_csv(
    required_files["60D Forecast"]
)

forecast_60["date"] = pd.to_datetime(
    forecast_60["date"]
)

forecast_60["forecast_units"] = pd.to_numeric(
    forecast_60["forecast_units"],
    errors="coerce"
).fillna(0)

print("Rows:", len(forecast_60))


# ============================================================
# LOAD 90-DAY FORECAST
# ============================================================

section("LOADING 90-DAY FORECAST")

forecast_90 = pd.read_csv(
    required_files["90D Forecast"]
)

forecast_90["date"] = pd.to_datetime(
    forecast_90["date"]
)

forecast_90["forecast_units"] = pd.to_numeric(
    forecast_90["forecast_units"],
    errors="coerce"
).fillna(0)

print("Rows:", len(forecast_90))


# ============================================================
# LOAD EVALUATION FILES
# ============================================================

section("LOADING EVALUATION DATA")

store_eval = pd.read_csv(
    required_files["Store Evaluation"]
)

sku_eval = pd.read_csv(
    required_files["SKU Evaluation"]
)

store_sku_eval = pd.read_csv(
    required_files["Store-SKU Evaluation"]
)

daily_eval = pd.read_csv(
    required_files["Daily Forecast"]
)

print("Store evaluation rows:", len(store_eval))
print("SKU evaluation rows:", len(sku_eval))
print("Store-SKU evaluation rows:", len(store_sku_eval))
print("Daily evaluation rows:", len(daily_eval))


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print()
print("Store evaluation columns:")
print(store_eval.columns.tolist())

print()
print("SKU evaluation columns:")
print(sku_eval.columns.tolist())

print()
print("Store-SKU evaluation columns:")
print(store_sku_eval.columns.tolist())


# ============================================================
# STANDARDIZE NUMERIC COLUMNS
# ============================================================

for df in [store_eval, sku_eval, store_sku_eval]:

    for col in df.columns:

        if (
            "forecast" in col.lower()
            or "demand" in col.lower()
            or "units" in col.lower()
            or "total" in col.lower()
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


# ============================================================
# 30-DAY STORE DEMAND
# ============================================================

section("CREATING STORE-LEVEL BUSINESS INSIGHTS")

store_30 = (
    forecast_30
    .groupby("store_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_30d_units"
        }
    )
)


# ============================================================
# 60-DAY STORE DEMAND
# ============================================================

store_60 = (
    forecast_60
    .groupby("store_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_60d_units"
        }
    )
)


# ============================================================
# 90-DAY STORE DEMAND
# ============================================================

store_90 = (
    forecast_90
    .groupby("store_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_90d_units"
        }
    )
)


# ============================================================
# MERGE STORE FORECASTS
# ============================================================

store_insights = store_30.merge(
    store_60,
    on="store_id",
    how="outer"
)

store_insights = store_insights.merge(
    store_90,
    on="store_id",
    how="outer"
)

store_insights = store_insights.fillna(0)


# ============================================================
# STORE GROWTH
# ============================================================

store_insights["growth_30_to_60_pct"] = np.where(
    store_insights["forecast_30d_units"] > 0,
    (
        (
            store_insights["forecast_60d_units"]
            / store_insights["forecast_30d_units"]
        ) - 1
    ) * 100,
    0
)

store_insights["growth_60_to_90_pct"] = np.where(
    store_insights["forecast_60d_units"] > 0,
    (
        (
            store_insights["forecast_90d_units"]
            / store_insights["forecast_60d_units"]
        ) - 1
    ) * 100,
    0
)


# ============================================================
# STORE RANKING
# ============================================================

store_insights = store_insights.sort_values(
    "forecast_30d_units",
    ascending=False
)

store_insights["demand_rank"] = (
    store_insights["forecast_30d_units"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# STORE PRIORITY
# ============================================================

q75 = store_insights[
    "forecast_30d_units"
].quantile(0.75)

q50 = store_insights[
    "forecast_30d_units"
].quantile(0.50)

store_insights["priority"] = np.select(
    [
        store_insights["forecast_30d_units"] >= q75,
        store_insights["forecast_30d_units"] >= q50
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


save_csv(
    store_insights,
    "store_demand_insights.csv"
)


# ============================================================
# TOP STORES
# ============================================================

print()
print("TOP 10 STORES BY 30-DAY FORECAST")

print(
    store_insights[
        [
            "store_id",
            "forecast_30d_units",
            "forecast_60d_units",
            "forecast_90d_units",
            "priority"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SKU-LEVEL BUSINESS INSIGHTS
# ============================================================

section("CREATING SKU-LEVEL BUSINESS INSIGHTS")


sku_30 = (
    forecast_30
    .groupby("sku_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_30d_units"
        }
    )
)

sku_60 = (
    forecast_60
    .groupby("sku_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_60d_units"
        }
    )
)

sku_90 = (
    forecast_90
    .groupby("sku_id", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_90d_units"
        }
    )
)


# ============================================================
# MERGE SKU FORECASTS
# ============================================================

sku_insights = sku_30.merge(
    sku_60,
    on="sku_id",
    how="outer"
)

sku_insights = sku_insights.merge(
    sku_90,
    on="sku_id",
    how="outer"
)

sku_insights = sku_insights.fillna(0)


# ============================================================
# SKU GROWTH
# ============================================================

sku_insights["growth_30_to_60_pct"] = np.where(
    sku_insights["forecast_30d_units"] > 0,
    (
        (
            sku_insights["forecast_60d_units"]
            / sku_insights["forecast_30d_units"]
        ) - 1
    ) * 100,
    0
)

sku_insights["growth_60_to_90_pct"] = np.where(
    sku_insights["forecast_60d_units"] > 0,
    (
        (
            sku_insights["forecast_90d_units"]
            / sku_insights["forecast_60d_units"]
        ) - 1
    ) * 100,
    0
)


# ============================================================
# SKU RANK
# ============================================================

sku_insights = sku_insights.sort_values(
    "forecast_30d_units",
    ascending=False
)

sku_insights["demand_rank"] = (
    sku_insights["forecast_30d_units"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# SKU PRIORITY
# ============================================================

sku_q75 = sku_insights[
    "forecast_30d_units"
].quantile(0.75)

sku_q50 = sku_insights[
    "forecast_30d_units"
].quantile(0.50)

sku_insights["priority"] = np.select(
    [
        sku_insights["forecast_30d_units"] >= sku_q75,
        sku_insights["forecast_30d_units"] >= sku_q50
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


save_csv(
    sku_insights,
    "sku_demand_insights.csv"
)


# ============================================================
# TOP SKUs
# ============================================================

print()
print("TOP 20 SKUs BY 30-DAY FORECAST")

print(
    sku_insights[
        [
            "sku_id",
            "forecast_30d_units",
            "forecast_60d_units",
            "forecast_90d_units",
            "priority"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# STORE-SKU 30-DAY DEMAND
# ============================================================

section("CREATING STORE-SKU DEMAND PRIORITIES")

store_sku_30 = (
    forecast_30
    .groupby(
        ["store_id", "sku_id"],
        as_index=False
    )
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_30d_units"
        }
    )
)


# ============================================================
# STORE-SKU 60-DAY
# ============================================================

store_sku_60 = (
    forecast_60
    .groupby(
        ["store_id", "sku_id"],
        as_index=False
    )
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_60d_units"
        }
    )
)


# ============================================================
# STORE-SKU 90-DAY
# ============================================================

store_sku_90 = (
    forecast_90
    .groupby(
        ["store_id", "sku_id"],
        as_index=False
    )
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "forecast_90d_units"
        }
    )
)


# ============================================================
# MERGE
# ============================================================

store_sku = store_sku_30.merge(
    store_sku_60,
    on=["store_id", "sku_id"],
    how="outer"
)

store_sku = store_sku.merge(
    store_sku_90,
    on=["store_id", "sku_id"],
    how="outer"
)

store_sku = store_sku.fillna(0)


# ============================================================
# DEMAND GROWTH
# ============================================================

store_sku["growth_30_to_60_pct"] = np.where(
    store_sku["forecast_30d_units"] > 0,
    (
        (
            store_sku["forecast_60d_units"]
            / store_sku["forecast_30d_units"]
        ) - 1
    ) * 100,
    0
)


# ============================================================
# STORE-SKU RANK
# ============================================================

store_sku = store_sku.sort_values(
    "forecast_30d_units",
    ascending=False
)

store_sku["demand_rank"] = (
    store_sku["forecast_30d_units"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)


# ============================================================
# PRIORITY SCORE
# ============================================================

store_sku_q90 = store_sku[
    "forecast_30d_units"
].quantile(0.90)

store_sku_q75 = store_sku[
    "forecast_30d_units"
].quantile(0.75)

store_sku_q50 = store_sku[
    "forecast_30d_units"
].quantile(0.50)


store_sku["priority"] = np.select(
    [
        store_sku["forecast_30d_units"] >= store_sku_q90,
        store_sku["forecast_30d_units"] >= store_sku_q75,
        store_sku["forecast_30d_units"] >= store_sku_q50
    ],
    [
        "CRITICAL",
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)


# ============================================================
# BUSINESS ACTION
# ============================================================

store_sku["recommended_action"] = np.select(
    [
        store_sku["priority"] == "CRITICAL",
        store_sku["priority"] == "HIGH",
        store_sku["priority"] == "MEDIUM"
    ],
    [
        "Prioritize replenishment",
        "Increase inventory monitoring",
        "Monitor demand"
    ],
    default="Low priority / review stock"
)


save_csv(
    store_sku,
    "store_sku_demand_priorities.csv"
)


# ============================================================
# HIGH-DEMAND ITEMS
# ============================================================

section("CREATING HIGH-DEMAND ITEMS")

high_demand = store_sku[
    store_sku["priority"].isin(
        ["CRITICAL", "HIGH"]
    )
].copy()

high_demand = high_demand.head(100)

save_csv(
    high_demand,
    "high_priority_demand_items.csv"
)


# ============================================================
# LOW-DEMAND ITEMS
# ============================================================

section("CREATING LOW-DEMAND ITEMS")

low_demand = store_sku[
    store_sku["priority"] == "LOW"
].copy()

low_demand = low_demand.sort_values(
    "forecast_30d_units",
    ascending=True
)

low_demand = low_demand.head(100)

save_csv(
    low_demand,
    "low_demand_items.csv"
)


# ============================================================
# DAILY DEMAND TREND
# ============================================================

section("ANALYZING DAILY FORECAST TREND")

daily = (
    forecast_30
    .groupby("date", as_index=False)
    ["forecast_units"]
    .sum()
    .rename(
        columns={
            "forecast_units":
            "daily_forecast_units"
        }
    )
)

daily["rolling_7d_forecast"] = (
    daily["daily_forecast_units"]
    .rolling(7, min_periods=1)
    .mean()
)

save_csv(
    daily,
    "daily_forecast_business_trend.csv"
)


# ============================================================
# DEMAND CONCENTRATION
# ============================================================

section("DEMAND CONCENTRATION ANALYSIS")

total_30d = (
    store_sku["forecast_30d_units"]
    .sum()
)

top_10_pct_count = max(
    1,
    int(len(store_sku) * 0.10)
)

top_10_items = store_sku.head(
    top_10_pct_count
)

top_10_demand = (
    top_10_items["forecast_30d_units"]
    .sum()
)

concentration_pct = (
    top_10_demand / total_30d * 100
    if total_30d > 0
    else 0
)

print(
    f"Top 10% Store-SKU combinations account for "
    f"{concentration_pct:.2f}% of 30-day forecast demand."
)


# ============================================================
# FORECAST HORIZON SUMMARY
# ============================================================

section("FORECAST HORIZON SUMMARY")

horizon_summary = pd.DataFrame(
    {
        "horizon": [
            "30D",
            "60D",
            "90D"
        ],
        "total_forecast_units": [
            forecast_30["forecast_units"].sum(),
            forecast_60["forecast_units"].sum(),
            forecast_90["forecast_units"].sum()
        ]
    }
)

horizon_summary[
    "average_daily_forecast"
] = [
    horizon_summary.loc[0, "total_forecast_units"] / 30,
    horizon_summary.loc[1, "total_forecast_units"] / 60,
    horizon_summary.loc[2, "total_forecast_units"] / 90
]

save_csv(
    horizon_summary,
    "forecast_horizon_business_summary.csv"
)


# ============================================================
# BUSINESS REPORT
# ============================================================

section("CREATING BUSINESS INSIGHTS REPORT")

report_path = (
    OUTPUT_PATH /
    "business_insights_report.txt"
)

top_store = store_insights.iloc[0]

top_sku = sku_insights.iloc[0]

critical_count = (
    store_sku["priority"]
    .eq("CRITICAL")
    .sum()
)

high_count = (
    store_sku["priority"]
    .eq("HIGH")
    .sum()
)

low_count = (
    store_sku["priority"]
    .eq("LOW")
    .sum()
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT FORESIGHT\n"
    )

    f.write(
        "PHASE 6.2 - BUSINESS INSIGHTS REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "FORECAST SUMMARY\n"
    )

    f.write(
        f"30-day forecast: "
        f"{forecast_30['forecast_units'].sum():,.2f} units\n"
    )

    f.write(
        f"60-day forecast: "
        f"{forecast_60['forecast_units'].sum():,.2f} units\n"
    )

    f.write(
        f"90-day forecast: "
        f"{forecast_90['forecast_units'].sum():,.2f} units\n\n"
    )

    f.write(
        "TOP STORE\n"
    )

    f.write(
        f"Store ID: {top_store['store_id']}\n"
    )

    f.write(
        f"30-day forecast: "
        f"{top_store['forecast_30d_units']:,.2f} units\n\n"
    )

    f.write(
        "TOP SKU\n"
    )

    f.write(
        f"SKU ID: {top_sku['sku_id']}\n"
    )

    f.write(
        f"30-day forecast: "
        f"{top_sku['forecast_30d_units']:,.2f} units\n\n"
    )

    f.write(
        "STORE-SKU PRIORITY\n"
    )

    f.write(
        f"Critical combinations: {critical_count}\n"
    )

    f.write(
        f"High-priority combinations: {high_count}\n"
    )

    f.write(
        f"Low-priority combinations: {low_count}\n\n"
    )

    f.write(
        "DEMAND CONCENTRATION\n"
    )

    f.write(
        f"Top 10% Store-SKU combinations represent "
        f"{concentration_pct:.2f}% of 30-day forecast demand.\n\n"
    )

    f.write(
        "BUSINESS RECOMMENDATIONS\n"
    )

    f.write(
        "1. Prioritize replenishment for critical Store-SKU combinations.\n"
    )

    f.write(
        "2. Closely monitor high-demand stores and SKUs.\n"
    )

    f.write(
        "3. Review low-demand combinations before allocating additional inventory.\n"
    )

    f.write(
        "4. Use the 30-day forecast for short-term replenishment planning.\n"
    )

    f.write(
        "5. Use the 60-day and 90-day forecasts for medium-term capacity and inventory planning.\n"
    )

print(
    f"Business report saved:\n{report_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

section("PHASE 6.2 COMPLETED")

print(
    "Business insights analysis completed successfully."
)

print()
print(
    "Outputs saved to:"
)

print(
    OUTPUT_PATH
)

print()
print(
    "Key outputs:"
)

print(
    "1. store_demand_insights.csv"
)

print(
    "2. sku_demand_insights.csv"
)

print(
    "3. store_sku_demand_priorities.csv"
)

print(
    "4. high_priority_demand_items.csv"
)

print(
    "5. low_demand_items.csv"
)

print(
    "6. daily_forecast_business_trend.csv"
)

print(
    "7. forecast_horizon_business_summary.csv"
)

print(
    "8. business_insights_report.txt"
)

print()
print(
    "NEXT PHASE: INVENTORY RECOMMENDATIONS"
)