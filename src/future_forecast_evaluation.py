# ============================================================
# PROJECT FORESIGHT
# Phase 6.1 - Future Forecast Evaluation
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

FUTURE_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
)

OUTPUT_PATH = FUTURE_PATH / "evaluation"

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_FILES = {
    "30D": FUTURE_PATH / "future_30_day_forecast.csv",
    "60D": FUTURE_PATH / "future_60_day_forecast.csv",
    "90D": FUTURE_PATH / "future_90_day_forecast.csv",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_wape(actual, forecast):
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return (
        np.abs(actual - forecast).sum()
        / denominator
        * 100
    )


def safe_bias(actual, forecast):
    return np.mean(forecast - actual)


def safe_mae(actual, forecast):
    return np.mean(
        np.abs(actual - forecast)
    )


def safe_rmse(actual, forecast):
    return np.sqrt(
        np.mean((actual - forecast) ** 2)
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FUTURE FORECAST EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print_section("CHECKING REQUIRED FILES")

for horizon, file_path in FORECAST_FILES.items():

    if file_path.exists():

        print(f"PASS: {horizon} forecast found")
        print(f"      {file_path}")

    else:

        print(f"FAIL: {horizon} forecast missing")
        print(f"      {file_path}")

        raise FileNotFoundError(
            f"Required forecast file not found: {file_path}"
        )


# ============================================================
# LOAD FORECAST FILES
# ============================================================

print_section("LOADING FUTURE FORECASTS")


forecast_data = {}


for horizon, file_path in FORECAST_FILES.items():

    print()
    print(f"Loading {horizon} forecast...")

    df = pd.read_csv(
        file_path,
        usecols=[
            "store_id",
            "sku_id",
            "date",
            "forecast_units"
        ]
    )

    print(f"Rows loaded: {len(df):,}")

    print(
        f"Columns: {df.columns.tolist()}"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["forecast_units"] = pd.to_numeric(
        df["forecast_units"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    invalid_dates = df["date"].isna().sum()
    missing_forecasts = df["forecast_units"].isna().sum()
    negative_forecasts = (
        df["forecast_units"] < 0
    ).sum()

    print(
        f"Invalid dates: {invalid_dates:,}"
    )

    print(
        f"Missing forecast values: {missing_forecasts:,}"
    )

    print(
        f"Negative forecast values: {negative_forecasts:,}"
    )

    # --------------------------------------------------------
    # Replace invalid negative predictions
    # --------------------------------------------------------

    df["forecast_units"] = (
        df["forecast_units"]
        .clip(lower=0)
    )

    forecast_data[horizon] = df


# ============================================================
# BASIC FORECAST SUMMARY
# ============================================================

print_section("FORECAST SUMMARY")


summary_rows = []


for horizon, df in forecast_data.items():

    summary_rows.append({

        "horizon": horizon,

        "start_date": df["date"].min(),

        "end_date": df["date"].max(),

        "forecast_days":
            df["date"].nunique(),

        "stores":
            df["store_id"].nunique(),

        "skus":
            df["sku_id"].nunique(),

        "store_sku_combinations":
            df[
                ["store_id", "sku_id"]
            ].drop_duplicates().shape[0],

        "rows":
            len(df),

        "total_forecast_units":
            df["forecast_units"].sum(),

        "average_daily_forecast":
            df.groupby("date")[
                "forecast_units"
            ].sum().mean(),

        "minimum_forecast":
            df["forecast_units"].min(),

        "maximum_forecast":
            df["forecast_units"].max(),

        "median_forecast":
            df["forecast_units"].median(),

    })


forecast_summary = pd.DataFrame(
    summary_rows
)


print(
    forecast_summary.to_string(
        index=False
    )
)


# ============================================================
# FORECAST CONSISTENCY CHECK
# ============================================================

print_section("FORECAST CONSISTENCY CHECK")


for horizon, df in forecast_data.items():

    expected_days = {
        "30D": 30,
        "60D": 60,
        "90D": 90
    }[horizon]

    actual_days = df["date"].nunique()

    expected_combinations = (
        df[
            ["store_id", "sku_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_rows = (
        expected_days
        * expected_combinations
    )

    actual_rows = len(df)

    print()
    print(f"{horizon}:")

    print(
        f"Expected days: {expected_days}"
    )

    print(
        f"Actual days:   {actual_days}"
    )

    print(
        f"Store-SKU combinations: "
        f"{expected_combinations:,}"
    )

    print(
        f"Expected rows: {expected_rows:,}"
    )

    print(
        f"Actual rows:   {actual_rows:,}"
    )

    if (
        actual_days == expected_days
        and actual_rows == expected_rows
    ):

        print("Status: PASS")

    else:

        print("Status: CHECK")


# ============================================================
# 30 / 60 / 90 DAY TOTAL COMPARISON
# ============================================================

print_section("30 / 60 / 90 DAY FORECAST COMPARISON")


comparison = (
    forecast_summary[
        [
            "horizon",
            "total_forecast_units",
            "average_daily_forecast",
            "minimum_forecast",
            "maximum_forecast"
        ]
    ]
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# DAILY FORECAST TREND
# ============================================================

print_section("CREATING DAILY FORECAST TREND")


daily_rows = []


for horizon, df in forecast_data.items():

    daily = (
        df.groupby("date", as_index=False)
        ["forecast_units"]
        .sum()
    )

    daily["horizon"] = horizon

    daily_rows.append(
        daily
    )


daily_forecast = pd.concat(
    daily_rows,
    ignore_index=True
)


daily_forecast = daily_forecast[
    [
        "horizon",
        "date",
        "forecast_units"
    ]
]


daily_forecast.to_csv(
    OUTPUT_PATH
    / "future_daily_forecast_summary.csv",
    index=False
)


print(
    "Daily forecast summary saved:"
)

print(
    OUTPUT_PATH
    / "future_daily_forecast_summary.csv"
)


# ============================================================
# STORE LEVEL ANALYSIS
# ============================================================

print_section("CREATING STORE-LEVEL EVALUATION")


store_rows = []


for horizon, df in forecast_data.items():

    store_summary = (
        df.groupby("store_id")
        .agg(
            total_forecast_units=(
                "forecast_units",
                "sum"
            ),
            average_forecast_units=(
                "forecast_units",
                "mean"
            ),
            max_daily_forecast=(
                "forecast_units",
                "max"
            ),
            active_forecast_days=(
                "forecast_units",
                lambda x: (x > 0).sum()
            )
        )
        .reset_index()
    )

    store_summary["horizon"] = horizon

    store_rows.append(
        store_summary
    )


store_evaluation = pd.concat(
    store_rows,
    ignore_index=True
)


store_evaluation = store_evaluation[
    [
        "horizon",
        "store_id",
        "total_forecast_units",
        "average_forecast_units",
        "max_daily_forecast",
        "active_forecast_days"
    ]
]


store_evaluation.to_csv(
    OUTPUT_PATH
    / "store_forecast_evaluation.csv",
    index=False
)


print(
    "Store evaluation saved:"
)

print(
    OUTPUT_PATH
    / "store_forecast_evaluation.csv"
)


# ============================================================
# SKU LEVEL ANALYSIS
# ============================================================

print_section("CREATING SKU-LEVEL EVALUATION")


sku_rows = []


for horizon, df in forecast_data.items():

    sku_summary = (
        df.groupby("sku_id")
        .agg(
            total_forecast_units=(
                "forecast_units",
                "sum"
            ),
            average_forecast_units=(
                "forecast_units",
                "mean"
            ),
            max_daily_forecast=(
                "forecast_units",
                "max"
            ),
            active_forecast_days=(
                "forecast_units",
                lambda x: (x > 0).sum()
            )
        )
        .reset_index()
    )

    sku_summary["horizon"] = horizon

    sku_rows.append(
        sku_summary
    )


sku_evaluation = pd.concat(
    sku_rows,
    ignore_index=True
)


sku_evaluation = sku_evaluation[
    [
        "horizon",
        "sku_id",
        "total_forecast_units",
        "average_forecast_units",
        "max_daily_forecast",
        "active_forecast_days"
    ]
]


sku_evaluation.to_csv(
    OUTPUT_PATH
    / "sku_forecast_evaluation.csv",
    index=False
)


print(
    "SKU evaluation saved:"
)

print(
    OUTPUT_PATH
    / "sku_forecast_evaluation.csv"
)


# ============================================================
# STORE-SKU ANALYSIS
# ============================================================

print_section(
    "CREATING STORE-SKU FORECAST EVALUATION"
)


storesku_rows = []


for horizon, df in forecast_data.items():

    storesku_summary = (
        df.groupby(
            [
                "store_id",
                "sku_id"
            ]
        )
        .agg(
            total_forecast_units=(
                "forecast_units",
                "sum"
            ),
            average_forecast_units=(
                "forecast_units",
                "mean"
            ),
            max_daily_forecast=(
                "forecast_units",
                "max"
            ),
            active_forecast_days=(
                "forecast_units",
                lambda x: (x > 0).sum()
            )
        )
        .reset_index()
    )

    storesku_summary["horizon"] = horizon

    storesku_rows.append(
        storesku_summary
    )


storesku_evaluation = pd.concat(
    storesku_rows,
    ignore_index=True
)


storesku_evaluation = storesku_evaluation[
    [
        "horizon",
        "store_id",
        "sku_id",
        "total_forecast_units",
        "average_forecast_units",
        "max_daily_forecast",
        "active_forecast_days"
    ]
]


storesku_evaluation.to_csv(
    OUTPUT_PATH
    / "store_sku_forecast_evaluation.csv",
    index=False
)


print(
    "Store-SKU evaluation saved:"
)

print(
    OUTPUT_PATH
    / "store_sku_forecast_evaluation.csv"
)


# ============================================================
# TOP HIGH-DEMAND STORE-SKU
# ============================================================

print_section(
    "IDENTIFYING HIGH-DEMAND STORE-SKU COMBINATIONS"
)


top_30 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "30D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=False
    )
    .head(50)
)


top_60 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "60D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=False
    )
    .head(50)
)


top_90 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "90D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=False
    )
    .head(50)
)


top_30.to_csv(
    OUTPUT_PATH
    / "top_50_high_demand_30D.csv",
    index=False
)


top_60.to_csv(
    OUTPUT_PATH
    / "top_50_high_demand_60D.csv",
    index=False
)


top_90.to_csv(
    OUTPUT_PATH
    / "top_50_high_demand_90D.csv",
    index=False
)


print(
    "Top 50 high-demand files created."
)


# ============================================================
# LOW-DEMAND STORE-SKU
# ============================================================

print_section(
    "IDENTIFYING LOW-DEMAND STORE-SKU COMBINATIONS"
)


low_30 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "30D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=True
    )
    .head(50)
)


low_60 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "60D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=True
    )
    .head(50)
)


low_90 = (
    storesku_evaluation[
        storesku_evaluation["horizon"] == "90D"
    ]
    .sort_values(
        "total_forecast_units",
        ascending=True
    )
    .head(50)
)


low_30.to_csv(
    OUTPUT_PATH
    / "top_50_low_demand_30D.csv",
    index=False
)


low_60.to_csv(
    OUTPUT_PATH
    / "top_50_low_demand_60D.csv",
    index=False
)


low_90.to_csv(
    OUTPUT_PATH
    / "top_50_low_demand_90D.csv",
    index=False
)


print(
    "Top 50 low-demand files created."
)


# ============================================================
# FORECAST GROWTH ANALYSIS
# ============================================================

print_section(
    "FORECAST HORIZON GROWTH ANALYSIS"
)


total_30 = forecast_summary.loc[
    forecast_summary["horizon"] == "30D",
    "total_forecast_units"
].iloc[0]


total_60 = forecast_summary.loc[
    forecast_summary["horizon"] == "60D",
    "total_forecast_units"
].iloc[0]


total_90 = forecast_summary.loc[
    forecast_summary["horizon"] == "90D",
    "total_forecast_units"
].iloc[0]


growth_30_60 = (
    (total_60 - total_30)
    / total_30
    * 100
)


growth_60_90 = (
    (total_90 - total_60)
    / total_60
    * 100
)


growth_30_90 = (
    (total_90 - total_30)
    / total_30
    * 100
)


growth_summary = pd.DataFrame(
    [
        {
            "comparison": "30D_to_60D",
            "percentage_change": growth_30_60
        },
        {
            "comparison": "60D_to_90D",
            "percentage_change": growth_60_90
        },
        {
            "comparison": "30D_to_90D",
            "percentage_change": growth_30_90
        }
    ]
)


print(
    growth_summary.to_string(
        index=False
    )
)


growth_summary.to_csv(
    OUTPUT_PATH
    / "forecast_horizon_growth.csv",
    index=False
)


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

print_section(
    "BUSINESS INTERPRETATION"
)


daily_30 = total_30 / 30
daily_60 = total_60 / 60
daily_90 = total_90 / 90


print(
    f"30-day total forecast: "
    f"{total_30:,.2f} units"
)

print(
    f"60-day total forecast: "
    f"{total_60:,.2f} units"
)

print(
    f"90-day total forecast: "
    f"{total_90:,.2f} units"
)

print()

print(
    f"30-day average daily demand: "
    f"{daily_30:,.2f} units"
)

print(
    f"60-day average daily demand: "
    f"{daily_60:,.2f} units"
)

print(
    f"90-day average daily demand: "
    f"{daily_90:,.2f} units"
)

print()

print(
    f"30D → 60D change: "
    f"{growth_30_60:.2f}%"
)

print(
    f"60D → 90D change: "
    f"{growth_60_90:.2f}%"
)

print(
    f"30D → 90D change: "
    f"{growth_30_90:.2f}%"
)


# ============================================================
# SAVE MAIN SUMMARY
# ============================================================

forecast_summary.to_csv(
    OUTPUT_PATH
    / "future_forecast_summary.csv",
    index=False
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_PATH
    / "future_forecast_evaluation_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT FORESIGHT - FUTURE FORECAST EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "FINAL PRODUCTION MODEL: LightGBM\n\n"
    )

    f.write(
        f"30-day total forecast: "
        f"{total_30:,.2f}\n"
    )

    f.write(
        f"60-day total forecast: "
        f"{total_60:,.2f}\n"
    )

    f.write(
        f"90-day total forecast: "
        f"{total_90:,.2f}\n\n"
    )

    f.write(
        f"30-day average daily forecast: "
        f"{daily_30:,.2f}\n"
    )

    f.write(
        f"60-day average daily forecast: "
        f"{daily_60:,.2f}\n"
    )

    f.write(
        f"90-day average daily forecast: "
        f"{daily_90:,.2f}\n\n"
    )

    f.write(
        f"30D to 60D change: "
        f"{growth_30_60:.2f}%\n"
    )

    f.write(
        f"60D to 90D change: "
        f"{growth_60_90:.2f}%\n"
    )

    f.write(
        f"30D to 90D change: "
        f"{growth_30_90:.2f}%\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print_section(
    "PHASE 6.1 COMPLETED"
)


print(
    "Forecast evaluation completed successfully."
)


print()

print(
    "Evaluation outputs saved to:"
)

print(
    OUTPUT_PATH
)


print()

print(
    "Main summary:"
)

print(
    OUTPUT_PATH
    / "future_forecast_summary.csv"
)


print()

print(
    "Daily summary:"
)

print(
    OUTPUT_PATH
    / "future_daily_forecast_summary.csv"
)


print()

print(
    "Store evaluation:"
)

print(
    OUTPUT_PATH
    / "store_forecast_evaluation.csv"
)


print()

print(
    "SKU evaluation:"
)

print(
    OUTPUT_PATH
    / "sku_forecast_evaluation.csv"
)


print()

print(
    "Store-SKU evaluation:"
)

print(
    OUTPUT_PATH
    / "store_sku_forecast_evaluation.csv"
)


print()

print(
    "Report:"
)

print(
    report_path
)


print()

print(
    "NEXT PHASE: BUSINESS INSIGHTS & INVENTORY RECOMMENDATIONS"
)