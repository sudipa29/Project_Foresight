# ============================================================
# PROJECT FORESIGHT
# Phase 6.7 - FINAL FORECAST QUALITY CHECK
#
# Production Model:
# CALIBRATED INTERMITTENT MODEL
#
# IMPORTANT:
# This version validates:
#     calibrated_forecast_units
#
# NOT:
#     forecast_units
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
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
    / "calibrated"
)

REGIME_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "validation"
    / "store_sku_demand_regimes.csv"
)

OUTPUT_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "validation"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.7 - FINAL FORECAST QUALITY CHECK")
print("=" * 70)


# ============================================================
# FORECAST FILES
# ============================================================

forecast_files = {
    30: FORECAST_PATH / "calibrated_intermittent_30_day_forecast.csv",
    60: FORECAST_PATH / "calibrated_intermittent_60_day_forecast.csv",
    90: FORECAST_PATH / "calibrated_intermittent_90_day_forecast.csv"
}


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING CALIBRATED FORECAST FILES")
print("=" * 70)

for horizon, path in forecast_files.items():

    print()
    print(f"Checking {horizon}-day forecast:")
    print(path)

    if path.exists():
        print("FOUND")
    else:
        raise FileNotFoundError(
            f"Missing forecast file:\n{path}"
        )


# ============================================================
# LOAD DEMAND REGIMES
# ============================================================

print()
print("=" * 70)
print("LOADING DEMAND REGIMES")
print("=" * 70)

regimes = pd.read_csv(REGIME_PATH)

print("Regime shape:", regimes.shape)

print()
print("Demand regime counts:")
print(
    regimes["demand_regime"].value_counts()
)


# ============================================================
# EXPECTED STORE-SKU COUNT
# ============================================================

expected_store_sku_count = (
    regimes[["store_id", "sku_id"]]
    .drop_duplicates()
    .shape[0]
)

print()
print(
    "Expected Store-SKU combinations:",
    expected_store_sku_count
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_forecast(
    df,
    horizon,
    regimes
):

    print()
    print("-" * 70)
    print(f"VALIDATING {horizon}-DAY CALIBRATED FORECAST")
    print("-" * 70)

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "store_id",
        "sku_id",
        "date",
        "demand_regime",
        "calibration_factor",
        "occurrence_probability",
        "positive_demand_quantity",
        "forecast_units",
        "calibrated_forecast_units"
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    print()
    print("Date range:")
    print(df["date"].min())
    print("to")
    print(df["date"].max())

    unique_dates = df["date"].nunique()

    print(
        "Unique dates:",
        unique_dates
    )

    # --------------------------------------------------------
    # STORE-SKU
    # --------------------------------------------------------

    store_sku_count = (
        df[["store_id", "sku_id"]]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Store-SKU combinations:",
        store_sku_count
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_rows = df.duplicated(
        subset=[
            "store_id",
            "sku_id",
            "date"
        ]
    ).sum()

    print(
        "Duplicate Store-SKU-Date rows:",
        duplicate_rows
    )

    # --------------------------------------------------------
    # MISSING FORECAST
    # --------------------------------------------------------

    missing_forecasts = (
        df["calibrated_forecast_units"]
        .isna()
        .sum()
    )

    print(
        "Missing calibrated forecasts:",
        missing_forecasts
    )

    # --------------------------------------------------------
    # NEGATIVE FORECAST
    # --------------------------------------------------------

    negative_forecasts = (
        df["calibrated_forecast_units"] < 0
    ).sum()

    print(
        "Negative calibrated forecasts:",
        negative_forecasts
    )

    # --------------------------------------------------------
    # ZERO FORECAST
    # --------------------------------------------------------

    zero_forecasts = (
        df["calibrated_forecast_units"] == 0
    ).sum()

    positive_forecasts = (
        df["calibrated_forecast_units"] > 0
    ).sum()

    print(
        "Zero calibrated forecast rows:",
        zero_forecasts
    )

    print(
        "Positive calibrated forecast rows:",
        positive_forecasts
    )

    # --------------------------------------------------------
    # TOTAL FORECAST
    # --------------------------------------------------------

    total_forecast = (
        df["calibrated_forecast_units"]
        .sum()
    )

    avg_daily_forecast = (
        df.groupby("date")[
            "calibrated_forecast_units"
        ]
        .sum()
        .mean()
    )

    min_daily_forecast = (
        df.groupby("date")[
            "calibrated_forecast_units"
        ]
        .sum()
        .min()
    )

    max_daily_forecast = (
        df.groupby("date")[
            "calibrated_forecast_units"
        ]
        .sum()
        .max()
    )

    print()
    print(
        "TOTAL CALIBRATED FORECAST:",
        f"{total_forecast:,.2f}"
    )

    print(
        "Average daily calibrated forecast:",
        f"{avg_daily_forecast:,.2f}"
    )

    print(
        "Minimum daily calibrated forecast:",
        f"{min_daily_forecast:,.2f}"
    )

    print(
        "Maximum daily calibrated forecast:",
        f"{max_daily_forecast:,.2f}"
    )

    # --------------------------------------------------------
    # TOP STORES
    # --------------------------------------------------------

    print()
    print("Top 5 stores:")

    top_stores = (
        df.groupby("store_id")[
            "calibrated_forecast_units"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(5)
    )

    print(top_stores)

    # --------------------------------------------------------
    # TOP SKUS
    # --------------------------------------------------------

    print()
    print("Top 5 SKUs:")

    top_skus = (
        df.groupby("sku_id")[
            "calibrated_forecast_units"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(5)
    )

    print(top_skus)

    # --------------------------------------------------------
    # DEMAND REGIME CHECK
    # --------------------------------------------------------

    missing_regimes = (
        df["demand_regime"]
        .isna()
        .sum()
    )

    print()
    print(
        "Missing demand regimes:",
        missing_regimes
    )

    # --------------------------------------------------------
    # REGIME SUMMARY
    # --------------------------------------------------------

    regime_summary = (
        df.groupby("demand_regime")[
            "calibrated_forecast_units"
        ]
        .agg([
            "count",
            "sum",
            "mean",
            "min",
            "max"
        ])
    )

    print()
    print(
        "Calibrated regime forecast summary:"
    )

    print(regime_summary)

    # --------------------------------------------------------
    # EXPECTED REGIME CALIBRATION CHECK
    # --------------------------------------------------------

    print()
    print(
        "Checking calibration factors..."
    )

    calibration_check = (
        df.groupby("demand_regime")[
            "calibration_factor"
        ]
        .mean()
    )

    print(calibration_check)

    # --------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------

    return {
        "horizon_days": horizon,
        "rows": len(df),
        "unique_dates": unique_dates,
        "store_sku_count": store_sku_count,
        "duplicate_rows": duplicate_rows,
        "missing_forecasts": missing_forecasts,
        "negative_forecasts": negative_forecasts,
        "zero_forecast_rows": zero_forecasts,
        "positive_forecast_rows": positive_forecasts,
        "total_forecast_units": total_forecast,
        "avg_daily_forecast": avg_daily_forecast,
        "min_daily_forecast": min_daily_forecast,
        "max_daily_forecast": max_daily_forecast,
        "missing_regimes": missing_regimes
    }


# ============================================================
# RUN VALIDATION
# ============================================================

results = []

for horizon, path in forecast_files.items():

    df = pd.read_csv(path)

    result = validate_forecast(
        df,
        horizon,
        regimes
    )

    results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(results)


print()
print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


# ============================================================
# QUALITY CONTROL
# ============================================================

print()
print("=" * 70)
print("QUALITY CONTROL DECISION")
print("=" * 70)


# ------------------------------------------------------------
# TEST 1: ALL FILES
# ------------------------------------------------------------

files_pass = all(
    path.exists()
    for path in forecast_files.values()
)

print(
    f"{'30/60/90 calibrated files':35}: "
    f"{'PASS' if files_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# TEST 2: STORE-SKU
# ------------------------------------------------------------

store_sku_pass = all(
    summary["store_sku_count"]
    == expected_store_sku_count
)

print(
    f"{'10,000 Store-SKU combinations':35}: "
    f"{'PASS' if store_sku_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# TEST 3: DUPLICATES
# ------------------------------------------------------------

duplicate_pass = (
    summary["duplicate_rows"].sum()
    == 0
)

print(
    f"{'No duplicate rows':35}: "
    f"{'PASS' if duplicate_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# TEST 4: MISSING FORECASTS
# ------------------------------------------------------------

missing_pass = (
    summary["missing_forecasts"].sum()
    == 0
)

print(
    f"{'No missing calibrated forecasts':35}: "
    f"{'PASS' if missing_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# TEST 5: NEGATIVE FORECASTS
# ------------------------------------------------------------

negative_pass = (
    summary["negative_forecasts"].sum()
    == 0
)

print(
    f"{'No negative calibrated forecasts':35}: "
    f"{'PASS' if negative_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# TEST 6: REGIMES
# ------------------------------------------------------------

regime_pass = (
    summary["missing_regimes"].sum()
    == 0
)

print(
    f"{'Regimes complete':35}: "
    f"{'PASS' if regime_pass else 'FAIL'}"
)


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

overall_pass = all([
    files_pass,
    store_sku_pass,
    duplicate_pass,
    missing_pass,
    negative_pass,
    regime_pass
])


# ============================================================
# SAVE SUMMARY
# ============================================================

output_file = (
    OUTPUT_PATH
    / "final_forecast_quality_check.csv"
)

summary.to_csv(
    output_file,
    index=False
)

print()
print("Saved:")
print(output_file)


# ============================================================
# FINAL DECISION
# ============================================================

print()
print("=" * 70)

if overall_pass:

    print(
        "FINAL FORECAST QUALITY CHECK: PASS"
    )

    print()
    print(
        "Production forecast:"
    )

    print(
        "CALIBRATED INTERMITTENT MODEL"
    )

    print()
    print(
        "Validated forecast column:"
    )

    print(
        "calibrated_forecast_units"
    )

    print()
    print(
        "Ready for:"
    )

    print(
        "PHASE 7 - FORECAST + INVENTORY INTEGRATION"
    )

else:

    print(
        "FINAL FORECAST QUALITY CHECK: FAIL"
    )

    print()
    print(
        "DO NOT proceed to Phase 7."
    )

print("=" * 70)